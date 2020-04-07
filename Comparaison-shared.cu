#include <cstdio>
#include <cmath>
#include <random>
#include <ctime>
#include <iostream>



// Merge sort algorithm without shared memory
__global__
void merge_sort(int *a, int *a_new, int length_a, int nb_loop){

  // Constant initialization
  int threads_per_block = blockDim.x;
  int i_th = threadIdx.x;
  int a_begin = blockIdx.x * nb_loop * threads_per_block;
  int b_begin = blockIdx.x * nb_loop * threads_per_block + length_a;


  // Loop if size of array >= 1024
  for (int j = 0; j < nb_loop ; j++) {

    int i_thread = i_th + j*threads_per_block;
    int Kx=0, Ky=0, Px=0, Py=0, Qx=0, Qy=0, offset=0;


    // Algorithm for 2 sorted arrays
    if (i_thread > length_a) {
      Kx = i_thread - length_a, Ky = length_a;
      Px = length_a, Py = i_thread - length_a;
    }
    else {
      Ky = i_thread, Px = i_thread;
    }



    while (true) {
      offset = abs(Ky - Py)/2;
      Qx = Kx + offset;
      Qy = Ky - offset;

      if ( (Qy >= 0 ) && (Qx <= length_a) && ((Qy == length_a) || (Qx == 0) || (a[a_begin+Qy] > a[b_begin+Qx-1]))) {
        if ((Qx == length_a) || (Qy == 0) || (a[a_begin + Qy - 1] <= a[b_begin+Qx])) {
          if ( (Qy < length_a) && ((Qx == length_a) || (a[a_begin+Qy] <= a[b_begin+Qx]))) {
            a_new[a_begin + i_th + j*threads_per_block] = a[a_begin+Qy];
            //__syncthreads();
          }
          else {
            a_new[a_begin + i_th + j*threads_per_block] = a[b_begin+Qx];
            //__syncthreads();
          }
          break;
        }
        else {
          Kx = Qx + 1;
          Ky = Qy - 1;
        }
      }
      else {
        Px = Qx - 1;
        Py = Qy + 1;
      }


    }


  }

}




// Merge-sort with shared memory (for n <= pow(2,12))
__global__
void merge_shared(int *a, int length_a, int nb_loop){

  // Constant initialization
  int threads_per_block = blockDim.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i_th = threadIdx.x;


  // Shared memory (for each block)
  extern __shared__ int a_union_b[];
  for (int j = 0 ; j < nb_loop ; j++) {
    a_union_b[i_th + j*threads_per_block] = a[i + j*threads_per_block];
    __syncthreads();
  }



  for (int j = 0; j < nb_loop ; j++) {
    int i_thread = i_th + j*threads_per_block;
    int Kx=0, Ky=0, Px=0, Py=0, Qx=0, Qy=0, offset=0;


    // Algorithm for 2 sorted arrays
    if (i_thread > length_a) {
      Kx = i_thread - length_a, Ky = length_a;
      Px = length_a, Py = i_thread - length_a;
    }
    else {
      Ky = i_thread, Px = i_thread;
    }



    while (true) {
      offset = abs(Ky - Py)/2;
      Qx = Kx + offset;
      Qy = Ky - offset;

      if ( (Qy >= 0 ) && (Qx <= length_a) && ((Qy == length_a) || (Qx == 0) || (a_union_b[Qy] > a_union_b[length_a+Qx-1]))) {
        if ((Qx == length_a) || (Qy == 0) || (a_union_b[Qy - 1] <= a_union_b[length_a+Qx])) {
          if ( (Qy < length_a) && ((Qx == length_a) || (a_union_b[Qy] <= a_union_b[length_a+Qx]))) {
            a[blockIdx.x*threads_per_block*nb_loop + i_th + j*threads_per_block] = a_union_b[Qy];
            //__syncthreads();
          }
          else {
            a[blockIdx.x*threads_per_block*nb_loop + i_th + j*threads_per_block] = a_union_b[length_a+Qx];
            //__syncthreads();
          }

          break;
        }
        else {
          Kx = Qx + 1;
          Ky = Qy - 1;
        }
      }
      else {
        Px = Qx - 1;
        Py = Qy + 1;
      }


    }

  }



}





int main(void) {

  // XXXXXXXXXXXXXXXX Initialization XXXXXXXXXXXXXXXXXXXXXXX //
  std::cout << "Enter the power of two of the array : (<= 12)" << std::endl;
  int p=10;
	std::cin >> p;
  const int n = pow(2,p);
  int *a, *a_res; // host
  int *d_a, *d_a_copie; // device
  a = new int[n];
  a_res = new int[n];
  cudaMalloc(&d_a, n * sizeof(int));
  cudaMalloc(&d_a_copie, n * sizeof(int));


  std::mt19937 G(time(NULL));
  std::uniform_int_distribution<int> U(0,n);
  for (int i = 0; i < n; i++) {
    a[i] = U(G);
  }

  cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a_copie, a, n * sizeof(int), cudaMemcpyHostToDevice);

  // XXXXXXXXXXXXXXX Timer and algorithm, without shared memory XXXXXXXXXXXXXXXX

  float Tim1;
	cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventRecord(start1,0);



  int nb_blocs = n/2;
  int nb_threads = 2;
  int size = 1;


  while (size < n) {
    int tf = min(nb_threads, 1024);
    int nb_loop = max(1,nb_threads/1024);
    merge_sort<<<nb_blocs, tf>>>(d_a, d_a_copie, size, nb_loop);
    cudaDeviceSynchronize();
    nb_blocs /= 2;
    nb_threads *= 2;
    size *= 2;
    cudaMemcpy(d_a, d_a_copie, n * sizeof(int), cudaMemcpyDeviceToDevice);
  }



  cudaEventRecord(stop1,0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&Tim1,start1, stop1);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);


  // XXXXXXXXXXXXXXX Timer and algorithm, with shared memory XXXXXXXXXXXXXXXX

  cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

  float Tim2;
	cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2,0);



  nb_blocs = n/2;
  nb_threads = 2;
  size = 1;


  while (size < n) {
    int tf = min(nb_threads, 1024);
    int nb_loop = max(1,nb_threads/1024);
    merge_shared<<<nb_blocs, tf, nb_threads*sizeof(int)>>>(d_a, size, nb_loop);
    cudaDeviceSynchronize();
    nb_blocs /= 2;
    nb_threads *= 2;
    size *= 2;
  }



  cudaEventRecord(stop2,0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&Tim2,start2, stop2);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);




  // XXXXXXXXXXXXXXX Résults XXXXXXXXXXXXXXXX
  cudaMemcpy(a_res, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Execution time for merge sort without shared memory : " << Tim1 << " ms" << std::endl;
  std::cout << "Execution time for merge sort with shared memory : " << Tim2 << " ms" << std::endl;


  delete [] a; delete [] a_res;
  cudaFree(d_a); cudaFree(d_a_copie);

  return 0;

}
