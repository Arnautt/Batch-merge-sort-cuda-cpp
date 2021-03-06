#include <cstdio>
#include <cmath>
#include <random>
#include <ctime>
#include <iostream>



// Take a batch of vectors concatenated in a, result will be in a_new
__global__
void batch_merge(int *a, int *a_new, int length_a, int nb_loop, const int n){

  // Constants initialization
  int dimy = blockIdx.y;
  int threads_per_block = blockDim.x;
  int i_th = threadIdx.x;
  int a_begin = blockIdx.x * nb_loop * threads_per_block;
  int b_begin = blockIdx.x * nb_loop * threads_per_block + length_a;


  // Loop if size of array >= 1024 (max. nb. of threads = 1024)
  for (int j = 0; j < nb_loop ; j++) {

    int i_thread = i_th + j*threads_per_block;
    int Kx=0, Ky=0, Px=0, Py=0, Qx=0, Qy=0, offset=0;


    // Algorithm with two sorted arrays

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
            a_new[dimy*n + a_begin + i_th + j*threads_per_block] = a[a_begin+Qy];
            //__syncthreads();
          }
          else {
            a_new[dimy*n+ a_begin + i_th + j*threads_per_block] = a[b_begin+Qx];
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
  int p=10, nb_array=1;

	std::cout << "Enter the power of two of the array : " << std::endl;
	std::cin >> p;

  std::cout << "Enter the number of array for batch sort : " << std::endl;
	std::cin >> nb_array;

  const int n = pow(2,p);
  int total_size = n*nb_array;

  int *a, *a_res; // host
  int *d_a, *d_a_copie; // device
  a = new int[total_size];
  a_res = new int[total_size];
  cudaMalloc(&d_a, total_size * sizeof(int));
  cudaMalloc(&d_a_copie, total_size * sizeof(int));

  // Batch random vectors to sort
  std::mt19937 G(time(NULL));
  std::uniform_int_distribution<int> U(0,n);
  for (int i = 0; i < total_size; i++) {
    a[i] = U(G);
  }

  cudaMemcpy(d_a, a, total_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a_copie, a, total_size * sizeof(int), cudaMemcpyHostToDevice);

  // GPU timer
  float Tim;
	cudaEvent_t start, stop;
  cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

  // XXXXXXXXXXXXXXXX Algorithm XXXXXXXXXXXXXXXXXXXXXXX //


  int nb_blocs = n/2;
  int nb_threads = 2;
  int size = 1;


  while (size < n) {
    int tf = min(nb_threads, 1024);
    int nb_loop = max(1,nb_threads/1024);
    dim3 nb_blocs2(nb_blocs,nb_array,1);
    batch_merge<<<nb_blocs2, tf>>>(d_a, d_a_copie, size, nb_loop, n);
    cudaDeviceSynchronize();
    nb_blocs /= 2;
    nb_threads *= 2;
    size *= 2;
    cudaMemcpy(d_a, d_a_copie, total_size * sizeof(int), cudaMemcpyDeviceToDevice);
  }


  cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&Tim,start, stop);



  // XXXXXXXXXXXXXXXX Résults XXXXXXXXXXXXXXXXXXXXXXX //

  cudaMemcpy(a_res, d_a, total_size * sizeof(int), cudaMemcpyDeviceToHost);

  // Error verification

  int nb_error = 0;
  for (int array_nb = 0 ; array_nb < nb_array ; array_nb++) {
    for (int i = 0 ; i < n-1 ; i++) {
      if (a_res[i+array_nb*n] > a_res[i+1+array_nb*n]) {nb_error++;}
    }
  }


  std::cout << "Nb of error : " << nb_error << std::endl;
  std::cout << "Size of the array : " << n << std::endl;
  std::cout << "Nb of array : " << nb_array << std::endl;
  std::cout << "Execution time for batch merge sort : " << Tim << " ms" << std::endl;
  delete [] a;
  delete [] a_res;
  cudaFree(d_a); cudaFree(d_a_copie);
  cudaEventDestroy(start); cudaEventDestroy(stop);

  return 0;

}
