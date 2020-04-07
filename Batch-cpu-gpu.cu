#include <cstdio>
#include <cmath>
#include <random>
#include <ctime>
#include <iostream>




void merge_sorted(int *a, int *b, int *m, int length_a, int length_b){
	// a, b = 2 sorted arrays
	// m will contain the result

	int i=0, j=0;
	while (i+j < length_a + length_b) {
		if (i >= length_a) {
			m[i+j] = b[j];
			j++;
		}
		else if (j >= length_b || a[i] < b[j]) {
			m[i+j] = a[i];
			i++;
		}
		else {
			m[i+j] = b[j];
			j++;
		}

	}
}



void merge(int *a, int left_low, int left_high, int right_low, int right_high) {
    // Size of vectors
    int length = right_high-left_low+1;
    int length_left = left_high - left_low + 1;
    int length_right = right_high - right_low + 1;

    // Temporary vectors
    int * temp_right, * temp_left, * temp;
    temp_right = new int[length_right];
    temp_left = new int[length_left];
    temp = new int[length];

		// Fill with already sorted value of a
    for (int i = 0 ; i < length_right ; i++) {
      temp_right[i] = a[right_low+i];
    }
    for (int i = 0 ; i < length_left ; i++) {
      temp_left[i] = a[left_low+i];
    }

		// We use function for sorted arrays
		// Reult (temp_left and temp_right merged, sorted) will be in temp
    merge_sorted(temp_left, temp_right, temp, length_left, length_right);

		// Fill a with temp to reuse what we did
    for (int i=0; i< length; ++i){
      a[left_low++] = temp[i];
    }


    delete [] temp_right;
    delete [] temp_left;
    delete [] temp;
}



void merge_sort(int *a, int low, int high) {
    // Recursive function to sort a on CPU
    // Starting point : two single value
    if (low >= high)
      return;
    else {
      int mid = (low + high)/2;       // Middle of array
      merge_sort(a, low, mid);        // Sort first part from low to mid
      merge_sort(a, mid+1, high);     // Sort second part from mid+1 to high
      merge(a, low, mid, mid+1,high); // Merge sub-arrays
    }
}





// Take a batch of vectors in a and copy result in a_new for vectors of size n
__global__
void batch_merge(int *a, int *a_new, int length_a, int nb_loop, const int n){

  // Constant initialization
  int dimy = blockIdx.y;
  int threads_per_block = blockDim.x;
  int i_th = threadIdx.x;
  int a_begin = blockIdx.x * nb_loop * threads_per_block;
  int b_begin = blockIdx.x * nb_loop * threads_per_block + length_a;


  // Lopp if size of array >= 1024
  for (int j = 0; j < nb_loop ; j++) {

    int i_thread = i_th + j*threads_per_block;
    int Kx=0, Ky=0, Px=0, Py=0, Qx=0, Qy=0, offset=0;


    // Algorithm for two sorted arrays
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

  // XXXXXXXXXXXXXXXX Initialisation XXXXXXXXXXXXXXXXXXXXXXX //
  int p=12, nb_array=5;

	std::cout << "Enter the power of two of the arrays : " << std::endl;
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

  // Random generation of batch arrays
  std::mt19937 G(time(NULL));
  std::uniform_int_distribution<int> U(0,n);
  for (int i = 0; i < total_size; i++) {
    a[i] = U(G);
  }

  cudaMemcpy(d_a, a, total_size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a_copie, a, total_size * sizeof(int), cudaMemcpyHostToDevice);

  // Timer
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



  // Timer and algorithm on CPU
  double total_duration_cpu = 0;
  double duration=0, duration_cpu=0;
  std::clock_t start_cpu, end_cpu;


  for (int array_nb = 0 ; array_nb < nb_array ; array_nb++) {
    // To have a unique vector and not the batch (don't take this part into account)
    int *a_to_sort;
    a_to_sort = new int[n];
    for (int i = 0 ; i < n ; i++) {
      a_to_sort[i] = a[i+array_nb*n];
    }

    // Timer for the sort part
    start_cpu = std::clock();
    merge_sort(a_to_sort, 0, n-1);
    end_cpu = std::clock();
    duration = end_cpu - start_cpu;
    duration_cpu = (float)duration/(CLOCKS_PER_SEC/1000);
    total_duration_cpu += duration_cpu;

    // Copy result in a
    for (int i = 0 ; i < n ; i++) {
      a[i+array_nb*n] = a_to_sort[i];
    }

    delete [] a_to_sort;
  }



  // XXXXXXXXXXXXXXXX Résults XXXXXXXXXXXXXXXXXXXXXXX //

  cudaMemcpy(a_res, d_a, total_size * sizeof(int), cudaMemcpyDeviceToHost);

  // Verification

  int nb_error_gpu = 0;
  int nb_error_cpu = 0;
  for (int array_nb = 0 ; array_nb < nb_array ; array_nb++) {
    for (int i = 0 ; i < n-1 ; i++) {
      if (a_res[i+array_nb*n] > a_res[i+1+array_nb*n]) {nb_error_gpu++;}
      if (a[i+array_nb*n] > a[i+1+array_nb*n]) {nb_error_cpu++;}
    }
  }


  std::cout << "-------------------------------------" << std::endl;

  std::cout << "Size of the array : " << n << std::endl;
  std::cout << "Nb of array : " << nb_array << std::endl;

  std::cout << "-------------------------------------" << std::endl;

  std::cout << "Nb of error (GPU): " << nb_error_gpu << std::endl;
  std::cout << "Nb of error (CPU): " << nb_error_cpu << std::endl;

  std::cout << "-------------------------------------" << std::endl;

  std::cout << "Execution time for merge sort with GPU : " << Tim << " ms" << std::endl;
  std::cout << "Execution time for merge sort with CPU : " << total_duration_cpu << " ms" << std::endl;
  std::cout << "=> Merge sort with GPU is " << total_duration_cpu/Tim << "x much faster" << std::endl;
  delete [] a;
  delete [] a_res;
  cudaFree(d_a); cudaFree(d_a_copie);
  cudaEventDestroy(start); cudaEventDestroy(stop);

  return 0;

}
