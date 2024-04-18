#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "param.h"
#include "slice.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/copy.h>

unsigned long long int h_v[N1];// for print mast be copied from d_v


int Slice::Init(unsigned int k)
{

	cudaError_t err1, err = cudaGetLastError();

	//	printf("before all error %d , %s \n",err,cudaGetErrorString(err));
	//	if (err!=0)exit(0);
	length = k;
	NN = (((k % SIZE_OF_LONG_INT) == 0) ? (k / SIZE_OF_LONG_INT) : (k / SIZE_OF_LONG_INT + 1));
	//    printf("slice.init %u ", NN);
#ifdef ssss
	int* d_i;
	printf("Slice init %d %s\n", err, cudaGetErrorString(err));
	err = cudaMalloc(&d_i, sizeof(int));

	d_first_non_zero = d_i;
	printf("Slice alloc error %d %s \n", err, cudaGetErrorString(err));
#endif
	//	err1 = cudaGetLastError();
	//	printf("before alloc error %d , %s \n",err1,cudaGetErrorString(err1));
	err = cudaMalloc(&d_v, NN * sizeof(unsigned long long int));
	//   printf("Slice alloc error %d , %s ,%p \n",err,cudaGetErrorString(err),d_v);
#ifdef ssss
	printf("Slice alloc error %d %s \n", err, cudaGetErrorString(err));
#endif
	cudaMemset(d_v, 0, NN * sizeof(unsigned long long int));

	//	exit(0);

	return err;
}
