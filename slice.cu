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

__global__ void set_long_values(unsigned long long int* d_v, unsigned long long int num)
{

	d_v[blockIdx.x] = num;
}

//заполнить единичками,
void Slice::SET()
{
	unsigned long long int zero = 0;
	zero = ~zero;
#ifdef ss
	char s[100];
	long_to_binary(zero, s);
	printf("SET %s \n", s);


	cudaError_t err = cudaGetLastError();
	printf("error before set_lon_values %d \n", err);
	cudaError_t err_c = cudaMemcpy(h_v, d_v, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	long_to_binary(h_v[0], s);
	printf("h_v[0] %llu err %d %s\n", h_v[0], err_c, s);
	print("q1", 1);
#endif
	set_long_values << <NN, 1 >> > (d_v, zero);
	//    printf("SET: %i->%llu \n",NN,zero);
#ifdef qq
	err_c = cudaMemcpy(h_v, d_v, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	long_to_binary(h_v[0], s);
	printf("h_v[0] %llu err %d %s\n", h_v[0], err_c, s);


	print("q2", 1);

	err = cudaGetLastError();
	printf("error after set_lon_values %d \n", err);
#endif
}

//заполнить нулями,
void Slice::CLR()
{
	unsigned long long int zero = 0;
	set_long_values << <NN, 1 >> > (d_v, zero);
}

__global__ void set_mask_values(unsigned long long int* d_v, int num)
{
	unsigned long long int zero = 1;
	int num_el = num >> 6; // номер элемента, содержащий переход от 0 к 1;
	int el = num % SIZE_OF_LONG_INT;
	//  printf("%i in %i \n", num,num_el);
	if (blockIdx.x == num_el)
	{
		zero = (el == 0) ? 0 : (zero << (el - 1)) - 1;
		zero = ~zero;
	}
	else
	{
		zero = 0;
		if (blockIdx.x > num_el)
		{
			zero = ~zero;
		}
	}
	d_v[blockIdx.x] = zero;
}

void Slice::MASK(int i)
{
	set_mask_values << <NN, 1 >> > (d_v, i);
}