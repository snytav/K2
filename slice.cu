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
