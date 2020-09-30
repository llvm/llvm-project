/*
* @@name:   ompd_types.h
*/
#ifndef __OPMD_TYPES_H
#define __OPMD_TYPES_H

extern "C" {
#include "omp-tools.h"
}

#define OMPD_TYPES_VERSION   20170927 /* YYYYMMDD Format */

/* Kinds of device threads  */
#define OMPD_THREAD_ID_PTHREAD      ((ompd_thread_id_t)0)
#define OMPD_THREAD_ID_LWP          ((ompd_thread_id_t)1)
#define OMPD_THREAD_ID_WINTHREAD    ((ompd_thread_id_t)2)
#define OMPD_THREAD_ID_CUDALOGICAL  ((ompd_thread_id_t)3)
/* The range of non-standard implementation defined values */
#define OMPD_THREAD_ID_LO       ((ompd_thread_id_t)1000000)
#define OMPD_THREAD_ID_HI       ((ompd_thread_id_t)1100000)

/* Target Cuda device-specific thread identification */
typedef struct ompd_dim3_t {
    ompd_addr_t x;
    ompd_addr_t y;
    ompd_addr_t z;
} ompd_dim3_t;

typedef struct ompd_cudathread_coord_t {
    ompd_addr_t cudaDevId;
    ompd_addr_t cudaContext;
    ompd_addr_t warpSize;
    ompd_addr_t gridId;
    ompd_dim3_t  gridDim;
    ompd_dim3_t  blockDim;
    ompd_dim3_t  blockIdx;
    ompd_dim3_t  threadIdx;
} ompd_cudathread_coord_t;

/* Memory Access Segment definitions for Host and Target Devices */
#define OMPD_SEGMENT_UNSPECIFIED             ((ompd_seg_t)0)

/* Cuda-specific values consistent with those defined in cudadebugger.h */
#define OMPD_SEGMENT_CUDA_PTX_UNSPECIFIED    ((ompd_seg_t)0)
#define OMPD_SEGMENT_CUDA_PTX_CODE           ((ompd_seg_t)1)
#define OMPD_SEGMENT_CUDA_PTX_REG            ((ompd_seg_t)2)
#define OMPD_SEGMENT_CUDA_PTX_SREG           ((ompd_seg_t)3)
#define OMPD_SEGMENT_CUDA_PTX_CONST          ((ompd_seg_t)4)
#define OMPD_SEGMENT_CUDA_PTX_GLOBAL         ((ompd_seg_t)5)
#define OMPD_SEGMENT_CUDA_PTX_LOCAL          ((ompd_seg_t)6)
#define OMPD_SEGMENT_CUDA_PTX_PARAM          ((ompd_seg_t)7)
#define OMPD_SEGMENT_CUDA_PTX_SHARED         ((ompd_seg_t)8)
#define OMPD_SEGMENT_CUDA_PTX_SURF           ((ompd_seg_t)9)
#define OMPD_SEGMENT_CUDA_PTX_TEX            ((ompd_seg_t)10)
#define OMPD_SEGMENT_CUDA_PTX_TEXSAMPLER     ((ompd_seg_t)11)
#define OMPD_SEGMENT_CUDA_PTX_GENERIC        ((ompd_seg_t)12)
#define OMPD_SEGMENT_CUDA_PTX_IPARAM         ((ompd_seg_t)13)
#define OMPD_SEGMENT_CUDA_PTX_OPARAM         ((ompd_seg_t)14)
#define OMPD_SEGMENT_CUDA_PTX_FRAME          ((ompd_seg_t)15)

/* Kinds of device device address spaces */
#define OMPD_DEVICE_KIND_HOST     ((ompd_device_t)1)
#define OMPD_DEVICE_KIND_CUDA     ((ompd_device_t)2)
/* The range of non-standard implementation defined values */
#define OMPD_DEVICE_IMPL_LO       ((ompd_device_t)1000000)
#define OMPD_DEVICE_IMPL_HI ((ompd_device_t)1100000)
#endif

