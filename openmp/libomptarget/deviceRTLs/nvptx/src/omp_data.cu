//===------------ omp_data.cu - NVPTX OpenMP GPU objects --------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the data objects used on the GPU device.
//
//===----------------------------------------------------------------------===//

#include "omptarget-nvptx.h"

////////////////////////////////////////////////////////////////////////////////
// global device envrionment
////////////////////////////////////////////////////////////////////////////////

__device__ omptarget_device_environmentTy omptarget_device_environment;

////////////////////////////////////////////////////////////////////////////////
// global data holding OpenMP state information
////////////////////////////////////////////////////////////////////////////////

__device__
    omptarget_nvptx_Queue<omptarget_nvptx_ThreadPrivateContext, OMP_STATE_COUNT>
        omptarget_nvptx_device_State[MAX_SM];

__device__ omptarget_nvptx_SimpleMemoryManager
    omptarget_nvptx_simpleMemoryManager;
__device__ __shared__ uint32_t usedMemIdx;
__device__ __shared__ uint32_t usedSlotIdx;

__device__ __shared__ uint8_t parallelLevel;

// Pointer to this team's OpenMP state object
__device__ __shared__
    omptarget_nvptx_ThreadPrivateContext *omptarget_nvptx_threadPrivateContext;

////////////////////////////////////////////////////////////////////////////////
// The team master sets the outlined parallel function in this variable to
// communicate with the workers.  Since it is in shared memory, there is one
// copy of these variables for each kernel, instance, and team.
////////////////////////////////////////////////////////////////////////////////
volatile __device__ __shared__ omptarget_nvptx_WorkFn omptarget_nvptx_workFn;

////////////////////////////////////////////////////////////////////////////////
// OpenMP kernel execution parameters
////////////////////////////////////////////////////////////////////////////////
__device__ __shared__ uint32_t execution_param;

////////////////////////////////////////////////////////////////////////////////
// Data sharing state
////////////////////////////////////////////////////////////////////////////////
__device__ __shared__ DataSharingStateTy DataSharingState;

////////////////////////////////////////////////////////////////////////////////
// Scratchpad for teams reduction.
////////////////////////////////////////////////////////////////////////////////
__device__ __shared__ void *ReductionScratchpadPtr;

////////////////////////////////////////////////////////////////////////////////
// Data sharing related variables.
////////////////////////////////////////////////////////////////////////////////
__device__ __shared__ omptarget_nvptx_SharedArgs omptarget_nvptx_globalArgs;
