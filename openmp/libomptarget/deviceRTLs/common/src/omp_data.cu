//===------------ omp_data.cu - OpenMP GPU objects --------------- CUDA -*-===//
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
#pragma omp declare target

#include "common/allocator.h"
#include "common/omptarget.h"

////////////////////////////////////////////////////////////////////////////////
// global device environment
////////////////////////////////////////////////////////////////////////////////

PLUGIN_ACCESSIBLE
DeviceEnvironmentTy omptarget_device_environment;

////////////////////////////////////////////////////////////////////////////////
// global data holding OpenMP state information
////////////////////////////////////////////////////////////////////////////////

// OpenMP will try to call its ctor if we don't add the attribute explicitly
[[clang::loader_uninitialized]] omptarget_nvptx_Queue<
    omptarget_nvptx_ThreadPrivateContext, OMP_STATE_COUNT>
    omptarget_nvptx_device_State[MAX_SM];

omptarget_nvptx_SimpleMemoryManager omptarget_nvptx_simpleMemoryManager;
uint32_t SHARED(usedMemIdx);
uint32_t SHARED(usedSlotIdx);

// SHARED doesn't work with array so we add the attribute explicitly.
[[clang::loader_uninitialized]] uint8_t
    parallelLevel[MAX_THREADS_PER_TEAM / WARPSIZE];
#pragma omp allocate(parallelLevel) allocator(omp_pteam_mem_alloc)
uint16_t SHARED(threadLimit);
uint16_t SHARED(threadsInTeam);
uint16_t SHARED(nThreads);
// Pointer to this team's OpenMP state object
omptarget_nvptx_ThreadPrivateContext *
    SHARED(omptarget_nvptx_threadPrivateContext);

////////////////////////////////////////////////////////////////////////////////
// The team master sets the outlined parallel function in this variable to
// communicate with the workers.  Since it is in shared memory, there is one
// copy of these variables for each kernel, instance, and team.
////////////////////////////////////////////////////////////////////////////////
omptarget_nvptx_WorkFn SHARED(omptarget_nvptx_workFn);

////////////////////////////////////////////////////////////////////////////////
// OpenMP kernel execution parameters
////////////////////////////////////////////////////////////////////////////////
int8_t SHARED(execution_param);

////////////////////////////////////////////////////////////////////////////////
// Scratchpad for teams reduction.
////////////////////////////////////////////////////////////////////////////////
void *SHARED(ReductionScratchpadPtr);

#pragma omp end declare target
