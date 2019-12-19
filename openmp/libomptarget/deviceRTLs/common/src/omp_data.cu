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

#include "common/omptarget.h"
#include "common/device_environment.h"

////////////////////////////////////////////////////////////////////////////////
// global device environment
////////////////////////////////////////////////////////////////////////////////

DEVICE omptarget_device_environmentTy omptarget_device_environment;

////////////////////////////////////////////////////////////////////////////////
// global data holding OpenMP state information
////////////////////////////////////////////////////////////////////////////////

DEVICE
    omptarget_nvptx_Queue<omptarget_nvptx_ThreadPrivateContext, OMP_STATE_COUNT>
        omptarget_nvptx_device_State[MAX_SM];

DEVICE omptarget_nvptx_SimpleMemoryManager
    omptarget_nvptx_simpleMemoryManager;
DEVICE SHARED uint32_t usedMemIdx;
DEVICE SHARED uint32_t usedSlotIdx;

DEVICE SHARED uint8_t parallelLevel[MAX_THREADS_PER_TEAM / WARPSIZE];
DEVICE SHARED uint16_t threadLimit;
DEVICE SHARED uint16_t threadsInTeam;
DEVICE SHARED uint16_t nThreads;
// Pointer to this team's OpenMP state object
DEVICE SHARED
    omptarget_nvptx_ThreadPrivateContext *omptarget_nvptx_threadPrivateContext;

////////////////////////////////////////////////////////////////////////////////
// The team master sets the outlined parallel function in this variable to
// communicate with the workers.  Since it is in shared memory, there is one
// copy of these variables for each kernel, instance, and team.
////////////////////////////////////////////////////////////////////////////////
volatile DEVICE SHARED omptarget_nvptx_WorkFn omptarget_nvptx_workFn;

////////////////////////////////////////////////////////////////////////////////
// OpenMP kernel execution parameters
////////////////////////////////////////////////////////////////////////////////
DEVICE SHARED uint32_t execution_param;

////////////////////////////////////////////////////////////////////////////////
// Data sharing state
////////////////////////////////////////////////////////////////////////////////
DEVICE SHARED DataSharingStateTy DataSharingState;

////////////////////////////////////////////////////////////////////////////////
// Scratchpad for teams reduction.
////////////////////////////////////////////////////////////////////////////////
DEVICE SHARED void *ReductionScratchpadPtr;

////////////////////////////////////////////////////////////////////////////////
// Data sharing related variables.
////////////////////////////////////////////////////////////////////////////////
DEVICE SHARED omptarget_nvptx_SharedArgs omptarget_nvptx_globalArgs;
