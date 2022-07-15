//===--- Kernel.cpp - OpenMP device kernel interface -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
// Notified per clause 4(b) of the license.
//
//===----------------------------------------------------------------------===//
//
// This file contains the kernel entry points for the device.
//
//===----------------------------------------------------------------------===//

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"

using namespace _OMP;

#pragma omp begin declare target device_type(nohost)

static void inititializeRuntime(bool IsSPMD) {
  // Order is important here.
  synchronize::init(IsSPMD);
  mapping::init(IsSPMD);
  state::init(IsSPMD);
  if (__kmpc_get_hardware_thread_id_in_block() == 0)
    __init_ThreadDSTPtrPtr();
}

/// Simple generic state machine for worker threads.
static void genericStateMachine(IdentTy *Ident) {
  FunctionTracingRAII();

  uint32_t TId = mapping::getThreadIdInBlock();

  do {
    ParallelRegionFnTy WorkFn = 0;
    // Wait for the signal that we have a new work function.
    synchronize::workersStartBarrier();

    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_parallel(&WorkFn);

    // If there is nothing more to do, break out of the state machine by
    // returning to the caller.
    if (!WorkFn)
      return;

    if (IsActive) {
      ASSERT(!mapping::isSPMDMode());
      ((void (*)(uint32_t, uint32_t))WorkFn)(0, TId);
      __kmpc_kernel_end_parallel();
    }

    synchronize::workersDoneBarrier();

  } while (true);
}

extern "C" {

/// Initialization
///
/// \param Ident               Source location identification, can be NULL.
///
int32_t __kmpc_target_init(IdentTy *Ident, int8_t Mode,
                           bool UseGenericStateMachine, bool) {
  FunctionTracingRAII();

  const bool IsSPMD = Mode & OMP_TGT_EXEC_MODE_SPMD;
#ifdef __AMDGCN__
  if (__kmpc_get_hardware_thread_id_in_block() == 0) {
    synchronize::omptarget_workers_done = false;
    synchronize::omptarget_master_ready = false;
  }
  synchronize::threadsAligned();
#endif
  if (IsSPMD) {
    inititializeRuntime(/* IsSPMD */ true);
    synchronize::threadsAligned();
  } else {
    inititializeRuntime(/* IsSPMD */ false);
    // No need to wait since only the main threads will execute user
    // code and workers will run into a barrier right away.
  }

  if (IsSPMD) {
    state::assumeInitialState(IsSPMD);
    return -1;
  }

  if (mapping::isInitialThreadInLevel0(IsSPMD))
    return -1;

  // Enter the generic state machine if enabled and if this thread can possibly
  // be an active worker thread.
  //
  // The latter check is important for NVIDIA Pascal (but not Volta) and AMD
  // GPU.  In those cases, a single thread can apparently satisfy a barrier on
  // behalf of all threads in the same warp.  Thus, it would not be safe for
  // other threads in the main thread's warp to reach the first
  // synchronize::threads call in genericStateMachine before the main thread
  // reaches its corresponding synchronize::threads call: that would permit all
  // active worker threads to proceed before the main thread has actually set
  // state::ParallelRegionFn, and then they would immediately quit without
  // doing any work.  mapping::getBlockSize() does not include any of the main
  // thread's warp, so none of its threads can ever be active worker threads.
  if (UseGenericStateMachine &&
      mapping::getThreadIdInBlock() < mapping::getBlockSize(IsSPMD))
    genericStateMachine(Ident);

  return mapping::getThreadIdInBlock();
}

/// De-Initialization
///
/// In non-SPMD, this function releases the workers trapped in a state machine
/// and also any memory dynamically allocated by the runtime.
///
/// \param Ident Source location identification, can be NULL.
///
void __kmpc_target_deinit(IdentTy *Ident, int8_t Mode, bool) {
  FunctionTracingRAII();
  const bool IsSPMD = Mode & OMP_TGT_EXEC_MODE_SPMD;
  state::assumeInitialState(IsSPMD);
  if (IsSPMD)
    return;

  // Signal the workers to exit the state machine and exit the kernel.
  state::ParallelRegionFn = nullptr;

  // make sure workers cannot continue before the initial thread
  // has reset the Fn pointer for termination
  synchronize::omptarget_master_ready = true;
  synchronize::threads();
}

#ifndef FORTRAN_NO_LONGER_NEEDS

int32_t __kmpc_target_init_v1(int64_t *, int8_t Mode,
                              int8_t UseGenericStateMachine,
                              int8_t RequiresFullRuntime) {
  FunctionTracingRAII();
  int32_t res = __kmpc_target_init(nullptr, Mode, UseGenericStateMachine,
                                   RequiresFullRuntime);
  if (Mode & OMP_TGT_EXEC_MODE_SPMD) {

    uint32_t TId = mapping::getThreadIdInBlock();

    uint32_t NThreadsICV = icv::NThreads;
    uint32_t NumThreads = mapping::getBlockSize();

    if (NThreadsICV != 0 && NThreadsICV < NumThreads)
      NumThreads = NThreadsICV;

    synchronize::threadsAligned();
    if (TId == 0) {
      // Note that the order here is important. `icv::Level` has to be updated
      // last or the other updates will cause a thread specific state to be
      // created.
      state::ParallelTeamSize = NumThreads;
      icv::ActiveLevel = 1u;
      icv::Level = 1u;
    }
    synchronize::threadsAligned();
  }
  return res;
}

void __kmpc_target_deinit_v1(int64_t *, int8_t Mode,
                             int8_t RequiresFullRuntime) {
  FunctionTracingRAII();
  uint32_t TId = mapping::getThreadIdInBlock();
  synchronize::threadsAligned();

  if (TId == 0) {
    // Reverse order of deinitialization
    icv::Level = 0u;
    icv::ActiveLevel = 0u;
    state::ParallelTeamSize = 1u;
  }
  // Synchronize all threads to make sure every thread exits the scope above;
  // otherwise the following assertions and the assumption in
  // __kmpc_target_deinit may not hold.
  synchronize::threadsAligned();
  __kmpc_target_deinit(nullptr, Mode, RequiresFullRuntime);
}

#endif

int8_t __kmpc_is_spmd_exec_mode() {
  FunctionTracingRAII();
  return mapping::isSPMDMode();
}
}

#pragma omp end declare target
