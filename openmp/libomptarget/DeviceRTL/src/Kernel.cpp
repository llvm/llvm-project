//===--- Kernel.cpp - OpenMP device kernel interface -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the kernel entry points for the device.
//
//===----------------------------------------------------------------------===//

#include "Shared/Environment.h"

#include "Allocator.h"
#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"

#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"

using namespace ompx;

#pragma omp begin declare target device_type(nohost)

static void
inititializeRuntime(bool IsSPMD, KernelEnvironmentTy &KernelEnvironment,
                    KernelLaunchEnvironmentTy &KernelLaunchEnvironment) {
  // Order is important here.
  synchronize::init(IsSPMD);
  mapping::init(IsSPMD);
  if (__kmpc_get_hardware_thread_id_in_block() == 0)
    __init_ThreadDSTPtrPtr();

  state::init(IsSPMD, KernelEnvironment, KernelLaunchEnvironment);
  allocator::init(IsSPMD, KernelEnvironment);
}

/// Simple generic state machine for worker threads.
static void genericStateMachine(IdentTy *Ident) {
  uint32_t TId = mapping::getThreadIdInBlock();

  do {
    ParallelRegionFnTy WorkFn = nullptr;

    // Wait for the signal that we have a new work function.
    synchronize::threads(atomic::seq_cst);

    // Retrieve the work function from the runtime.
    bool IsActive = __kmpc_kernel_parallel(&WorkFn);

    // If there is nothing more to do, break out of the state machine by
    // returning to the caller.
    if (!WorkFn)
      return;

    if (IsActive) {
      ASSERT(!mapping::isSPMDMode(), nullptr);
      ((void (*)(uint32_t, uint32_t))WorkFn)(0, TId);
      __kmpc_kernel_end_parallel();
    }

    synchronize::threads(atomic::seq_cst);

  } while (true);
}

extern "C" {

/// Initialization
///
/// \param Ident               Source location identification, can be NULL.
///
int32_t __kmpc_target_init(KernelEnvironmentTy &KernelEnvironment,
                           KernelLaunchEnvironmentTy &KernelLaunchEnvironment) {
  ConfigurationEnvironmentTy &Configuration = KernelEnvironment.Configuration;
  bool IsSPMD = Configuration.ExecMode &
                llvm::omp::OMPTgtExecModeFlags::OMP_TGT_EXEC_MODE_SPMD;
  bool UseGenericStateMachine = Configuration.UseGenericStateMachine;
  if (IsSPMD) {
    inititializeRuntime(/*IsSPMD=*/true, KernelEnvironment,
                        KernelLaunchEnvironment);
    synchronize::threadsAligned(atomic::relaxed);
  } else {
    inititializeRuntime(/*IsSPMD=*/false, KernelEnvironment,
                        KernelLaunchEnvironment);
    // No need to wait since only the main threads will execute user
    // code and workers will run into a barrier right away.
  }

  if (IsSPMD) {
    state::assumeInitialState(IsSPMD);

    // Synchronize to ensure the assertions above are in an aligned region.
    // The barrier is eliminated later.
    synchronize::threadsAligned(atomic::relaxed);
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
  // doing any work.  mapping::getMaxTeamThreads() does not include any of the
  // main thread's warp, so none of its threads can ever be active worker
  // threads.
  if (UseGenericStateMachine &&
      mapping::getThreadIdInBlock() < mapping::getMaxTeamThreads(IsSPMD))
    genericStateMachine(KernelEnvironment.Ident);

  return mapping::getThreadIdInBlock();
}

/// De-Initialization
///
/// In non-SPMD, this function releases the workers trapped in a state machine
/// and also any memory dynamically allocated by the runtime.
///
/// \param Ident Source location identification, can be NULL.
///
void __kmpc_target_deinit() {
  bool IsSPMD = mapping::isSPMDMode();
  if (IsSPMD)
    return;

  if (mapping::isInitialThreadInLevel0(IsSPMD)) {
    // Signal the workers to exit the state machine and exit the kernel.
    state::ParallelRegionFn = nullptr;
  } else if (!state::getKernelEnvironment()
                  .Configuration.UseGenericStateMachine) {
    // Retrieve the work function just to ensure we always call
    // __kmpc_kernel_parallel even if a custom state machine is used.
    // TODO: this is not super pretty. The problem is we create the call to
    // __kmpc_kernel_parallel in the openmp-opt pass but while we optimize it
    // is not there yet. Thus, we assume we never reach it from
    // __kmpc_target_deinit. That allows us to remove the store in there to
    // ParallelRegionFn, which leads to bad results later on.
    ParallelRegionFnTy WorkFn = nullptr;
    __kmpc_kernel_parallel(&WorkFn);
    ASSERT(WorkFn == nullptr, nullptr);
  }
}

#ifndef FORTRAN_NO_LONGER_NEEDS
int32_t __kmpc_target_init_v1(int64_t *, int8_t Mode,
                              int8_t UseGenericStateMachine,
                              int8_t RequiresFullRuntime) {
  KernelEnvironmentTy KE{{static_cast<uint8_t>(UseGenericStateMachine),
                          /*MayUseNestedParallelism=*/1,
                          static_cast<llvm::omp::OMPTgtExecModeFlags>(Mode)},
                         /*Ident=*/nullptr,
                         /*DebugIndentionLevel=*/0};
  KernelLaunchEnvironmentTy KernelLaunchEnvironment;
  int32_t res = __kmpc_target_init(KE, KernelLaunchEnvironment);
  if (Mode & llvm::omp::OMP_TGT_EXEC_MODE_SPMD) {

    uint32_t TId = mapping::getThreadIdInBlock();

    uint32_t NThreadsICV = icv::NThreads;
    uint32_t NumThreads = mapping::getNumberOfThreadsInBlock();

    if (NThreadsICV != 0 && NThreadsICV < NumThreads)
      NumThreads = NThreadsICV;

    synchronize::threadsAligned(atomic::seq_cst);
    if (TId == 0) {
      // Note that the order here is important. `icv::Level` has to be updated
      // last or the other updates will cause a thread specific state to be
      // created.
      state::ParallelTeamSize = NumThreads;
      icv::ActiveLevel = 1u;
      icv::Level = 1u;
    }
    synchronize::threadsAligned(atomic::seq_cst);
  }
  return res;
}

void __kmpc_target_deinit_v1(int64_t *, int8_t Mode,
                             int8_t RequiresFullRuntime) {
  uint32_t TId = mapping::getThreadIdInBlock();
  synchronize::threadsAligned(atomic::seq_cst);

  if (TId == 0) {
    // Reverse order of deinitialization
    icv::Level = 0u;
    icv::ActiveLevel = 0u;
    state::ParallelTeamSize = 1u;
  }
  // Synchronize all threads to make sure every thread exits the scope above;
  // otherwise the following assertions and the assumption in
  // __kmpc_target_deinit may not hold.
  synchronize::threadsAligned(atomic::seq_cst);
  __kmpc_target_deinit();
}

#endif

int8_t __kmpc_is_spmd_exec_mode() { return mapping::isSPMDMode(); }
}

#pragma omp end declare target
