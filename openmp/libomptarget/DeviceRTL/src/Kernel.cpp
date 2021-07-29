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

#include "Debug.h"
#include "Interface.h"
#include "Mapping.h"
#include "State.h"
#include "Synchronization.h"
#include "Types.h"

using namespace _OMP;

#pragma omp declare target

static void inititializeRuntime(bool IsSPMD) {
  // Order is important here.
  synchronize::init(IsSPMD);
  mapping::init(IsSPMD);
  state::init(IsSPMD);
}

/// Simple generic state machine for worker threads.
static void genericStateMachine(IdentTy *Ident) {

  uint32_t TId = mapping::getThreadIdInBlock();

  do {
    ParallelRegionFnTy WorkFn = 0;

    // Wait for the signal that we have a new work function.
    synchronize::threads();

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

    synchronize::threads();

  } while (true);
}

extern "C" {

/// Initialization
///
/// \param Ident               Source location identification, can be NULL.
///
int32_t __kmpc_target_init(IdentTy *Ident, bool IsSPMD,
                           bool UseGenericStateMachine, bool) {
  if (IsSPMD) {
    inititializeRuntime(/* IsSPMD */ true);
    synchronize::threads();
  } else {
    inititializeRuntime(/* IsSPMD */ false);
    // No need to wait since only the main threads will execute user
    // code and workers will run into a barrier right away.
  }

  if (IsSPMD) {
    state::assumeInitialState(IsSPMD);
    return -1;
  }

  if (mapping::isMainThreadInGenericMode())
    return -1;

  if (UseGenericStateMachine)
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
void __kmpc_target_deinit(IdentTy *Ident, bool IsSPMD, bool) {
  state::assumeInitialState(IsSPMD);
  if (IsSPMD)
    return;

  // Signal the workers to exit the state machine and exit the kernel.
  state::ParallelRegionFn = nullptr;
}

int8_t __kmpc_is_spmd_exec_mode() { return mapping::isSPMDMode(); }
}

#pragma omp end declare target
