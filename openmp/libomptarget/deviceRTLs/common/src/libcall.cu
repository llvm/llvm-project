//===------------ libcall.cu - OpenMP GPU user calls ------------- CUDA -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenMP runtime functions that can be
// invoked by the user in an OpenMP region
//
//===----------------------------------------------------------------------===//
#pragma omp declare target

#include <stdint.h>

#include "DeviceEnvironment.h"
#include "common/omptarget.h"
#include "target_impl.h"

EXTERN double omp_get_wtick(void) {
  double rc = __kmpc_impl_get_wtick();
  PRINT(LD_IO, "omp_get_wtick() returns %g\n", rc);
  return rc;
}

EXTERN double omp_get_wtime(void) {
  double rc = __kmpc_impl_get_wtime();
  PRINT(LD_IO, "call omp_get_wtime() returns %g\n", rc);
  return rc;
}

EXTERN void omp_set_num_threads(int num) {
  // Ignore it for SPMD mode.
  if (__kmpc_is_spmd_exec_mode())
    return;
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");
  PRINT(LD_IO, "call omp_set_num_threads(num %d)\n", num);
  if (num <= 0) {
    WARNING0(LW_INPUT, "expected positive num; ignore\n");
  } else if (parallelLevel[GetWarpId()] == 0) {
    nThreads = num;
  }
}

EXTERN int omp_get_num_threads(void) {
  int rc = GetNumberOfOmpThreads(__kmpc_is_spmd_exec_mode());
  PRINT(LD_IO, "call omp_get_num_threads() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_max_threads(void) {
  if (parallelLevel[GetWarpId()] > 0)
    // We're already in parallel region.
    return 1; // default is 1 thread avail
  // Not currently in a parallel region, return what was set.
  int rc = 1;
  if (parallelLevel[GetWarpId()] == 0)
    rc = nThreads;
  ASSERT0(LT_FUSSY, rc >= 0, "bad number of threads");
  PRINT(LD_IO, "call omp_get_max_threads() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_thread_limit(void) {
  if (__kmpc_is_spmd_exec_mode())
    return __kmpc_get_hardware_num_threads_in_block();
  int rc = threadLimit;
  PRINT(LD_IO, "call omp_get_thread_limit() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_thread_num() {
  int rc = GetOmpThreadId();
  PRINT(LD_IO, "call omp_get_thread_num() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_num_procs(void) {
  int rc = GetNumberOfProcsInDevice(__kmpc_is_spmd_exec_mode());
  PRINT(LD_IO, "call omp_get_num_procs() returns %d\n", rc);
  return rc;
}

EXTERN int omp_in_parallel(void) {
  int rc = parallelLevel[GetWarpId()] > OMP_ACTIVE_PARALLEL_LEVEL ? 1 : 0;
  PRINT(LD_IO, "call omp_in_parallel() returns %d\n", rc);
  return rc;
}

EXTERN int omp_in_final(void) {
  // treat all tasks as final... Specs may expect runtime to keep
  // track more precisely if a task was actively set by users... This
  // is not explicitly specified; will treat as if runtime can
  // actively decide to put a non-final task into a final one.
  int rc = 1;
  PRINT(LD_IO, "call omp_in_final() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_dynamic(int flag) {
  PRINT(LD_IO, "call omp_set_dynamic(%d) is ignored (no support)\n", flag);
}

EXTERN int omp_get_dynamic(void) {
  int rc = 0;
  PRINT(LD_IO, "call omp_get_dynamic() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_nested(int flag) {
  PRINT(LD_IO, "call omp_set_nested(%d) is ignored (no nested support)\n",
        flag);
}

EXTERN int omp_get_nested(void) {
  int rc = 0;
  PRINT(LD_IO, "call omp_get_nested() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_max_active_levels(int level) {
  PRINT(LD_IO,
        "call omp_set_max_active_levels(%d) is ignored (no nested support)\n",
        level);
}

EXTERN int omp_get_max_active_levels(void) {
  int rc = 1;
  PRINT(LD_IO, "call omp_get_max_active_levels() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_level(void) {
  int level = __kmpc_parallel_level();
  PRINT(LD_IO, "call omp_get_level() returns %d\n", level);
  return level;
}

EXTERN int omp_get_active_level(void) {
  int level = parallelLevel[GetWarpId()] > OMP_ACTIVE_PARALLEL_LEVEL ? 1 : 0;
  PRINT(LD_IO, "call omp_get_active_level() returns %d\n", level)
  return level;
}

EXTERN int omp_get_ancestor_thread_num(int level) {
  if (__kmpc_is_spmd_exec_mode())
    return level == 1 ? __kmpc_get_hardware_thread_id_in_block() : 0;
  int rc = -1;
  // If level is 0 or all parallel regions are not active - return 0.
  unsigned parLevel = parallelLevel[GetWarpId()];
  if (level == 1 && parLevel > OMP_ACTIVE_PARALLEL_LEVEL) {
    int totLevel = omp_get_level();
    if (level <= totLevel) {
      omptarget_nvptx_TaskDescr *currTaskDescr =
          getMyTopTaskDescriptor(/*isSPMDExecutionMode=*/false);
      int steps = totLevel - level;
      PRINT(LD_IO, "backtrack %d steps\n", steps);
      ASSERT0(LT_FUSSY, currTaskDescr,
              "do not expect fct to be called in a non-active thread");
      do {
        if (DON(LD_IOD)) {
          // print current state
          omp_sched_t sched = currTaskDescr->GetRuntimeSched();
          PRINT(LD_ALL,
                "task descr %s %d: %s, in par %d, rt sched %d,"
                " chunk %" PRIu64 "; tid %d, tnum %d, nthreads %d\n",
                "ancestor", steps,
                (currTaskDescr->IsParallelConstruct() ? "par" : "task"),
                (int)currTaskDescr->InParallelRegion(), (int)sched,
                currTaskDescr->RuntimeChunkSize(),
                (int)currTaskDescr->ThreadId(), (int)threadsInTeam,
                (int)nThreads);
        }

        if (currTaskDescr->IsParallelConstruct()) {
          // found the level
          if (!steps) {
            rc = currTaskDescr->ThreadId();
            break;
          }
          steps--;
        }
        currTaskDescr = currTaskDescr->GetPrevTaskDescr();
      } while (currTaskDescr);
      ASSERT0(LT_FUSSY, !steps, "expected to find all steps");
    }
  } else if (level == 0 ||
             (level > 0 && parLevel < OMP_ACTIVE_PARALLEL_LEVEL &&
              level <= parLevel) ||
             (level > 1 && parLevel > OMP_ACTIVE_PARALLEL_LEVEL &&
              level <= (parLevel - OMP_ACTIVE_PARALLEL_LEVEL))) {
    rc = 0;
  }
  PRINT(LD_IO, "call omp_get_ancestor_thread_num(level %d) returns %d\n", level,
        rc)
  return rc;
}

EXTERN int omp_get_team_size(int level) {
  if (__kmpc_is_spmd_exec_mode())
    return level == 1 ? __kmpc_get_hardware_num_threads_in_block() : 1;
  int rc = -1;
  unsigned parLevel = parallelLevel[GetWarpId()];
  // If level is 0 or all parallel regions are not active - return 1.
  if (level == 1 && parLevel > OMP_ACTIVE_PARALLEL_LEVEL) {
    rc = threadsInTeam;
  } else if (level == 0 ||
             (level > 0 && parLevel < OMP_ACTIVE_PARALLEL_LEVEL &&
              level <= parLevel) ||
             (level > 1 && parLevel > OMP_ACTIVE_PARALLEL_LEVEL &&
              level <= (parLevel - OMP_ACTIVE_PARALLEL_LEVEL))) {
    rc = 1;
  }
  PRINT(LD_IO, "call omp_get_team_size(level %d) returns %d\n", level, rc)
  return rc;
}

EXTERN void omp_get_schedule(omp_sched_t *kind, int *modifier) {
  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, __kmpc_is_spmd_exec_mode(),
            "Expected SPMD mode only with uninitialized runtime.");
    *kind = omp_sched_static;
    *modifier = 1;
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        getMyTopTaskDescriptor(__kmpc_is_spmd_exec_mode());
    *kind = currTaskDescr->GetRuntimeSched();
    *modifier = currTaskDescr->RuntimeChunkSize();
  }
  PRINT(LD_IO, "call omp_get_schedule returns sched %d and modif %d\n",
        (int)*kind, *modifier);
}

EXTERN void omp_set_schedule(omp_sched_t kind, int modifier) {
  PRINT(LD_IO, "call omp_set_schedule(sched %d, modif %d)\n", (int)kind,
        modifier);
  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, __kmpc_is_spmd_exec_mode(),
            "Expected SPMD mode only with uninitialized runtime.");
    return;
  }
  if (kind >= omp_sched_static && kind < omp_sched_auto) {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        getMyTopTaskDescriptor(__kmpc_is_spmd_exec_mode());
    currTaskDescr->SetRuntimeSched(kind);
    currTaskDescr->RuntimeChunkSize() = modifier;
    PRINT(LD_IOD, "omp_set_schedule did set sched %d & modif %" PRIu64 "\n",
          (int)currTaskDescr->GetRuntimeSched(),
          currTaskDescr->RuntimeChunkSize());
  }
}

EXTERN omp_proc_bind_t omp_get_proc_bind(void) {
  PRINT0(LD_IO, "call omp_get_proc_bin() is true, regardless on state\n");
  return omp_proc_bind_true;
}

EXTERN int omp_get_num_places(void) {
  PRINT0(LD_IO, "call omp_get_num_places() returns 0\n");
  return 0;
}

EXTERN int omp_get_place_num_procs(int place_num) {
  PRINT0(LD_IO, "call omp_get_place_num_procs() returns 0\n");
  return 0;
}

EXTERN void omp_get_place_proc_ids(int place_num, int *ids) {
  PRINT0(LD_IO, "call to omp_get_place_proc_ids()\n");
}

EXTERN int omp_get_place_num(void) {
  PRINT0(LD_IO, "call to omp_get_place_num() returns 0\n");
  return 0;
}

EXTERN int omp_get_partition_num_places(void) {
  PRINT0(LD_IO, "call to omp_get_partition_num_places() returns 0\n");
  return 0;
}

EXTERN void omp_get_partition_place_nums(int *place_nums) {
  PRINT0(LD_IO, "call to omp_get_partition_place_nums()\n");
}

EXTERN int omp_get_cancellation(void) {
  int rc = 0;
  PRINT(LD_IO, "call omp_get_cancellation() returns %d\n", rc);
  return rc;
}

EXTERN void omp_set_default_device(int deviceId) {
  PRINT0(LD_IO, "call omp_get_default_device() is undef on device\n");
}

EXTERN int omp_get_default_device(void) {
  PRINT0(LD_IO,
         "call omp_get_default_device() is undef on device, returns 0\n");
  return 0;
}

#ifdef __AMDGCN__
EXTERN int omp_get_num_devices(void) {
  PRINT(LD_IO, "call omp_get_num_devices() returns device_size %d\n",
        omptarget_device_environment.NumDevices);
  return omptarget_device_environment.NumDevices;
}
EXTERN int omp_get_device_num(void) {
  PRINT(LD_IO, "call omp_get_device_num() returns device_num %d\n",
        omptarget_device_environment.DeviceNum);
  return omptarget_device_environment.DeviceNum;
}
#endif

EXTERN int omp_get_num_teams(void) {
  int rc = GetNumberOfOmpTeams();
  PRINT(LD_IO, "call omp_get_num_teams() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_team_num() {
  int rc = GetOmpTeamId();
  PRINT(LD_IO, "call omp_get_team_num() returns %d\n", rc);
  return rc;
}

EXTERN int omp_is_initial_device_OVERLOAD(void) asm("omp_is_initial_device");
EXTERN int omp_is_initial_device_OVERLOAD(void) {
  PRINT0(LD_IO, "call omp_is_initial_device() returns 0\n");
  return 0; // 0 by def on device
}

// Unspecified on the device.
EXTERN int omp_get_initial_device(void) {
  PRINT0(LD_IO, "call omp_get_initial_device() returns 0\n");
  return 0;
}

// Unused for now.
EXTERN int omp_get_max_task_priority(void) {
  PRINT0(LD_IO, "call omp_get_max_task_priority() returns 0\n");
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
// locks
////////////////////////////////////////////////////////////////////////////////

EXTERN void omp_init_lock(omp_lock_t *lock) {
  __kmpc_impl_init_lock(lock);
  PRINT0(LD_IO, "call omp_init_lock()\n");
}

EXTERN void omp_destroy_lock(omp_lock_t *lock) {
  __kmpc_impl_destroy_lock(lock);
  PRINT0(LD_IO, "call omp_destroy_lock()\n");
}

EXTERN void omp_set_lock(omp_lock_t *lock) {
  __kmpc_impl_set_lock(lock);
  PRINT0(LD_IO, "call omp_set_lock()\n");
}

EXTERN void omp_unset_lock(omp_lock_t *lock) {
  __kmpc_impl_unset_lock(lock);
  PRINT0(LD_IO, "call omp_unset_lock()\n");
}

EXTERN int omp_test_lock(omp_lock_t *lock) {
  int rc = __kmpc_impl_test_lock(lock);
  PRINT(LD_IO, "call omp_test_lock() return %d\n", rc);
  return rc;
}

#pragma omp end declare target
