//===------------ libcall.cu - NVPTX OpenMP user calls ----------- CUDA -*-===//
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

#include "omptarget-nvptx.h"

// Timer precision is 1ns
#define TIMER_PRECISION ((double)1E-9)

EXTERN double omp_get_wtick(void) {
  PRINT(LD_IO, "omp_get_wtick() returns %g\n", TIMER_PRECISION);
  return TIMER_PRECISION;
}

EXTERN double omp_get_wtime(void) {
  unsigned long long nsecs;
  asm("mov.u64  %0, %%globaltimer;" : "=l"(nsecs));
  double rc = (double)nsecs * TIMER_PRECISION;
  PRINT(LD_IO, "call omp_get_wtime() returns %g\n", rc);
  return rc;
}

EXTERN void omp_set_num_threads(int num) {
  // Ignore it for SPMD mode.
  if (isSPMDMode())
    return;
  ASSERT0(LT_FUSSY, isRuntimeInitialized(), "Expected initialized runtime.");
  PRINT(LD_IO, "call omp_set_num_threads(num %d)\n", num);
  if (num <= 0) {
    WARNING0(LW_INPUT, "expected positive num; ignore\n");
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        getMyTopTaskDescriptor(/*isSPMDExecutionMode=*/false);
    currTaskDescr->NThreads() = num;
  }
}

EXTERN int omp_get_num_threads(void) {
  bool isSPMDExecutionMode = isSPMDMode();
  int tid = GetLogicalThreadIdInBlock(isSPMDExecutionMode);
  int rc = GetNumberOfOmpThreads(tid, isSPMDExecutionMode);
  PRINT(LD_IO, "call omp_get_num_threads() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_max_threads(void) {
  if (isSPMDMode())
    // We're already in parallel region.
    return 1;  // default is 1 thread avail
  omptarget_nvptx_TaskDescr *currTaskDescr =
      getMyTopTaskDescriptor(/*isSPMDExecutionMode=*/false);
  int rc = 1; // default is 1 thread avail
  if (!currTaskDescr->InParallelRegion()) {
    // Not currently in a parallel region, return what was set.
    rc = currTaskDescr->NThreads();
    ASSERT0(LT_FUSSY, rc >= 0, "bad number of threads");
  }
  PRINT(LD_IO, "call omp_get_max_threads() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_thread_limit(void) {
  if (isSPMDMode())
    return GetNumberOfThreadsInBlock();
  // per contention group.. meaning threads in current team
  omptarget_nvptx_TaskDescr *currTaskDescr =
      getMyTopTaskDescriptor(/*isSPMDExecutionMode=*/false);
  int rc = currTaskDescr->ThreadLimit();
  PRINT(LD_IO, "call omp_get_thread_limit() return %d\n", rc);
  return rc;
}

EXTERN int omp_get_thread_num() {
  bool isSPMDExecutionMode = isSPMDMode();
  int tid = GetLogicalThreadIdInBlock(isSPMDExecutionMode);
  int rc = GetOmpThreadId(tid, isSPMDExecutionMode);
  PRINT(LD_IO, "call omp_get_thread_num() returns %d\n", rc);
  return rc;
}

EXTERN int omp_get_num_procs(void) {
  int rc = GetNumberOfProcsInDevice(isSPMDMode());
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
  // is not explicitely specified; will treat as if runtime can
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
  int level = parallelLevel[GetWarpId()] & (OMP_ACTIVE_PARALLEL_LEVEL - 1);
  PRINT(LD_IO, "call omp_get_level() returns %d\n", level);
  return level;
}

EXTERN int omp_get_active_level(void) {
  int level = parallelLevel[GetWarpId()] > OMP_ACTIVE_PARALLEL_LEVEL ? 1 : 0;
  PRINT(LD_IO, "call omp_get_active_level() returns %d\n", level)
  return level;
}

EXTERN int omp_get_ancestor_thread_num(int level) {
  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, isSPMDMode(),
            "Expected SPMD mode only with uninitialized runtime.");
    return level == 1 ? GetThreadIdInBlock() : 0;
  }
  int rc = -1;
  if (level == 0) {
    rc = 0;
  } else if (level > 0) {
    int totLevel = omp_get_level();
    if (level <= totLevel) {
      omptarget_nvptx_TaskDescr *currTaskDescr =
          getMyTopTaskDescriptor(isSPMDMode());
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
                (int)currTaskDescr->ThreadId(),
                (int)currTaskDescr->ThreadsInTeam(),
                (int)currTaskDescr->NThreads());
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
  }
  PRINT(LD_IO, "call omp_get_ancestor_thread_num(level %d) returns %d\n", level,
        rc)
  return rc;
}

EXTERN int omp_get_team_size(int level) {
  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, isSPMDMode(),
            "Expected SPMD mode only with uninitialized runtime.");
    return level == 1 ? GetNumberOfThreadsInBlock() : 1;
  }
  int rc = -1;
  if (level == 0) {
    rc = 1;
  } else if (level > 0) {
    int totLevel = omp_get_level();
    if (level <= totLevel) {
      omptarget_nvptx_TaskDescr *currTaskDescr =
          getMyTopTaskDescriptor(isSPMDMode());
      int steps = totLevel - level;
      ASSERT0(LT_FUSSY, currTaskDescr,
              "do not expect fct to be called in a non-active thread");
      do {
        if (currTaskDescr->IsParallelConstruct()) {
          if (!steps) {
            // found the level
            rc = currTaskDescr->ThreadsInTeam();
            break;
          }
          steps--;
        }
        currTaskDescr = currTaskDescr->GetPrevTaskDescr();
      } while (currTaskDescr);
      ASSERT0(LT_FUSSY, !steps, "expected to find all steps");
    }
  }
  PRINT(LD_IO, "call omp_get_team_size(level %d) returns %d\n", level, rc)
  return rc;
}

EXTERN void omp_get_schedule(omp_sched_t *kind, int *modifier) {
  if (isRuntimeUninitialized()) {
    ASSERT0(LT_FUSSY, isSPMDMode(),
            "Expected SPMD mode only with uninitialized runtime.");
    *kind = omp_sched_static;
    *modifier = 1;
  } else {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        getMyTopTaskDescriptor(isSPMDMode());
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
    ASSERT0(LT_FUSSY, isSPMDMode(),
            "Expected SPMD mode only with uninitialized runtime.");
    return;
  }
  if (kind >= omp_sched_static && kind < omp_sched_auto) {
    omptarget_nvptx_TaskDescr *currTaskDescr =
        getMyTopTaskDescriptor(isSPMDMode());
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
  int rc = FALSE; // currently false only
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

EXTERN int omp_get_num_devices(void) {
  PRINT0(LD_IO, "call omp_get_num_devices() is undef on device, returns 0\n");
  return 0;
}

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

EXTERN int omp_is_initial_device(void) {
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

#define __OMP_SPIN 1000
#define UNSET 0
#define SET 1

EXTERN void omp_init_lock(omp_lock_t *lock) {
  omp_unset_lock(lock);
  PRINT0(LD_IO, "call omp_init_lock()\n");
}

EXTERN void omp_destroy_lock(omp_lock_t *lock) {
  omp_unset_lock(lock);
  PRINT0(LD_IO, "call omp_destroy_lock()\n");
}

EXTERN void omp_set_lock(omp_lock_t *lock) {
  // int atomicCAS(int* address, int compare, int val);
  // (old == compare ? val : old)

  // TODO: not sure spinning is a good idea here..
  while (atomicCAS(lock, UNSET, SET) != UNSET) {
    clock_t start = clock();
    clock_t now;
    for (;;) {
      now = clock();
      clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
      if (cycles >= __OMP_SPIN * blockIdx.x) {
        break;
      }
    }
  } // wait for 0 to be the read value

  PRINT0(LD_IO, "call omp_set_lock()\n");
}

EXTERN void omp_unset_lock(omp_lock_t *lock) {
  (void)atomicExch(lock, UNSET);

  PRINT0(LD_IO, "call omp_unset_lock()\n");
}

EXTERN int omp_test_lock(omp_lock_t *lock) {
  // int atomicCAS(int* address, int compare, int val);
  // (old == compare ? val : old)
  int ret = atomicAdd(lock, 0);

  PRINT(LD_IO, "call omp_test_lock() return %d\n", ret);

  return ret;
}

// for xlf Fotran
// Fotran, the return is LOGICAL type

#define FLOGICAL long
EXTERN FLOGICAL __xlf_omp_is_initial_device_i8() {
  int ret = omp_is_initial_device();
  if (ret == 0)
    return (FLOGICAL)0;
  else
    return (FLOGICAL)1;
}

EXTERN int __xlf_omp_is_initial_device_i4() {
  int ret = omp_is_initial_device();
  if (ret == 0)
    return 0;
  else
    return 1;
}

EXTERN long __xlf_omp_get_team_num_i4() {
  int ret = omp_get_team_num();
  return (long)ret;
}

EXTERN long __xlf_omp_get_num_teams_i4() {
  int ret = omp_get_num_teams();
  return (long)ret;
}

EXTERN void xlf_debug_print_int(int *p) {
  printf("xlf DEBUG %d): %p %d\n", omp_get_team_num(), p, p == 0 ? 0 : *p);
}

EXTERN void xlf_debug_print_long(long *p) {
  printf("xlf DEBUG %d): %p %ld\n", omp_get_team_num(), p, p == 0 ? 0 : *p);
}

EXTERN void xlf_debug_print_float(float *p) {
  printf("xlf DEBUG %d): %p %f\n", omp_get_team_num(), p, p == 0 ? 0 : *p);
}

EXTERN void xlf_debug_print_double(double *p) {
  printf("xlf DEBUG %d): %p %f\n", omp_get_team_num(), p, p == 0 ? 0 : *p);
}

EXTERN void xlf_debug_print_addr(void *p) {
  printf("xlf DEBUG %d): %p \n", omp_get_team_num(), p);
}
