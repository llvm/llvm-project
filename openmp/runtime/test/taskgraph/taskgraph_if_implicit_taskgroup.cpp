// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A taskgraph region is a taskgroup region whether or not it records: the 'if'
// clause governs recording, and 'nogroup' is the only way to drop the implicit
// taskgroup.  So the tasks a taskgraph generates must have completed by the end
// of the region for *both* values of the condition.
//
// The recording path gets its taskgroup from __kmpc_taskgraph.  The if(false)
// path bypasses that call entirely, and used to be emitted as a bare call to
// the outlined region with no taskgroup at all, so the generated tasks were
// still running after the region had ended -- silent under-synchronisation
// rather than a crash, hence this test.
//
// Each task sleeps before reporting, so an un-waited-for region is caught
// rather than merely raced on: with the taskgroup missing, the check below
// reliably sees a count of zero.

#include <atomic>
#include <cstdio>
#include <unistd.h>

static constexpr int NumIters = 4;
static constexpr int NumTasks = 8;

static std::atomic<int> done;

// Returns the number of tasks that had not finished by the end of the region.
static int missing_at_end_of_region(bool record) {
  int outstanding = 0;

  for (int iter = 0; iter < NumIters; ++iter) {
    done.store(0, std::memory_order_relaxed);

    if (record) {
#pragma omp taskgraph graph_id(1) if (true)
      {
        for (int i = 0; i < NumTasks; ++i) {
#pragma omp task
          {
            usleep(1000);
            done.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    } else {
#pragma omp taskgraph graph_id(2) if (false)
      {
        for (int i = 0; i < NumTasks; ++i) {
#pragma omp task
          {
            usleep(1000);
            done.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    }

    outstanding += NumTasks - done.load(std::memory_order_relaxed);
  }

  return outstanding;
}

int main() {
  int recorded = -1, not_recorded = -1;

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      recorded = missing_at_end_of_region(/*record=*/true);
      not_recorded = missing_at_end_of_region(/*record=*/false);
    }
  }

  if (recorded != 0 || not_recorded != 0) {
    std::fprintf(stderr,
                 "FAIL taskgraph if: outstanding tasks if(true)=%d "
                 "if(false)=%d, expected 0 and 0\n",
                 recorded, not_recorded);
    return 1;
  }

  std::fprintf(stderr, "PASS taskgraph if implicit taskgroup\n");
  return 0;
}

// CHECK: PASS taskgraph if implicit taskgroup
