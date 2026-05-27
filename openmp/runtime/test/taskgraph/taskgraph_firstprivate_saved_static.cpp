// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// OpenMP 6.0 [14.3]: the 'saved' modifier on a 'firstprivate' clause of a
// replayable construct extends the saved data environment to also include
// copies of variables with static storage duration that appear in the
// clause.  Within the current Clang/libomp implementation those snapshots
// live in the per-task '.kmp_privates.t' tail struct, which is allocated
// once at recording time and reused for every replay of the recorded task.
// This test exercises four flavours of static-storage list items and
// verifies that the saved snapshot is what each replay observes, regardless
// of any subsequent mutation of the underlying static variable between
// taskgraph encounters.

#include <cstdio>

static int FileScopeStaticInt = 100;
static const int FileScopeConstStaticInt = 200;

struct WithStaticMember {
  static int StaticMember;
  static const int StaticConstMember = 400;
};
int WithStaticMember::StaticMember = 300;
// Out-of-line definition is required (pre-C++17) because the saved
// firstprivate slot captures the static by-value via odr-use.
const int WithStaticMember::StaticConstMember;

__attribute__((noinline)) static void
run_taskgraph_saved_static(int *out_fs, int *out_fsc, int *out_local,
                           int *out_member, int *out_member_const) {
  static int LocalStaticInt = 500;

#pragma omp taskgraph graph_id(811)
  {
#pragma omp task firstprivate(saved : FileScopeStaticInt,                      \
                                  FileScopeConstStaticInt, LocalStaticInt,     \
                                  WithStaticMember::StaticMember,              \
                                  WithStaticMember::StaticConstMember)         \
    shared(out_fs, out_fsc, out_local, out_member, out_member_const)
    {
      *out_fs = FileScopeStaticInt;
      *out_fsc = FileScopeConstStaticInt;
      *out_local = LocalStaticInt;
      *out_member = WithStaticMember::StaticMember;
      *out_member_const = WithStaticMember::StaticConstMember;
    }
  }
}

int main() {
  bool failed = false;
  int fs = -1, fsc = -1, local = -1, member = -1, member_const = -1;

  // First call: recording.  Each captured-by-saved static is snapshotted
  // into the task's '.kmp_privates.t' slot at this point.
  run_taskgraph_saved_static(&fs, &fsc, &local, &member, &member_const);
  if (fs != 100 || fsc != 200 || local != 500 || member != 300 ||
      member_const != 400) {
    std::fprintf(stderr,
                 "FAIL initial record fs=%d fsc=%d local=%d member=%d "
                 "member_const=%d\n",
                 fs, fsc, local, member, member_const);
    failed = true;
  }

  // Mutate the underlying non-const statics.  Because the task's firstprivate
  // slots were snapshotted with 'saved:' at recording, every subsequent
  // replay must continue to observe the recorded values (100, 500, 300) for
  // the non-const statics, not the mutated values.
  FileScopeStaticInt = 11;
  WithStaticMember::StaticMember = 13;
  // LocalStaticInt is not visible here; we can rely on the fact that the
  // function-local static is also snapshotted at recording.

  for (int i = 0; i < 4; ++i) {
    fs = fsc = local = member = member_const = -1;
    run_taskgraph_saved_static(&fs, &fsc, &local, &member, &member_const);
    if (fs != 100 || fsc != 200 || local != 500 || member != 300 ||
        member_const != 400) {
      std::fprintf(stderr,
                   "FAIL replay %d fs=%d fsc=%d local=%d member=%d "
                   "member_const=%d\n",
                   i, fs, fsc, local, member, member_const);
      failed = true;
    }
  }

  if (failed)
    return 1;

  std::fprintf(stderr,
               "PASS firstprivate(saved) statics persist across replays\n");
  return 0;
}

// CHECK: PASS firstprivate(saved) statics persist across replays
