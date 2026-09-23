// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// default(firstprivate) on a replayable taskloop.  'x' becomes (plain)
// firstprivate and is therefore snapshotted at recording and reused on every
// replay; 'SavedBump' is captured via firstprivate(saved) (an aggregate, so the
// aggregate save path is exercised); the loop iteration variable is
// predetermined private; results are written both through an explicitly shared
// output array (which must be relocated to the current call on replay) and a
// reduction.
//
// The reduction here is a regression guard: default(firstprivate) previously
// caused the enclosing taskgraph region to capture the reduction variable by
// copy, so the reduced value never propagated back to the caller (it read as
// 0).
//
// The shared output array is a local of run_default_firstprivate_taskloop
// rather than the caller's array, and the taskgraph is entered through
// call_with_depth() at a different recursion depth per encounter (with
// clobber_stack() overwriting the abandoned frames in between).  That is what
// makes the relocation of the shared array load-bearing: reached at one fixed
// depth the frame would recur at one address and a replay writing through the
// recorded (stale) shareds pointer would hit the live array by accident.  Using
// a local rather than a caller-supplied pointer also means a missing relocation
// shows up as a stale write to a dead frame (leaving the -1 initialiser behind)
// instead of a wild store through a clobbered pointer.

#include <cstdio>

struct Bump {
  int a, b;
};

static Bump SavedBump = {5, 7}; // a+b == 12

static volatile int StackSink = 0;

__attribute__((noinline)) static void clobber_stack(int base) {
  volatile int scratch[4096];

  for (int i = 0; i < 4096; ++i)
    scratch[i] = base + i;

  StackSink += scratch[base & 63];
}

__attribute__((noinline)) static int
run_default_firstprivate_taskloop(int seed, int *out_arr) {
  int x = seed;
  int total = 0;
  int local_out[8];
  for (int i = 0; i < 8; ++i)
    local_out[i] = -1;

#pragma omp taskgraph graph_id(9106)
  {
#pragma omp taskloop replayable num_tasks(4) default(firstprivate)             \
    shared(local_out) firstprivate(saved : SavedBump) reduction(+ : total)
    for (int i = 0; i < 8; ++i) {
      int v = x + i + SavedBump.a + SavedBump.b;
      local_out[i] = v;
      total += v;
    }
  }

  for (int i = 0; i < 8; ++i)
    out_arr[i] = local_out[i];

  return total;
}

__attribute__((noinline)) static int call_with_depth(int seed, int *out_arr,
                                                     int depth) {
  volatile int padding[128];

  for (int i = 0; i < 128; ++i)
    padding[i] = seed + depth + i;

  StackSink += padding[(seed + depth) & 127];

  if (depth == 0)
    return run_default_firstprivate_taskloop(seed, out_arr);
  return call_with_depth(seed, out_arr, depth - 1);
}

static bool check(const int *arr, int base, const char *tag) {
  for (int i = 0; i < 8; ++i) {
    if (arr[i] != base + i) {
      std::fprintf(stderr,
                   "FAIL default(firstprivate) taskloop %s arr[%d]=%d "
                   "expected=%d\n",
                   tag, i, arr[i], base + i);
      return false;
    }
  }
  return true;
}

int main() {
  bool ok = true;
  int o1[8], o2[8];
  for (int i = 0; i < 8; ++i) {
    o1[i] = -1;
    o2[i] = -1;
  }

  // Recording: out_arr[i] = x(10) + i + 12 = 22 + i; total = sum(22+i) = 204.
  const int t1 = call_with_depth(10, o1, 0);
  ok &= check(o1, 22, "record");
  if (t1 != 204) {
    std::fprintf(stderr,
                 "FAIL default(firstprivate) taskloop record total=%d "
                 "expected=204\n",
                 t1);
    ok = false;
  }

  // Mutate the static; the plain firstprivate 'x' (frozen at 10) and the saved
  // 'SavedBump' (frozen at sum 12) must keep their recorded values, so the
  // replay still produces 22 + i (into the current call's array, reached at a
  // different stack depth) and the reduction total is again 204.
  SavedBump = Bump{100, 100};

  clobber_stack(99 * 1000);
  const int t2 = call_with_depth(99, o2, 4);
  ok &= check(o2, 22, "replay");
  if (t2 != 204) {
    std::fprintf(stderr,
                 "FAIL default(firstprivate) taskloop replay total=%d "
                 "expected=204\n",
                 t2);
    ok = false;
  }

  if (!ok)
    return 1;

  std::fprintf(stderr, "PASS default(firstprivate) taskloop total=%d\n", t1);
  return 0;
}

// CHECK: PASS default(firstprivate) taskloop total=204
