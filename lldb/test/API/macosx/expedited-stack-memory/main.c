#include <stdlib.h>

// A simple, deterministic, single-threaded nested call chain.  We stop at the
// innermost function and walk the stack.
//
// The breakpoint is in the innermost frame (func_e), and that frame carries
// locals of every kind, so that examining *just the stopped frame* already
// exercises the various memory-read paths in one frame:
//   - scalar locals (int / long / double)
//   - aggregate locals (a struct and a fixed stack array)
//   - a variable-length array (dynamically sized stack storage, like alloca)
//   - pointer locals, including a pointer to heap memory
//   - stack-passed parameters
//
// The outer frames (func_d / func_c) also carry locals of these kinds, so that
// walking the whole stack and examining every frame reads the same variety of
// memory across several frames.
//
// The frame-pointer backchain is expedited at the public stop, so the backtrace
// itself is packet-free; reading the *values* of these locals is not expedited
// and must read memory from the stub.

#define HEAP_COUNT 8

// A volatile sink so the locals are observably used.
volatile long g_sink;

struct Stats {
  long sum;
  long min;
  long max;
  double mean;
};

// A large by-value struct.  Passed as an argument it does not fit in registers,
// so it is passed on the stack, above the callee's frame in the caller.
struct Big {
  long v[8];
};

// The innermost frame, where we stop.  It carries several kind of local: a
// scalar, aggregates (struct + array), pointers (including one into heap
// memory) and a variable-length array.  Examining this single frame on a stop
// reads both stack and heap memory.
static int func_e(int depth, int a1, int a2, int a3, int a4, int a5, int a6,
                  int a7, int a8, int a9, struct Big big) {
  int i = depth + 1;
  long l = (long)depth * 1000;
  double d = depth + 0.5;
  struct Stats stats = {.sum = i, .min = i - 1, .max = i + 1, .mean = d};
  long arr[4] = {i, i + 1, i + 2, i + 3};
  int n = i + 4; // a runtime bound, so vla is a true variable-length array
  long vla[n];   // dynamically sized stack storage (like alloca)
  for (int k = 0; k < n; ++k)
    vla[k] = (long)i - k;
  long *heap = (long *)malloc(sizeof(long) * HEAP_COUNT);
  for (int k = 0; k < HEAP_COUNT; ++k)
    heap[k] = (long)i + k;
  const char *str = "hello from func_e";
  int *self = &i;
  g_sink = i + l + (long)d + stats.sum + arr[3] + vla[n - 1] +
           heap[HEAP_COUNT - 1] + str[0] + *self + a8 + a9 +
           big.v[7]; // break here
  int r = i + (int)l + (int)d + (int)stats.sum + (int)arr[3] + (int)vla[n - 1] +
          (int)heap[HEAP_COUNT - 1] + str[0] + *self + a8 + a9 + (int)big.v[7];
  free(heap);
  return r;
}

// Aggregate locals: a struct and a fixed-size stack array.
static int func_d(int x) {
  struct Stats stats = {.sum = x, .min = x - 1, .max = x + 1, .mean = x + 0.5};
  long arr[4] = {x, x + 1, x + 2, x + 3};
  struct Big big;
  for (int k = 0; k < 8; ++k)
    big.v[k] = 100 + k;
  int r = func_e(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7, x + 8,
                 x + 9, big);
  return r + (int)stats.sum + (int)arr[3];
}

// Pointer locals, including a pointer into heap memory.
static int func_c(int x) {
  long *heap = (long *)malloc(sizeof(long) * HEAP_COUNT);
  for (int i = 0; i < HEAP_COUNT; ++i)
    heap[i] = (long)x + i;
  const char *str = "hello from func_c";
  int *self = &x;
  int r = func_d(x);
  r += (int)heap[HEAP_COUNT - 1] + (int)str[0] + *self;
  free(heap);
  return r;
}

static int func_b(int x) { return func_c(x) + 1; }
static int func_a(int x) { return func_b(x) + 1; }

int main() {
  g_sink = func_a(0);
  return 0;
}
