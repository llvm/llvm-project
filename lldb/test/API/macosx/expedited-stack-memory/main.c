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

// The innermost frame, where we stop.  It carries several kind of local: a
// scalar, aggregates (struct + array), pointers (including one into heap
// memory) and a variable-length array.  Examining this single frame on a stop
// reads both stack and heap memory.
static int func_e(int depth) {
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
           heap[HEAP_COUNT - 1] + str[0] + *self; // break here
  int r = i + (int)l + (int)d + (int)stats.sum + (int)arr[3] + (int)vla[n - 1] +
          (int)heap[HEAP_COUNT - 1] + str[0] + *self;
  free(heap);
  return r;
}

// Aggregate locals: a struct and a fixed-size stack array.
static int func_d(int x) {
  struct Stats stats = {.sum = x, .min = x - 1, .max = x + 1, .mean = x + 0.5};
  long arr[4] = {x, x + 1, x + 2, x + 3};
  int r = func_e(x);
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
