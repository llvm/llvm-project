// RUN: %libomp-cxx-compile-and-run
// REQUIRES: linux
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <new>
#include <omp.h>

// Call the GNU compatibility entry directly so this regression also runs with
// Clang, whose task lowering otherwise exercises the native entry points.
extern "C" void GOMP_task(void (*)(void *), void *, void (*)(void *, void *),
                          long, long, bool, unsigned, void **);
extern "C" void GOMP_taskwait();

static std::atomic<int> Copies{0}, Destructions{0}, Errors{0};
static int ExpectedCopyFinal;

struct Value {
  int number;
  bool is_private;
  explicit Value(int n) : number(n), is_private(false) {}
  Value(const Value &other) : number(other.number + 100), is_private(true) {
    if (omp_in_final() != ExpectedCopyFinal)
      ++Errors;
    ++Copies;
  }
  ~Value() {
    if (is_private)
      ++Destructions;
  }
};

// GCC may pass a pointer to the original non-trivial firstprivate object in
// the source environment and construct the object inline in the copied one.
struct Source { Value *original; int *observed; };
struct alignas(128) Copied { Value value; int *observed; };

static void copy(void *destination, void *source) {
  auto *from = static_cast<Source *>(source);
  if (reinterpret_cast<std::uintptr_t>(destination) % alignof(Copied))
    ++Errors;
  new (destination) Copied{*from->original, from->observed};
}

static void run_copied(void *data) {
  // Fail cleanly on an unfixed runtime, without interpreting the source
  // environment as a Copied object when its constructor never ran.
  if (Copies.load() != 1) {
    ++Errors;
    return;
  }
  auto *arg = static_cast<Copied *>(data);
  *arg->observed = arg->value.number;
  arg->~Copied();
}

static void check_copy(bool deferred, unsigned flags, int copy_final) {
  Copies = 0;
  Destructions = 0;
  ExpectedCopyFinal = copy_final;
  Value original(7);
  int observed = 0;
  Source source{&original, &observed};
  GOMP_task(run_copied, &source, copy, sizeof(Copied), alignof(Copied),
            deferred, flags, nullptr);
  GOMP_taskwait();
  if (Copies != 1 || Destructions != 1 || observed != 107 ||
      original.number != 7)
    ++Errors;
}

static void *ExpectedData;
static void run_uncopied(void *data) {
  if (data != ExpectedData)
    ++Errors;
  if (data)
    ++*static_cast<int *>(data);
}

int main() {
  omp_set_dynamic(0);
  for (int threads = 1; threads <= 2; ++threads) {
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
      {
        if (omp_get_num_threads() != threads)
          ++Errors;
        check_copy(false, 0, 0);
        // GOMP_TASK_FLAG_FINAL: the copy constructor must run in the
        // undeferred task's context, not in its non-final parent.
        check_copy(false, 2, 1);
        check_copy(true, 0, 0);
        int value = 19;
        ExpectedData = &value;
        GOMP_task(run_uncopied, &value, nullptr, sizeof(value), alignof(int),
                  false, 0, nullptr);
        if (value != 20)
          ++Errors;
        ExpectedData = nullptr;
        GOMP_task(run_uncopied, nullptr, nullptr, 0, 1, false, 0, nullptr);
      }
    }
  }
  std::printf("GOMP_IF0_COPY errors=%d\n", Errors.load());
  return Errors ? 1 : 0;
}
