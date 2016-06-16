//===- afl_driver.cpp - a glue between AFL and libFuzzer --------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//

/* This file allows to fuzz libFuzzer-style target functions
 (LLVMFuzzerTestOneInput) with AFL using AFL's persistent (in-process) mode.

Usage:
################################################################################
cat << EOF > test_fuzzer.cc
#include <stdint.h>
#include <stddef.h>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > 0 && data[0] == 'H')
    if (size > 1 && data[1] == 'I')
       if (size > 2 && data[2] == '!')
       __builtin_trap();
  return 0;
}
EOF
# Build your target with -fsanitize-coverage=trace-pc using fresh clang.
clang -g -fsanitize-coverage=trace-pc test_fuzzer.cc -c
# Build afl-llvm-rt.o.c from the AFL distribution.
clang -c -w $AFL_HOME/llvm_mode/afl-llvm-rt.o.c
# Build this file, link it with afl-llvm-rt.o.o and the target code.
clang++ afl_driver.cpp test_fuzzer.o afl-llvm-rt.o.o
# Run AFL:
rm -rf IN OUT; mkdir IN OUT; echo z > IN/z;
$AFL_HOME/afl-fuzz -i IN -o OUT ./a.out
################################################################################
*/
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// libFuzzer interface is thin, so we don't include any libFuzzer headers.
extern "C" {
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
__attribute__((weak)) int LLVMFuzzerInitialize(int *argc, char ***argv);
}

// Notify AFL about persistent mode.
static volatile char AFL_PERSISTENT[] = "##SIG_AFL_PERSISTENT##";
extern "C" int __afl_persistent_loop(unsigned int);
static volatile char suppress_warning2 = AFL_PERSISTENT[0];

// Notify AFL about deferred forkserver.
static volatile char AFL_DEFER_FORKSVR[] = "##SIG_AFL_DEFER_FORKSRV##";
extern "C" void  __afl_manual_init();
static volatile char suppress_warning1 = AFL_DEFER_FORKSVR[0];

// Input buffer.
static const size_t kMaxAflInputSize = 1 << 20;
static uint8_t AflInputBuf[kMaxAflInputSize];

// If the user asks us to duplicate stderr, then do it.
static void maybe_duplicate_stderr() {
  char* stderr_duplicate_filename =
      getenv("AFL_DRIVER_STDERR_DUPLICATE_FILENAME");

  if (!stderr_duplicate_filename)
    return;

  FILE* stderr_duplicate_stream =
      freopen(stderr_duplicate_filename, "a+", stderr);

  if (!stderr_duplicate_stream) {
    fprintf(stderr,
            "Failed to duplicate stderr to AFL_DRIVER_STDERR_DUPLICATE_FILENAME"
            );
    abort();
  }
}

int main(int argc, char **argv) {
  fprintf(stderr, "Running in AFl-fuzz mode\nUsage:\n"
                  "afl-fuzz [afl-flags] %s [N] "
                  "-- run N fuzzing iterations before "
                  "re-spawning the process (default: 1000)\n",
          argv[0]);
  if (LLVMFuzzerInitialize)
    LLVMFuzzerInitialize(&argc, &argv);
  // Do any other expensive one-time initialization here.

  maybe_duplicate_stderr();

  __afl_manual_init();

  int N = 1000;
  if (argc >= 2)
    N = atoi(argv[1]);
  assert(N > 0);
  while (__afl_persistent_loop(N)) {
    ssize_t n_read = read(0, AflInputBuf, kMaxAflInputSize);
    if (n_read > 0) {
      // Copy AflInputBuf into a separate buffer to let asan find buffer
      // overflows. Don't use unique_ptr/etc to avoid extra dependencies.
      uint8_t *copy = new uint8_t[n_read];
      memcpy(copy, AflInputBuf, n_read);
      LLVMFuzzerTestOneInput(copy, n_read);
      delete[] copy;
    }
  }
}
