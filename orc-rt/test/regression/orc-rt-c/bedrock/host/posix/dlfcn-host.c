// Check that JIT'd code can call each of the dlfcn-host.h forwarders.
//
// Stresses: resolution of JIT'd code's references to the ORC runtime's C API
// symbols, which are defined in the executor rather than in JIT'd code. This
// depends on the symbols being exported from the runtime, and on the executor
// making them visible to the JIT linker.
//
// This test only checks that each forwarder can be reached and called. The
// forwarders' behavior is tested by the unit tests in
// test/unit/bedrock/host/posix/DlfcnHostTest.cpp.
//
// Unlike language tests, this test includes a header (the API under test, which
// includes <dlfcn.h>), so it needs the sysroot that %{cc} provides. The path of
// the shared library to load is passed as the program's last argument: take it
// from the end of argv, since llvm-jitlink does not supply a program name as
// argv[0].
//
// RUN: %{cc} %{orc-rt-cflags} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o -args %{test-dylib} | FileCheck %s
// RUN: %{cc} %{orc-rt-cflags} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o -args %{test-dylib} | FileCheck %s

// CHECK: JIT result: 0

#include "orc-rt-c/bedrock/host/posix/dlfcn-host.h"

int main(int argc, char *argv[]) {
  if (argc < 1)
    return 1;

  void *Lib = orc_rt_dlopen_host(argv[argc - 1], RTLD_NOW | RTLD_LOCAL);
  if (!Lib)
    return 2;

  int (*Return42)(void) =
      (int (*)(void))orc_rt_dlsym_host(Lib, "orc_rt_test_dylib_return42");
  if (!Return42 || Return42() != 42)
    return 3;

  Dl_info Info;
  if (!orc_rt_dladdr_host((void *)Return42, &Info))
    return 4;

  // A failed lookup, so that dlerror has an error to report.
  if (orc_rt_dlsym_host(Lib, "orc_rt_test_dylib_no_such_function"))
    return 5;
  if (!orc_rt_dlerror_host())
    return 6;

  if (orc_rt_dlclose_host(Lib) != 0)
    return 7;

  return 0;
}
