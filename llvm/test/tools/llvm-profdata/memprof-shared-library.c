// REQUIRES: x86_64-linux
// RUN: clang -fPIC -shared -Wl,-z,separate-code -Wl,--build-id %s -o %t.separate.so
// RUN: llvm-readelf -lW %t.separate.so | FileCheck %s --check-prefix=SEPARATE
// RUN: not llvm-profdata show --memory %p/Inputs/pic.memprofraw --profiled-binary %t.separate.so -o - 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: clang -nostdlib -fPIC -shared -fno-asynchronous-unwind-tables -fno-unwind-tables -Wl,-z,max-page-size=0x1 -Wl,-z,noseparate-code -Wl,-Ttext=0x1870 -Wl,--build-id %s -o %t.unaligned.so
// RUN: llvm-readelf -lW %t.unaligned.so | FileCheck %s --check-prefix=UNALIGNED
// RUN: not llvm-profdata show --memory %p/Inputs/pic.memprofraw --profiled-binary %t.unaligned.so -o - 2>&1 | FileCheck %s --check-prefix=CHECK

// These libraries are never executed. The unrelated profile should reach the
// normal build ID check instead of tripping symbolization assertions.
int shared_library_function(void) { return 42; }

// The first layout has an executable segment at nonzero file offset and
// preferred virtual address, as produced by -z separate-code.
// SEPARATE: LOAD{{ +}}0x{{[0-9a-f]+}} 0x{{[0-9a-f]+}} {{.*}} R E
// The second layout forces an executable segment with a non-page-aligned
// preferred virtual address.
// UNALIGNED: LOAD{{ +}}0x{{[0-9a-f]+}} 0x0000000000001870 {{.*}} R E
// CHECK: error: No matching executable segments found in binary