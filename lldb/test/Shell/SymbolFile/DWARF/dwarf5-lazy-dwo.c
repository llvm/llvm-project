// Test we load dwo information lazily.

// -gsplit-dwarf is supported only on Linux.
// REQUIRES: system-linux

// RUN: %clang_host %s -fno-standalone-debug -g -std=c99 \
// RUN:   -gdwarf-5 -gpubnames -gsplit-dwarf -c -o %t1.o -DONE
// RUN: %clang_host %s -fno-standalone-debug -g -std=c99 \
// RUN:   -gdwarf-5 -gpubnames -gsplit-dwarf -c -o %t2.o -DTWO
// RUN: %clang_host %t1.o %t2.o -o %t -std=c99
// RUN: %lldb %t -o "log enable 'lldb' object" -o "settings set stop-line-count-before 0" \
// RUN:   -o "b main" -o "run" -o "image lookup -n main -v" -b | FileCheck %s

// CHECK-NOT: 2.dwo,
// CHECK: (lldb) b main
// CHECK-NOT: 2.dwo,
// CHECK: 1.dwo,
// CHECK-NOT: 2.dwo,
// CHECK: (lldb) run
// CHECK-NOT: 2.dwo,
// CHECK: stop reason = breakpoint
// CHECK-NOT: 2.dwo,
// CHECK: (lldb) image lookup
// CHECK-NOT: 2.dwo,
// CHECK: CompileUnit: id =
// CHECK-SAME: /dwarf5-lazy-dwo.c", language = "c99"
// CHECK-NOT: 2.dwo,

#ifdef ONE
int main() { return 0; }
#else
int x;
#endif
