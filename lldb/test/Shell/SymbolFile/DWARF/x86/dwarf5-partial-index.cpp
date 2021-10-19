// Test that we return complete results when only a part of the binary is built
// with an index.

// REQUIRES: lld

// RUN: %clang %s -c -o %t-1.o --target=x86_64-pc-linux -DONE -gdwarf-5 -gpubnames
// RUN: llvm-readobj --sections %t-1.o | FileCheck %s --check-prefix NAMES
// RUN: %clang %s -c -o %t-2.o --target=x86_64-pc-linux -DTWO -gdwarf-5 -gno-pubnames
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: lldb-test symbols --find=variable --name=foo  %t | FileCheck %s

// NAMES: Name: .debug_names

// CHECK: Found 2 variables:
#ifdef ONE
namespace one {
int foo;
// CHECK-DAG: name = "foo", {{.*}} decl = dwarf5-partial-index.cpp:[[@LINE-1]]
} // namespace one
extern "C" void _start() {}
#else
namespace two {
int foo;
// CHECK-DAG: name = "foo", {{.*}} decl = dwarf5-partial-index.cpp:[[@LINE-1]]
} // namespace two
#endif
