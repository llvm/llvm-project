// Test for a bug where we crashed while processing a compressed debug_names
// section (use after free).

// REQUIRES: lld, zlib

// RUN: %clang -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames %s
// RUN: ld.lld %t.o -o %t --compress-debug-sections=zlib
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --find=variable --name=foo %t | FileCheck %s

// NAMES: Name: .debug_names

// CHECK: Found 1 variables:
int foo;
// CHECK-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = debug-names-compressed.cpp:[[@LINE-1]]

extern "C" void _start() {}
