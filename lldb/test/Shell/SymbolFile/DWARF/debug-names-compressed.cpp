// Test for a bug where we crashed while processing a compressed debug_names
// section (use after free).

// REQUIRES: lld, zlib

// RUN: %clang -g -c -o %t.o --target=x86_64-pc-linux -mllvm -accel-tables=Dwarf %s
// RUN: ld.lld %t.o -o %t --compress-debug-sections=zlib
// RUN: lldb-test symbols --find=variable --name=foo %t | FileCheck %s

// CHECK: Found 1 variables:
int foo;
// ONE-DAG: name = "foo", type = {{.*}} (int), {{.*}} decl = debug-names-compressed.cpp:[[@LINE-1]]

extern "C" void _start() {}
