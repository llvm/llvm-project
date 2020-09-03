// REQUIRES: lld

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -gno-pubnames
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=f.o --regex --find=function %t | FileCheck %s
//
// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=f.o --regex --find=function %t | FileCheck %s

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --name=f.o --regex --find=function %t | FileCheck %s

// NAMES: Name: .debug_names

// CHECK: Found 3 functions:
// CHECK-DAG: name = "foo()", mangled = "_Z3foov"
// CHECK-DAG: name = "ffo()", mangled = "_Z3ffov"
// CHECK-DAG: name = "bar::foo()", mangled = "_ZN3bar3fooEv"

void foo() {}
void ffo() {}
namespace bar {
void foo() {}
} // namespace bar
void fof() {}

extern "C" void _start() {}
