// REQUIRES: lld

// RUN: %clang %s -g -c -o %t.o --target=x86_64-pc-linux -gno-pubnames
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=function --function-flags=method %t | \
// RUN:   FileCheck %s
//
// RUN: %clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=foo --find=function --function-flags=method %t | \
// RUN:   FileCheck %s

// RUN: %clang %s -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gpubnames
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj --sections %t | FileCheck %s --check-prefix NAMES
// RUN: lldb-test symbols --name=foo --find=function --function-flags=method %t | \
// RUN:   FileCheck %s

// NAMES: Name: .debug_names

// CHECK-DAG: name = "A::foo()", mangled = "_ZN1A3fooEv"
// CHECK-DAG: name = "B::foo()", mangled = "_ZN1B3fooEv"
// CHECK-DAG: name = "C::foo()", mangled = "_ZN1C3fooEv"

struct A {
  void foo();
};
void A::foo() {}

class B {
  void foo();
};
void B::foo() {}

union C {
  void foo();
};
void C::foo() {}

extern "C" void _start() {}
