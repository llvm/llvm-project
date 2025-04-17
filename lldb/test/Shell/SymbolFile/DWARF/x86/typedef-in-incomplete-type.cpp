// RUN: %clangxx --target=x86_64-pc-linux -flimit-debug-info -o %t -c %s -g
// RUN: %lldb %t -o "target var a" -o "expr -- var" -o exit | FileCheck %s

// This forces lldb to attempt to complete the type A. Since it has no
// definition it will fail.
// CHECK: target var a
// CHECK: (A) a = <incomplete type "A">

// Now attempt to display the second variable, which will try to add a typedef
// to the incomplete type. Make sure that succeeds. Use the expression command
// to make sure the resulting AST can be imported correctly.
// CHECK: expr -- var
// CHECK: (A::X) $0 = 0

struct A {
  // Declare the constructor, but don't define it to avoid emitting the
  // definition in the debug info.
  A();
  using X = int;
};

A a;
A::X var;
