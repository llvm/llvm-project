// REQUIRES: lld

// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -g -c %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-dwp %t.dwo -o %t.dwp
// RUN: rm %t.dwo
// RUN: llvm-objcopy --only-keep-debug %t %t.debug
// RUN: llvm-objcopy --strip-all --add-gnu-debuglink=%t.debug %t
// RUN: %lldb %t -o "target variable a" -b | FileCheck %s

// CHECK: (A) a = (x = 47)

struct A {
  int x = 47;
};
A a;
int main() {}
