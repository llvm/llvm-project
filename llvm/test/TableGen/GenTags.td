// RUN: llvm-tblgen --gen-ctags %s | FileCheck %s -DFILE=%s

// CHECK: A [[FILE]] [[@LINE+1]]
class A;

// CHECK: A [[FILE]] [[@LINE+1]]
class A {
  string name = "A";
}
