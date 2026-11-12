// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: 4294901760

def X {
  int Y = 0xFFFF0000;
}
