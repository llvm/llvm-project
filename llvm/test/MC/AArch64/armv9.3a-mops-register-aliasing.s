// RUN: not llvm-mc -triple aarch64 -mattr=+mops < %s 2>&1 | FileCheck %s

  setp  [x0]!, x1!, x1
  setm  [x0]!, x1!, x1
  sete  [x0]!, x1!, x1

// CHECK:      error: invalid SET instruction, source and size registers are the same
// CHECK-NEXT:   setp  [x0]!, x1!, x1
// CHECK-NEXT:         ^
// CHECK-NEXT: error: invalid SET instruction, source and size registers are the same
// CHECK-NEXT:   setm  [x0]!, x1!, x1
// CHECK-NEXT:         ^
// CHECK-NEXT: error: invalid SET instruction, source and size registers are the same
// CHECK-NEXT:   sete  [x0]!, x1!, x1
// CHECK-NEXT:         ^
