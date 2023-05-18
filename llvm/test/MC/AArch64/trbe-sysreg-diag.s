// TRBE RO System register
//
// RUN: not llvm-mc -triple aarch64 -show-encoding < %s 2>&1 | FileCheck %s

// Write to system register
msr TRBIDR_EL1, x0

// CHECK:      expected writable system register or pstate
// CHECK-NEXT: msr TRBIDR_EL1, x0
