// RUN: not llvm-mc -triple aarch64 -show-encoding < %s 2>&1 | FileCheck %s

mrs x0, SPMZR_EL0
// CHECK: [[@LINE-1]]:9: error: expected readable system register
