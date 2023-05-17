// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+rasv2 < %s 2>&1| FileCheck %s

msr ERXGSR_EL1, x0
// CHECK: [[@LINE-1]]:5: error: expected writable system register or pstate
