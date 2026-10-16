// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR

msr MPAMBWIDR_EL1, x0
// CHECK-ERROR: error: expected writable system register or pstate