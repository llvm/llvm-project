// RUN: not llvm-mc -triple=aarch64 -mattr=+mtetc -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-REQUIRES-MTETC

//------------------------------------------------------------------------------
// FEAT_MTETC Extension instructions
//------------------------------------------------------------------------------

dc zgbva
// CHECK-ERROR: error: specified dc op requires a register
// CHECK-REQUIRES-MTETC: DC ZGBVA requires: mtetc

dc gbva
// CHECK-ERROR: error: specified dc op requires a register
// CHECK-REQUIRES-MTETC: DC GBVA requires: mtetc
