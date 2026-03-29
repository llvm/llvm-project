// RUN: not llvm-mc -triple=aarch64 -mattr=+mpamv2 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

//------------------------------------------------------------------------------
// Armv9.7-A FEAT_MPAMV2 Extensions
//------------------------------------------------------------------------------

mlbi alle1, x30
// CHECK-ERROR: error: specified mlbi op does not use a register

mlbi vmalle1, x30
// CHECK-ERROR: error: specified mlbi op does not use a register

mlbi vpide1
// CHECK-ERROR: error: specified mlbi op requires a register

mlbi vpmge1
// CHECK-ERROR: error: specified mlbi op requires a register
