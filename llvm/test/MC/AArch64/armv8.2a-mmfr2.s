// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.2a < %s 2>&1 | FileCheck %s --check-prefix=ERROR

  mrs x3, id_aa64mmfr2_el1
// CHECK: mrs x3, ID_AA64MMFR2_EL1       // encoding: [0x43,0x07,0x38,0xd5]
// ERROR: error: expected readable system register
