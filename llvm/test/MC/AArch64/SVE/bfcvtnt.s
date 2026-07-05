// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+bf16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme,+bf16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

bfcvtnt z0.H, p0/m, z1.S
// CHECK-INST: bfcvtnt z0.h, p0/m, z1.s
// CHECK-ENCODING: [0x20,0xa0,0x8a,0x64]
// CHECK-ERROR: instruction requires: bf16 sve or sme
