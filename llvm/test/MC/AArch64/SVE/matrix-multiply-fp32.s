// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+f32mm < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+f32mm < %s \
// RUN:        | llvm-objdump -d --mattr=+sve,+f32mm - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+f32mm < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// FMMLA (SVE)

fmmla z0.s, z1.s, z2.s
// CHECK-INST: fmmla z0.s, z1.s, z2.s
// CHECK-ENCODING: [0x20,0xe4,0xa2,0x64]
// CHECK-ERROR: instruction requires: f32mm sve
// CHECK-UNKNOWN: 64a2e420 <unknown>
