// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+sme < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+sme-f64 < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+sme-i64 < %s | FileCheck %s

// Verify +sme flags imply streaming compatible SVE instructions.
tbx z0.b, z1.b, z2.b
// CHECK: tbx z0.b, z1.b, z2.b

// Verify +sme flags imply +bf16
bfdot z0.s, z1.h, z2.h
// CHECK-INST: bfdot z0.s, z1.h, z2.h
