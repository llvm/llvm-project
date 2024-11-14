// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch_extension sme
smstart
// CHECK: smstart
zero {za}
// CHECK-NEXT: zero {za}

.arch_extension sme-f64f64
fmopa za0.d, p0/m, p0/m, z0.d, z0.d
// CHECK: fmopa za0.d, p0/m, p0/m, z0.d, z0.d

.arch_extension sme-i16i64
addha za0.d, p0/m, p0/m, z0.d
// CHECK: addha za0.d, p0/m, p0/m, z0.d

