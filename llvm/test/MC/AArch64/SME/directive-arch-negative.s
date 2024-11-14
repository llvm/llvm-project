// RUN: not llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sme+nosme
zero {za}
// CHECK: error: instruction requires: sme
// CHECK-NEXT: zero {za}

.arch armv9-a+sme-f64f64+nosme-f64f64
fmopa za0.d, p0/m, p0/m, z0.d, z0.d
// CHECK: error: instruction requires: sme-f64f64
// CHECK-NEXT: fmopa za0.d, p0/m, p0/m, z0.d, z0.d

.arch armv9-a+sme-i16i64+nosme-i16i64
addha za0.d, p0/m, p0/m, z0.d
// CHECK: error: instruction requires: sme-i16i64
// CHECK-NEXT: addha za0.d, p0/m, p0/m, z0.d