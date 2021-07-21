// RUN: not llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch_extension sme
.arch_extension nosme
smstart
// CHECK: error: instruction requires: sme
// CHECK-NEXT: smstart

.arch_extension sme-f64
.arch_extension nosme-f64
fmopa za0.d, p0/m, p0/m, z0.d, z0.d
// CHECK: error: instruction requires: sme-f64
// CHECK-NEXT: fmopa za0.d, p0/m, p0/m, z0.d, z0.d

.arch_extension sme-i64
.arch_extension nosme-i64
addha za0.d, p0/m, p0/m, z0.d
// CHECK: error: instruction requires: sme-i64
// CHECK-NEXT: addha za0.d, p0/m, p0/m, z0.d

.arch armv8-a+sme
.arch armv8-a+nosme
smstart
// CHECK: error: instruction requires: sme
// CHECK-NEXT: smstart

.arch armv8-a+sme-f64
.arch armv8-a+nosme-f64
fmopa za0.d, p0/m, p0/m, z0.d, z0.d
// CHECK: error: instruction requires: sme-f64
// CHECK-NEXT: fmopa za0.d, p0/m, p0/m, z0.d, z0.d

.arch armv8-a+sme-i64
.arch armv8-a+nosme-i64
addha za0.d, p0/m, p0/m, z0.d
// CHECK: error: instruction requires: sme-i64
// CHECK-NEXT: addha za0.d, p0/m, p0/m, z0.d
