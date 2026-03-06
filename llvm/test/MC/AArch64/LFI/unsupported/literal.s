// RUN: not llvm-mc -triple aarch64_lfi %s 2>&1 | FileCheck %s

ldr x0, label
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr w0, label
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr s0, label
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr d0, label
// CHECK: error: PC-relative literal loads are not supported in LFI

ldr q0, label
// CHECK: error: PC-relative literal loads are not supported in LFI

ldrsw x0, label
// CHECK: error: PC-relative literal loads are not supported in LFI

prfm pldl1keep, label
// CHECK: error: PC-relative literal loads are not supported in LFI

label:
.word 0x12345678

