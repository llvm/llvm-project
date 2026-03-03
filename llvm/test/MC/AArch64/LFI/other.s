// RUN: llvm-mc -triple aarch64_lfi %s | FileCheck %s

.lfi_rewrite_disable
ldr x0, [x1]
// CHECK: ldr x0, [x1]
.lfi_rewrite_enable
