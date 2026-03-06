// RUN: not llvm-mc -triple aarch64_lfi %s 2>&1 | FileCheck %s

.arch_extension pauth

eret
// CHECK: error: exception returns (ERET/ERETAA/ERETAB) are not supported by LFI

eretaa
// CHECK: error: exception returns (ERET/ERETAA/ERETAB) are not supported by LFI

eretab
// CHECK: error: exception returns (ERET/ERETAA/ERETAB) are not supported by LFI

