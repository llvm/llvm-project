; RUN: llc -mtriple=riscv64-unknown-linux-gnu < %s | FileCheck %s --check-prefixes=CHECK,EXTRA-FEATURES
; RUN: llc -mtriple=riscv64-unknown-linux-gnu -mattr=+d < %s | FileCheck %s --check-prefixes=CHECK,SAME-FEATURES

; This should work fine, because the module asm specifies the necessary
; target features

; SAME-FEATURES-NOT: .option arch
; EXTRA-FEATURES: .option push
; EXTRA-FEATURES: .option arch, +d
; CHECK: fld ft0, 0(sp)
; EXTRA-FEATURES: .option pop

module asm(target_features: "+d")
    ".globl func"
    "func:"
    "fld f0, 0(sp)"
    "ret"
