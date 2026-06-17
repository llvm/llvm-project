; RUN: llc -mtriple=riscv64-unknown-linux-gnu < %s | FileCheck %s

; This should work fine, because the module asm specifies the necessary
; target features

; CHECK: fld ft0, 0(sp)

module asm(target_features="+d")
    ".globl func"
    "func:"
    "fld f0, 0(sp)"
    "ret"
