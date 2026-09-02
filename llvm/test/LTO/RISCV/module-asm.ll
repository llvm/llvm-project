; RUN: llvm-as %s -o %t.o
; RUN: llvm-lto2 run -save-temps -filetype=asm -o %t.s %t.o -r=%t.o,func,p
; RUN: llvm-nm %t.o | FileCheck %s --check-prefix NM
; RUN: llvm-nm %t.s.0.5.precodegen.bc | FileCheck %s --check-prefix NM
; RUN: FileCheck %s --input-file %t.s.0

; NM: T func

; CHECK: fld ft0, 0(sp)

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

module asm(target_features: "+d")
    ".globl func"
    "func:"
    "fld f0, 0(sp)"
    "ret"
