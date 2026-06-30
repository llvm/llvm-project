; RUN: llvm-link %s %p/Inputs/module-asm.ll -S | FileCheck %s

; CHECK: module asm(target_features: "+foo")
; CHECK-NEXT: "asm 1"
; CHECK: module asm(target_features: "+bar")
; CHECK-NEXT: "asm 2"

module asm(target_features: "+foo")
    "asm 1"
