; RUN: llc --filetype=obj %s -o %t.dxbc
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

; CHECK: target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
; CHECK: target triple = "dxil-ms-dx"
; CHECK: define half @add(half %a, half %b)
; CHECK: %sum = fadd half %a, %b
; CHECK: ret half %sum

target triple = "dxil-unknown-shadermodel6.7-library"

define half @add(half %a, half %b) "hlsl.export" {
  %sum = fadd half %a, %b
  ret half %sum
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.nativelowprec", i32 0}
