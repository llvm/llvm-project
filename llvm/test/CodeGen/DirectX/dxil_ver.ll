; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.3-library"

; Make sure dx.valver metadata is generated.
; CHECK-DAG:!dx.valver = !{![[valver:[0-9]+]]}
; Make sure module flags still exist and only have 1 operand left.
; CHECK-DAG:!llvm.module.flags = !{{{![0-9]}}}
; Make sure validator version is 1.1.
; CHECK-DAG:![[valver]] = !{i32 1, i32 1}
; Make sure wchar_size still exist.
; CHECK-DAG:!{i32 1, !"wchar_size", i32 4}

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 1}
!2 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project 71de12113a0661649ecb2f533fba4a2818a1ad68)"}
