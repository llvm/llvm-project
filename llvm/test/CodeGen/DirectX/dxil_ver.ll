; RUN: opt -S -dxil-metadata-emit < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.3-library"

; Make sure dx.valver metadata is generated.
; CHECK:!dx.valver = !{![[valver:[0-9]+]]}
; Make sure module flags still exist and only have 1 operand left.
; CHECK:!llvm.module.flags = !{{{![0-9]}}}
; Make sure validator version is 1.1.
; CHECK:![[valver]] = !{i32 1, i32 1}
; Make sure wchar_size still exist.
; CHECK:!{i32 1, !"wchar_size", i32 4}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 6, !"dx.valver", !2}
!2 = !{i32 1, i32 1}
!3 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project 71de12113a0661649ecb2f533fba4a2818a1ad68)"}
