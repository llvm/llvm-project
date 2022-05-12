; RUN: opt -S -dxil-metadata-emit < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.3-library"

; CHECK:!dx.valver = !{![[valver:[0-9]+]]}
; CHECK:![[valver]] = !{i32 1, i32 1}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 6, !"dx.valver", !2}
!2 = !{i32 1, i32 1}
!3 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project 71de12113a0661649ecb2f533fba4a2818a1ad68)"}
