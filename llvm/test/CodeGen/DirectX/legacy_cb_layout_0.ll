; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s --check-prefix=DXILMD

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

; Make sure the size is 36 = 16 + 16 + 4 (float, double -> 16, float, half, i16, i64 -> 16 and int -> 4)
; DXILMD:!{i32 0, ptr @A.cb., !"", i32 0, i32 2, i32 1, i32 36}

@A.cb. = external local_unnamed_addr constant { float, double, float, half, i16, i64, i32 }


!hlsl.cbufs = !{!1}

!1 = !{ptr @A.cb., !"A.cb.ty", i32 13, i1 false, i32 2, i32 0}
