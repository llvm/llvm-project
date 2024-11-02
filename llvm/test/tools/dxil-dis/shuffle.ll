; RUN: llc --filetype=obj %s -o - 2>&1 | dxil-dis -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

; Make sure alloca is the same.
; CHECK:alloca <2 x float>, align 8
; Make sure shufflevector works for DXIL bitcode writer.
; CHECK:shufflevector <2 x float> %{{.*}}, <2 x float> undef, <2 x i32> <i32 1, i32 0>

; Function Attrs: noinline nounwind optnone
define noundef <2 x float> @foo(<2 x float> noundef %a) #0 {
entry:
  %a.addr = alloca <2 x float>, align 8
  store <2 x float> %a, ptr %a.addr, align 8
  %0 = load <2 x float>, ptr %a.addr, align 8
  %1 = shufflevector <2 x float> %0, <2 x float> poison, <2 x i32> <i32 1, i32 0>
  ret <2 x float> %1
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="64" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1, !3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 6, !"dx.valver", !2}
!2 = !{i32 1, i32 7}
!3 = !{i32 7, !"frame-pointer", i32 2}
