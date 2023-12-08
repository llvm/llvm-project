; RUN: llc --filetype=obj %s -o - 2>&1 | dxil-dis -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"
; Make sure global ctor type is changed to void ()*.
; CHECK:@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_static_global.hlsl, i8* null }]

@f = internal unnamed_addr global float 0.000000e+00, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_static_global.hlsl, ptr null }]

declare float @"?init@@YAMXZ"() local_unnamed_addr #0

; Function Attrs: nounwind
define float @"?foo@@YAMXZ"() local_unnamed_addr #1 {
entry:
  %0 = load float, ptr @f, align 4, !tbaa !4
  %inc = fadd float %0, 1.000000e+00
  store float %inc, ptr @f, align 4, !tbaa !4
  ret float %0
}

; Function Attrs: nounwind
define float @"?bar@@YAMXZ"() local_unnamed_addr #1 {
entry:
  %0 = load float, ptr @f, align 4, !tbaa !4
  %dec = fadd float %0, -1.000000e+00
  store float %dec, ptr @f, align 4, !tbaa !4
  ret float %0
}

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_static_global.hlsl() #1 {
entry:
  %call.i = tail call float @"?init@@YAMXZ"() #2
  store float %call.i, ptr @f, align 4, !tbaa !4
  ret void
}

attributes #0 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!dx.valver = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git c5dfff0e58cc66d74e666c31368f6d44328dd2f7)"}
!3 = !{i32 1, i32 7}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
