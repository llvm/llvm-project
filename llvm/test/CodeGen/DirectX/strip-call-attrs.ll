
; RUN: opt -S -dxil-prepare < %s | FileCheck %s
target triple = "dxil-unknown-shadermodel6.0-library"

@f = internal unnamed_addr global float 0.000000e+00, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_static_global.hlsl, ptr null }]

; Make sure noundef is removed for function.
; CHECK:declare float @"?init@@YAMXZ"()
declare noundef float @"?init@@YAMXZ"() local_unnamed_addr #0

; Make sure noundef is removed for call.
; CHECK: %call.i = tail call float @"?init@@YAMXZ"()
; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_static_global.hlsl() #1 {
entry:
  %call.i = tail call noundef float @"?init@@YAMXZ"() #2
  store float %call.i, ptr @f, align 4
  ret void
}

attributes #0 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

