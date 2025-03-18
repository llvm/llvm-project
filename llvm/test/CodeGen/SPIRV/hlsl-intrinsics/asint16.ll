; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Int16
; CHECK: OpCapability Float16
; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_int_16:]] = OpTypeVector %[[#int_16]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4


define i16 @test_int16(i16 returned %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#int_16]]
  ; CHECK: OpReturnValue %[[#arg0]]
  ret i16 %p0
}

define i16 @test_half(half nofpclass(nan inf) %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_16:]]
  ; CHECK: %[[#]] = OpBitcast %[[#int_16]] %[[#arg0]]
  %0 = bitcast half %p0 to i16
  ;CHECK: OpReturnValue %[[#]]
  ret i16 %0

}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <4 x i16> @test_vector_int4(<4 x i16> returned %p0) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_int_16]]
  ; CHECK: OpReturnValue %[[#arg0]]
  ret <4 x i16> %p0
}

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <4 x i16> @test_vector_half4(<4 x half> nofpclass(nan inf) %p1) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpBitcast %[[#vec4_int_16]] %[[#arg0]]
  %0 = bitcast <4 x half> %p1 to <4 x i16>
  ;CHECK: OpReturnValue %[[#]]
  ret <4 x i16> %0
}

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 8}
!2 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 5929de8c7731748bf58ad9b1fedfed75e7aae455)"}
