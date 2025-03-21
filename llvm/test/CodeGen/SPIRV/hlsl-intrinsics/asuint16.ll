; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#half:]] = OpTypeFloat 16
; CHECK-DAG: %[[#uint_16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#v4uint_16:]] = OpTypeVector %[[#uint_16]] 4
; CHECK-DAG: %[[#v4half:]] = OpTypeVector %[[#half]] 4

define i16 @test_half(half nofpclass(nan inf) %p0) {
entry:
  %0 = bitcast half %p0 to i16
  ret i16 %0

 ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#half]]
 ; CHECK: %[[#bit_cast:]] = OpBitcast %[[#uint_16]] %[[#arg0]]
 ; CHECK: OpReturnValue %[[#bit_cast]]
}

define noundef <4 x i16> @test_vector_half(<4 x half> nofpclass(nan inf) %p1) {
entry:
  %0 = bitcast <4 x half> %p1 to <4 x i16>
  ret <4 x i16> %0

 ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#v4half]]
 ; CHECK: %[[#bit_cast:]] = OpBitcast %[[#v4uint_16]] %[[#arg0]]
 ; CHECK: OpReturnValue %[[#bit_cast]]
}

attributes #0 = { alwaysinline mustprogress nofree norecurse nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0}
!dx.valver = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 8}
!2 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 8037234c865d98219b54a70d9d63aee1d1f2be1c)"}

