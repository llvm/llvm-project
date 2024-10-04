; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure lowering is correctly generating spirv code.

define spir_func noundef i32 @test_scalar(double noundef %D) local_unnamed_addr {
entry:
  ; CHECK: %[[#]] = OpBitcast %[[#]] %[[#]]
  %0 = bitcast double %D to <2 x i32>
  ; CHECK: %[[#]] = OpCompositeExtract %[[#]] %[[#]] 0
  %1 = extractelement <2 x i32> %0, i64 0
  ; CHECK: %[[#]] = OpCompositeExtract %[[#]] %[[#]] 1
  %2 = extractelement <2 x i32> %0, i64 1
  %add = add i32 %1, %2
  ret i32 %add
}
