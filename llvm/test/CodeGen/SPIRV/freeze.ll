; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[FloatTy:.*]] = OpTypeFloat 32
; CHECK-DAG: %[[ShortTy:.*]] = OpTypeInt 16 0
; CHECK-DAG: %[[IntTy:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[Undef16:.*]] = OpUndef %[[ShortTy]]
; CHECK-DAG: %[[Undef32:.*]] = OpUndef %[[IntTy]]
; CHECK-DAG: %[[UndefFloat:.*]] = OpUndef %[[FloatTy]]
; CHECK-DAG: %[[Const100:.*]] = OpConstant %[[IntTy]] 100

define spir_func i16 @test_nil0(i16 %arg2) {
entry:
; CHECK: %[[Arg2:.*]] = OpFunctionParameter
; CHECK: %[[NotAStaticPoison:.*]] = OpIAdd %[[ShortTy]] %[[Arg2]] %[[Undef16]]
; CHECK: OpReturnValue %[[NotAStaticPoison]]
  %poison1 = add i16 %arg2, undef
  %nil0 = freeze i16 %poison1
  ret i16 %nil0
}

define spir_func i32 @test_nil1() {
entry:
  %nil1 = freeze i32 undef
  ret i32 %nil1
}

define spir_func float @test_nil2() {
entry:
  %nil2 = freeze float poison
  ret float %nil2
}

define spir_func float @freeze_float(float %arg1) {
entry:
; CHECK: %[[Arg1:.*]] = OpFunctionParameter %[[FloatTy]]
  %val1 = freeze float %arg1
  ret float %val1
}

define spir_func i32 @foo() {
entry:
  %val2 = freeze i32 100
  %val3 = freeze i32 %val2
  ret i32 %val3
}
