; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[Int:.*]] = OpTypeInt 32 0
; CHECK-DAG: %[[Bool:.*]] = OpTypeBool
; CHECK-DAG: %[[NullInt:.*]] = OpConstantNull %[[Int]]

; CHECK: OpFunction
; CHECK: %[[A:.*]] = OpFunctionParameter %[[Int]]
; CHECK: %[[B:.*]] = OpFunctionParameter %[[Int]]
; CHECK: %[[Res:.*]] = OpIAdd %[[Int]] %[[A]] %[[B]]
; CHECK: %[[ResLtA:.*]] = OpSLessThan %[[Bool]] %[[Res]] %[[A]]
; CHECK: %[[BLt0:.*]] = OpSLessThan %[[Bool]] %[[B]] %[[NullInt]]
; CHECK: %[[Overflow:.*]] = OpLogicalNotEqual %[[Bool]] %[[BLt0]] %[[ResLtA]]
; CHECK: OpReturn
define spir_func void @test_sadd_overflow(ptr %out_result, ptr %out_overflow, i32 %a, i32 %b) {
entry:
  %res = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %res, 0
  %ofl = extractvalue { i32, i1 } %res, 1
  store i32 %val, ptr %out_result
  %zext_ofl = zext i1 %ofl to i8
  store i8 %zext_ofl, ptr %out_overflow
  ret void
}

; CHECK: OpFunction
; CHECK: %[[A2:.*]] = OpFunctionParameter %[[Int]]
; CHECK: %[[B2:.*]] = OpFunctionParameter %[[Int]]
; CHECK: %[[Res2:.*]] = OpISub %[[Int]] %[[A2]] %[[B2]]
; CHECK: %[[ResLtA2:.*]] = OpSLessThan %[[Bool]] %[[Res2]] %[[A2]]
; CHECK: %[[BGt0:.*]] = OpSGreaterThan %[[Bool]] %[[B2]] %[[NullInt]]
; CHECK: %[[Overflow2:.*]] = OpLogicalNotEqual %[[Bool]] %[[BGt0]] %[[ResLtA2]]
; CHECK: OpReturn
define spir_func void @test_ssub_overflow(ptr %out_result, ptr %out_overflow, i32 %a, i32 %b) {
entry:
  %res = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %res, 0
  %ofl = extractvalue { i32, i1 } %res, 1
  store i32 %val, ptr %out_result
  %zext_ofl = zext i1 %ofl to i8
  store i8 %zext_ofl, ptr %out_overflow
  ret void
}
