; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpTypeFloat type with bfloat requires the following SPIR-V extension: SPV_KHR_bfloat16

; CHECK-DAG: %[[#glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: OpCapability BFloat16TypeKHR
; CHECK-DAG: OpExtension "SPV_KHR_bfloat16"
; CHECK: %[[#BFLOAT:]] = OpTypeFloat 16 0
; CHECK: %[[#VEC:]] = OpTypeVector %[[#BFLOAT]] 2

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@G1 = global bfloat 0.0
@G2 = global <2 x bfloat> zeroinitializer

define spir_kernel void @test() {
entry:
  %addr1 = alloca bfloat
  %addr2 = alloca <2 x bfloat>
  %data1 = load bfloat, ptr %addr1
  %data2 = load <2 x bfloat>, ptr %addr2
  store bfloat %data1, ptr @G1
  store <2 x bfloat> %data2, ptr @G2
  ret void
}

; CHECK-DAG: %[[#zero:]] = OpConstantNull %[[#BFLOAT]]
; CHECK-DAG: %[[#one:]] = OpConstant %[[#BFLOAT]] 16256
; CHECK-DAG: %[[#vec_zero:]] = OpConstantNull %[[#VEC]]
; CHECK-DAG: %[[#vec_one:]] = OpConstantComposite %[[#VEC]] %[[#one]] %[[#one]]

define spir_func bfloat @saturate_bfloat(bfloat %a) {
entry:
  ; CHECK: %[[#]] = OpExtInst %[[#BFLOAT]] %[[#glsl]] FClamp %[[#]] %[[#zero]] %[[#one]]
  %r = call bfloat @llvm.spv.saturate.bf16(bfloat %a)
  ret bfloat %r
}

define spir_func <2 x bfloat> @saturate_bfloat2(<2 x bfloat> %a) {
entry:
  ; CHECK: %[[#]] = OpExtInst %[[#VEC]] %[[#glsl]] FClamp %[[#]] %[[#vec_zero]] %[[#vec_one]]
  %r = call <2 x bfloat> @llvm.spv.saturate.v2bf16(<2 x bfloat> %a)
  ret <2 x bfloat> %r
}

declare bfloat @llvm.spv.saturate.bf16(bfloat)
declare <2 x bfloat> @llvm.spv.saturate.v2bf16(<2 x bfloat>)
