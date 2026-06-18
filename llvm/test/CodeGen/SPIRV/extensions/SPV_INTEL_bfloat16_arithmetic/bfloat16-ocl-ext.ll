; Without SPV_INTEL_bfloat16_arithmetic: report error.
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: LLVM ERROR: Extended instructions with bfloat16 arguments require the following SPIR-V extension: SPV_INTEL_bfloat16_arithmetic

; With the extension: emit OpExtInst with OpExtension
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s --check-prefixes=COMMON,CHECK-FABS
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s --check-prefixes=COMMON,CHECK-FABSV
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s --check-prefix=COMMON

; COMMON-NOT: OpCapability BFloat16ArithmeticINTEL
; COMMON-DAG: OpCapability BFloat16TypeKHR
; COMMON-DAG: OpExtension "SPV_KHR_bfloat16"
; COMMON-DAG: OpExtension "SPV_INTEL_bfloat16_arithmetic"
; COMMON-DAG: [[EXTSET:%.*]] = OpExtInstImport "OpenCL.std"
; COMMON-DAG: [[BFLOAT:%.*]] = OpTypeFloat 16 0

; Scalar bfloat16 result and argument.
; CHECK-FABS: OpFunction [[BFLOAT]]
; CHECK-FABS: [[X:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK-FABS: OpExtInst [[BFLOAT]] [[EXTSET]] fabs [[X]]
define spir_func bfloat @test_fabs(bfloat %x) {
entry:
  %r = call bfloat @llvm.fabs.bf16(bfloat %x)
  ret bfloat %r
}

declare bfloat @llvm.fabs.bf16(bfloat)

; Vector bfloat16 result and argument.
; CHECK-FABSV-DAG: [[BFLOATV:%.*]] = OpTypeVector [[BFLOAT]] 4
; CHECK-FABSV: OpFunction [[BFLOATV]]
; CHECK-FABSV: [[XV:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK-FABSV: OpExtInst [[BFLOATV]] [[EXTSET]] fabs [[XV]]
define spir_func <4 x bfloat> @test_fabsv(<4 x bfloat> %x) {
entry:
  %r = call <4 x bfloat> @llvm.fabs.v4bf16(<4 x bfloat> %x)
  ret <4 x bfloat> %r
}

declare <4 x bfloat> @llvm.fabs.v4bf16(<4 x bfloat>)

; A non-bfloat16 (i32) result with a bfloat16 argument.
; CHECK-DAG: [[INT:%.*]] = OpTypeInt 32 0
; CHECK: OpFunction [[INT]]
; CHECK: [[XI:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK: OpExtInst [[INT]] [[EXTSET]] ilogb [[XI]]
define spir_func i32 @test_ilogb(bfloat %x) {
entry:
  %r = call i32 @_Z5ilogbDh(bfloat %x)
  ret i32 %r
}

declare i32 @_Z5ilogbDh(bfloat)