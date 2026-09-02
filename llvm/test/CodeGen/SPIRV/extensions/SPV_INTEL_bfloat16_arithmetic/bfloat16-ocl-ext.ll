; RUN: split-file %s %t

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %t/bf16_result_operand.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESULT-OPERAND-ERROR
; RESULT-OPERAND-ERROR:      error: <unknown>:0:0: in function test_fabs bfloat (bfloat): OpenCL Extended instructions with bfloat16 require the
; RESULT-OPERAND-ERROR-SAME:   following SPIR-V extension: SPV_INTEL_bfloat16_arithmetic

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %t/bf16_result.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESULT-ERROR
; RESULT-ERROR:      error: <unknown>:0:0: in function test_nan bfloat (i16): OpenCL Extended instructions with bfloat16 require the
; RESULT-ERROR-SAME:   following SPIR-V extension: SPV_INTEL_bfloat16_arithmetic

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %t/bf16_operand.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OPERAND-ERROR
; OPERAND-ERROR:      error: <unknown>:0:0: in function test_ilogb i32 (bfloat): OpenCL Extended instructions with bfloat16 require the
; OPERAND-ERROR-SAME:   following SPIR-V extension: SPV_INTEL_bfloat16_arithmetic

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %t/bf16_result_operand.ll -o - | FileCheck %s --check-prefixes=COMMON,BF16_RESULT_OPERAND
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %t/bf16_result.ll -o - | FileCheck %s --check-prefixes=COMMON,BF16_RESULT
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %t/bf16_operand.ll -o - | FileCheck %s --check-prefixes=COMMON,BF16_OPERAND

; TODO: re-enable spirv-val once it can verify SPV_INTEL_bfloat16_arithmetic with bfloat16 type on ExtInst
; RUNx: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %t/bf16_result_operand.ll -o - -filetype=obj | spirv-val %}
; RUNx: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %t/bf16_result.ll -o - -filetype=obj | spirv-val %}
; RUNx: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %t/bf16_operand.ll -o - -filetype=obj | spirv-val %}

; COMMON-DAG: OpCapability BFloat16ArithmeticINTEL
; COMMON-DAG: OpCapability BFloat16TypeKHR
; COMMON-DAG: OpExtension "SPV_KHR_bfloat16"
; COMMON-DAG: OpExtension "SPV_INTEL_bfloat16_arithmetic"
; COMMON-DAG: [[EXTSET:%.*]] = OpExtInstImport "OpenCL.std"
; COMMON-DAG: [[BFLOAT:%.*]] = OpTypeFloat 16 0
; COMMON-DAG: [[BFLOATV:%.*]] = OpTypeVector [[BFLOAT]] 4

; BF16_RESULT_OPERAND: [[OP:%.*]] = OpFunctionParameter [[BFLOAT]]
; BF16_RESULT_OPERAND: OpExtInst [[BFLOAT]] [[EXTSET]] fabs [[OP]]
; BF16_RESULT_OPERAND: [[OPV:%.*]] = OpFunctionParameter [[BFLOATV]]
; BF16_RESULT_OPERAND: OpExtInst [[BFLOATV]] [[EXTSET]] fabs [[OPV]]

; BF16_RESULT-DAG: [[INT:%.*]] = OpTypeInt 16 0
; BF16_RESULT-DAG: [[INTV:%.*]] = OpTypeVector [[INT]] 4
; BF16_RESULT: [[OP:%.*]] = OpFunctionParameter [[INT]]
; BF16_RESULT: OpExtInst [[BFLOAT]] [[EXTSET]] nan [[OP]]
; BF16_RESULT: [[OPV:%.*]] = OpFunctionParameter [[INTV]]
; BF16_RESULT: OpExtInst [[BFLOATV]] [[EXTSET]] nan [[OPV]]

; BF16_OPERAND-DAG: [[INT:%.*]] = OpTypeInt 32 0
; BF16_OPERAND-DAG: [[INTV:%.*]] = OpTypeVector [[INT]] 4
; BF16_OPERAND: [[OP:%.*]] = OpFunctionParameter [[BFLOAT]]
; BF16_OPERAND: OpExtInst [[INT]] [[EXTSET]] ilogb [[OP]]
; BF16_OPERAND: [[OPV:%.*]] = OpFunctionParameter [[BFLOATV]]
; BF16_OPERAND: OpExtInst [[INTV]] [[EXTSET]] ilogb [[OPV]]

;--- bf16_result_operand.ll
define spir_func bfloat @test_fabs(bfloat %x) {
  %r = call bfloat @llvm.fabs.bf16(bfloat %x)
  ret bfloat %r
}

define spir_func <4 x bfloat> @test_fabsv(<4 x bfloat> %x) {
  %r = call <4 x bfloat> @llvm.fabs.v4bf16(<4 x bfloat> %x)
  ret <4 x bfloat> %r
}

declare bfloat @llvm.fabs.bf16(bfloat)
declare <4 x bfloat> @llvm.fabs.v4bf16(<4 x bfloat>)

;--- bf16_result.ll
define spir_func bfloat @test_nan(i16 %x) {
  %r = call bfloat @_Z3nant(i16 %x)
  ret bfloat %r
}

define spir_func <4 x bfloat> @test_nanv(<4 x i16> %x) {
  %r = call <4 x bfloat> @_Z3nanDv4_t(<4 x i16> %x)
  ret <4 x bfloat> %r
}

declare bfloat @_Z3nant(i16)
declare <4 x bfloat> @_Z3nanDv4_t(<4 x i16>)

;--- bf16_operand.ll
define spir_func i32 @test_ilogb(bfloat %x) {
  %r = call i32 @_Z5ilogbDF16_(bfloat %x)
  ret i32 %r
}

define spir_func <4 x i32> @test_ilogbv(<4 x bfloat> %x) {
  %r = call <4 x i32> @_Z5ilogbDv4_DF16_(<4 x bfloat> %x)
  ret <4 x i32> %r
}

declare i32 @_Z5ilogbDF16_(bfloat)
declare <4 x i32> @_Z5ilogbDv4_DF16_(<4 x bfloat>)
