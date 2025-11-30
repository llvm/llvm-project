; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_arithmetic,+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: Arithmetic instructions with bfloat16 arguments require the following SPIR-V extension: SPV_INTEL_bfloat16_arithmetic

; CHECK-DAG: OpCapability BFloat16TypeKHR
; CHECK-DAG: OpCapability BFloat16ArithmeticINTEL
; CHECK-DAG: OpExtension "SPV_KHR_bfloat16"
; CHECK-DAG: OpExtension "SPV_INTEL_bfloat16_arithmetic"
; CHECK-DAG: OpName [[NEG:%.*]] "neg"
; CHECK-DAG: OpName [[NEGV:%.*]] "negv"
; CHECK-DAG: OpName [[ADD:%.*]] "add"
; CHECK-DAG: OpName [[ADDV:%.*]] "addv"
; CHECK-DAG: OpName [[SUB:%.*]] "sub"
; CHECK-DAG: OpName [[SUBV:%.*]] "subv"
; CHECK-DAG: OpName [[MUL:%.*]] "mul"
; CHECK-DAG: OpName [[MULV:%.*]] "mulv"
; CHECK-DAG: OpName [[DIV:%.*]] "div"
; CHECK-DAG: OpName [[DIVV:%.*]] "divv"
; CHECK-DAG: OpName [[REM:%.*]] "rem"
; CHECK-DAG: OpName [[REMV:%.*]] "remv"
; CHECK: [[BFLOAT:%.*]] = OpTypeFloat 16 0
; CHECK: [[BFLOATV:%.*]] = OpTypeVector [[BFLOAT]] 4

; CHECK-DAG: [[NEG]] = OpFunction [[BFLOAT]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK-DAG: [[R:%.*]] = OpFNegate [[BFLOAT]] [[X]]
define spir_func bfloat @neg(bfloat %x) {
entry:
  %r = fneg bfloat %x
  ret bfloat %r
}

; CHECK-DAG: [[NEGV]] = OpFunction [[BFLOATV]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK-DAG: [[R:%.*]] = OpFNegate [[BFLOATV]] [[X]]
define spir_func <4 x bfloat> @negv(<4 x bfloat> %x) {
entry:
  %r = fneg <4 x bfloat> %x
  ret <4 x bfloat> %r
}

; CHECK-DAG: [[ADD]] = OpFunction [[BFLOAT]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK-DAG: [[R:%.*]] = OpFAdd [[BFLOAT]] [[X]] [[Y]]
define spir_func bfloat @add(bfloat %x, bfloat %y) {
entry:
  %r = fadd bfloat %x, %y
  ret bfloat %r
}

; CHECK-DAG: [[ADDV]] = OpFunction [[BFLOATV]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK-DAG: [[R:%.*]] = OpFAdd [[BFLOATV]] [[X]] [[Y]]
define spir_func <4 x bfloat> @addv(<4 x bfloat> %x, <4 x bfloat> %y) {
entry:
  %r = fadd <4 x bfloat> %x, %y
  ret <4 x bfloat> %r
}

; CHECK-DAG: [[SUB]] = OpFunction [[BFLOAT]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK-DAG: [[R:%.*]] = OpFSub [[BFLOAT]] [[X]] [[Y]]
define spir_func bfloat @sub(bfloat %x, bfloat %y) {
entry:
  %r = fsub bfloat %x, %y
  ret bfloat %r
}

; CHECK-DAG: [[SUBV]] = OpFunction [[BFLOATV]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK-DAG: [[R:%.*]] = OpFSub [[BFLOATV]] [[X]] [[Y]]
define spir_func <4 x bfloat> @subv(<4 x bfloat> %x, <4 x bfloat> %y) {
entry:
  %r = fsub <4 x bfloat> %x, %y
  ret <4 x bfloat> %r
}

; CHECK-DAG: [[MUL]] = OpFunction [[BFLOAT]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK-DAG: [[R:%.*]] = OpFMul [[BFLOAT]] [[X]] [[Y]]
define spir_func bfloat @mul(bfloat %x, bfloat %y) {
entry:
  %r = fmul bfloat %x, %y
  ret bfloat %r
}

; CHECK-DAG: [[MULV]] = OpFunction [[BFLOATV]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK-DAG: [[R:%.*]] = OpFMul [[BFLOATV]] [[X]] [[Y]]
define spir_func <4 x bfloat> @mulv(<4 x bfloat> %x, <4 x bfloat> %y) {
entry:
  %r = fmul <4 x bfloat> %x, %y
  ret <4 x bfloat> %r
}

; CHECK-DAG: [[DIV]] = OpFunction [[BFLOAT]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK-DAG: [[R:%.*]] = OpFDiv [[BFLOAT]] [[X]] [[Y]]
define spir_func bfloat @div(bfloat %x, bfloat %y) {
entry:
  %r = fdiv bfloat %x, %y
  ret bfloat %r
}

; CHECK-DAG: [[DIVV]] = OpFunction [[BFLOATV]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK-DAG: [[R:%.*]] = OpFDiv [[BFLOATV]] [[X]] [[Y]]
define spir_func <4 x bfloat> @divv(<4 x bfloat> %x, <4 x bfloat> %y) {
entry:
  %r = fdiv <4 x bfloat> %x, %y
  ret <4 x bfloat> %r
}

; CHECK-DAG: [[REM]] = OpFunction [[BFLOAT]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOAT]]
; CHECK-DAG: [[R:%.*]] = OpFRem [[BFLOAT]] [[X]] [[Y]]
define spir_func bfloat @rem(bfloat %x, bfloat %y) {
entry:
  %r = frem bfloat %x, %y
  ret bfloat %r
}

; CHECK-DAG: [[REMV]] = OpFunction [[BFLOATV]]
; CHECK: [[X:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK: [[Y:%.*]] = OpFunctionParameter [[BFLOATV]]
; CHECK-DAG: [[R:%.*]] = OpFRem [[BFLOATV]] [[X]] [[Y]]
define spir_func <4 x bfloat> @remv(<4 x bfloat> %x, <4 x bfloat> %y) {
entry:
  %r = frem <4 x bfloat> %x, %y
  ret <4 x bfloat> %r
}
