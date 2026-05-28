; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; Without the extension, i128 cannot be lowered: the diagnostic must come from
; OpTypeInt emission, not a legalizer "unable to legalize G_ICMP" failure.

; CHECK-ERROR: LLVM ERROR: OpTypeInt type with a width other than 8, 16, 32 or 64 bits requires the following SPIR-V extension: SPV_ALTERA_arbitrary_precision_integers

; CHECK: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK: OpExtension "SPV_ALTERA_arbitrary_precision_integers"
; CHECK-DAG: %[[#BoolTy:]] = OpTypeBool
; CHECK-DAG: %[[#Int128Ty:]] = OpTypeInt 128 0

; CHECK: OpFunction
; CHECK: [[#]] = OpSGreaterThan %[[#BoolTy]]
; CHECK: [[#]] = OpIEqual %[[#BoolTy]]
; CHECK: [[#]] = OpULessThan %[[#BoolTy]]
define spir_func void @test_icmp_i128(i128 %a, i128 %b, ptr %out) {
entry:
  %sgt = icmp sgt i128 %a, %b
  %eq = icmp eq i128 %a, %b
  %ult = icmp ult i128 %a, %b
  %t1 = zext i1 %sgt to i32
  %t2 = zext i1 %eq to i32
  %t3 = zext i1 %ult to i32
  store i32 %t1, ptr %out
  store i32 %t2, ptr %out
  store i32 %t3, ptr %out
  ret void
}
