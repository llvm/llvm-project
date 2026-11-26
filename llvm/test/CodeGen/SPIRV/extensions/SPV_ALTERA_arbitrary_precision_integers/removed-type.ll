; Check that an integer type that legalization removes does not leave its
; extension and capability in the module.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_int4 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK-NOT: OpCapability Int4TypeINTEL
; CHECK-NOT: OpExtension "SPV_ALTERA_arbitrary_precision_integers"
; CHECK-NOT: OpExtension "SPV_INTEL_int4"
; CHECK-NOT: OpTypeInt
; CHECK: %[[#Int64:]] = OpTypeInt 64 0
; CHECK-NOT: OpTypeInt

; CHECK: OpFunction %[[#Int64]]
; CHECK: OpBitwiseAnd %[[#Int64]]
define spir_func i64 @trunc_zext_i27(i64 %x) {
  %t = trunc i64 %x to i27
  %z = zext i27 %t to i64
  ret i64 %z
}

; CHECK: OpFunction %[[#Int64]]
; CHECK: OpBitwiseAnd %[[#Int64]]
define spir_func i64 @trunc_zext_i4(i64 %x) {
  %t = trunc i64 %x to i4
  %z = zext i4 %t to i64
  ret i64 %z
}
