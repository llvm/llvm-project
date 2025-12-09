; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpTypeInt type with a width other than 8, 16, 32 or 64 bits requires the following SPIR-V extension: SPV_ALTERA_arbitrary_precision_integers

; CHECK: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK: OpExtension "SPV_ALTERA_arbitrary_precision_integers"
; CHECK: OpName %[[#Foo:]] "foo"
; CHECK: %[[#Int128Ty:]] = OpTypeInt 128 0

; CHECK: %[[#Foo]] = OpFunction
define i64 @foo(i64 %x, i64 %y, i32 %amt) {
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpSConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpBitwiseOr %[[#Int128Ty]]
; CHECK: {{.*}} = OpUConvert %[[#Int128Ty]]
; CHECK: {{.*}} = OpShiftRightLogical %[[#Int128Ty]]
  %tmp0 = zext i64 %x to i128
  %tmp1 = sext i64 %y to i128
  %tmp2 = or i128 %tmp0, %tmp1
  %tmp7 = zext i32 13 to i128
  %tmp3 = lshr i128 %tmp2, %tmp7
  %tmp4 = trunc i128 %tmp3 to i64
  ret i64 %tmp4
}
