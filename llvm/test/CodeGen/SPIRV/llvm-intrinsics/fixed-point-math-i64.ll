; i64 smul.fix/umul.fix lowering widens to i128, which SPIR-V only has via
; SPV_ALTERA_arbitrary_precision_integers. Without the extension s64 is marked
; unsupported so the diagnostic comes from the legalizer.

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: unable to legalize instruction: {{.*}} = G_SMULFIX

; CHECK: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK: OpExtension "SPV_ALTERA_arbitrary_precision_integers"
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#I128:]] = OpTypeInt 128 0
; CHECK-DAG: %[[#Scale:]] = OpConstant %[[#I128]] 3

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#WideA:]] = OpSConvert %[[#I128]] %[[#A]]
; CHECK: %[[#WideB:]] = OpSConvert %[[#I128]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I128]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightArithmetic %[[#I128]] %[[#Mul]] %[[#Scale]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I64]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i64 @smulfix_i64(i64 %a, i64 %b) {
  %r = call i64 @llvm.smul.fix.i64(i64 %a, i64 %b, i32 3)
  ret i64 %r
}

; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#B:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#WideA:]] = OpUConvert %[[#I128]] %[[#A]]
; CHECK: %[[#WideB:]] = OpUConvert %[[#I128]] %[[#B]]
; CHECK: %[[#Mul:]] = OpIMul %[[#I128]] %[[#WideA]] %[[#WideB]]
; CHECK: %[[#Shift:]] = OpShiftRightLogical %[[#I128]] %[[#Mul]] %[[#Scale]]
; CHECK: %[[#Res:]] = OpUConvert %[[#I64]] %[[#Shift]]
; CHECK: OpReturnValue %[[#Res]]
define i64 @umulfix_i64(i64 %a, i64 %b) {
  %r = call i64 @llvm.umul.fix.i64(i64 %a, i64 %b, i32 3)
  ret i64 %r
}
