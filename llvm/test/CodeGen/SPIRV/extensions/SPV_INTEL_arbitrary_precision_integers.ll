; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s

define i6 @getConstantI6() {
  ret i6 2
}

define i13 @getConstantI13() {
  ret i13 42
}

;; Capabilities:
; CHECK-DAG: OpExtension "SPV_INTEL_arbitrary_precision_integers"
; CHECK-DAG: OpCapability ArbitraryPrecisionIntegersINTEL

; CHECK-NOT: DAG-FENCE

;; Names:
; CHECK-DAG: OpName %[[#GET_I6:]] "getConstantI6"
; CHECK-DAG: OpName %[[#GET_I13:]] "getConstantI13"

; CHECK-NOT: DAG-FENCE

;; Types and Constants:
; CHECK-DAG: %[[#I6:]] = OpTypeInt 6 0
; CHECK-DAG: %[[#I13:]] = OpTypeInt 13 0
; CHECK-DAG: %[[#CST_I6:]] = OpConstant %[[#I6]] 2
; CHECK-DAG: %[[#CST_I13:]] = OpConstant %[[#I13]] 42

; CHECK: %[[#GET_I6]] = OpFunction %[[#I6]]
; CHECK: OpReturnValue %[[#CST_I6]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_I13]] = OpFunction %[[#I13]]
; CHECK: OpReturnValue %[[#CST_I13]]
; CHECK: OpFunctionEnd
