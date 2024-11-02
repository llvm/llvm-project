; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

define half @getConstantFP16() {
  ret half 0x3ff1340000000000 ; 0x3c4d represented as double.
}

define float @getConstantFP32() {
  ret float 0x3fd27c8be0000000 ; 0x3e93e45f represented as double
}

define double @getConstantFP64() {
  ret double 0x4f2de42b8c68f3f1
}

;; Capabilities
; CHECK-DAG: OpCapability Float16
; CHECK-DAG: OpCapability Float64

; CHECK-NOT: DAG-FENCE

;; Names:
; CHECK-DAG: OpName %[[#GET_FP16:]] "getConstantFP16"
; CHECK-DAG: OpName %[[#GET_FP32:]] "getConstantFP32"
; CHECK-DAG: OpName %[[#GET_FP64:]] "getConstantFP64"

; CHECK-NOT: DAG-FENCE

;; Types and Constants:
;; NOTE: These tests don't actually check the values of the constants because
;;       their representation isn't defined for textual output.
;; TODO: Test constant representation using binary output.
; CHECK-DAG: %[[#FP16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#FP32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#CST_FP16:]] = OpConstant %[[#FP16]]
; CHECK-DAG: %[[#CST_FP32:]] = OpConstant %[[#FP32]]
; CHECK-DAG: %[[#CST_FP64:]] = OpConstant %[[#FP64]]

; CHECK: %[[#GET_FP16]] = OpFunction %[[#FP16]]
; CHECK: OpReturnValue %[[#CST_FP16]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_FP32]] = OpFunction %[[#FP32]]
; CHECK: OpReturnValue %[[#CST_FP32]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_FP64]] = OpFunction %[[#FP64]]
; CHECK: OpReturnValue %[[#CST_FP64]]
; CHECK: OpFunctionEnd
