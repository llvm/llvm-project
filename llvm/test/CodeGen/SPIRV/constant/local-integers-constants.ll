; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

define i16 @getConstantI16() {
  ret i16 -58
}

define i32 @getConstantI32() {
  ret i32 42
}

define i64 @getConstantI64() {
  ret i64 123456789
}

define i64 @getLargeConstantI64() {
  ret i64 34359738368
}

;; Capabilities:
; CHECK-DAG: OpCapability Int16
; CHECK-DAG: OpCapability Int64

; CHECK-NOT: DAG-FENCE

;; Names:
; CHECK-DAG: OpName %[[#GET_I16:]] "getConstantI16"
; CHECK-DAG: OpName %[[#GET_I32:]] "getConstantI32"
; CHECK-DAG: OpName %[[#GET_I64:]] "getConstantI64"
; CHECK-DAG: OpName %[[#GET_LARGE_I64:]] "getLargeConstantI64"

; CHECK-NOT: DAG-FENCE

;; Types and Constants:
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#CST_I16:]] = OpConstant %[[#I16]] 65478
; CHECK-DAG: %[[#CST_I32:]] = OpConstant %[[#I32]] 42
; CHECK-DAG: %[[#CST_I64:]] = OpConstant %[[#I64]] 123456789 0
; CHECK-DAG: %[[#CST_LARGE_I64:]] = OpConstant %[[#I64]] 0 8

; CHECK: %[[#GET_I16]] = OpFunction %[[#I16]]
; CHECK: OpReturnValue %[[#CST_I16]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_I32]] = OpFunction %[[#I32]]
; CHECK: OpReturnValue %[[#CST_I32]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_I64]] = OpFunction %[[#I64]]
; CHECK: OpReturnValue %[[#CST_I64]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_LARGE_I64]] = OpFunction %[[#I64]]
; CHECK: OpReturnValue %[[#CST_LARGE_I64]]
; CHECK: OpFunctionEnd
