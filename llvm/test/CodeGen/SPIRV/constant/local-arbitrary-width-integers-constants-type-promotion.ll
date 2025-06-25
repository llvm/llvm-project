; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

define i4 @getConstantI4() {
  ret i4 2 ; i4 => OpTypeInt 8
}

define i11 @getConstantI11() {
  ret i11 7 ; i11 => OpTypeInt 16
}

define i24 @getConstantI24() {
  ret i24 42 ; i24 => OpTypeInt 32
}

define i63 @getConstantI63() {
  ret i63 5705 ; i63 => OpTypeInt 64
}

;; Capabilities:
; CHECK-DAG: OpCapability Int8
; CHECK-DAG: OpCapability Int16
; CHECK-DAG: OpCapability Int64

; CHECK-NOT: DAG-FENCE

;; Names:
; CHECK-DAG: OpName %[[#GET_I4:]] "getConstantI4"
; CHECK-DAG: OpName %[[#GET_I11:]] "getConstantI11"
; CHECK-DAG: OpName %[[#GET_I24:]] "getConstantI24"
; CHECK-DAG: OpName %[[#GET_I63:]] "getConstantI63"

; CHECK-NOT: DAG-FENCE

;; Types and Constants:
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#CST_I8:]] = OpConstant %[[#I8]] 2
; CHECK-DAG: %[[#CST_I16:]] = OpConstant %[[#I16]] 7
; CHECK-DAG: %[[#CST_I32:]] = OpConstant %[[#I32]] 42
; CHECK-DAG: %[[#CST_I64:]] = OpConstant %[[#I64]] 5705

; CHECK: %[[#GET_I4]] = OpFunction %[[#I8]]
; CHECK: OpReturnValue %[[#CST_I8]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_I11]] = OpFunction %[[#I16]]
; CHECK: OpReturnValue %[[#CST_I16]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_I24]] = OpFunction %[[#I32]]
; CHECK: OpReturnValue %[[#CST_I32]]
; CHECK: OpFunctionEnd

; CHECK: %[[#GET_I63]] = OpFunction %[[#I64]]
; CHECK: OpReturnValue %[[#CST_I64]]
; CHECK: OpFunctionEnd
