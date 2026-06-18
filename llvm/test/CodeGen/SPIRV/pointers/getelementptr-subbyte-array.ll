; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#int8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#int32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#size:]] = OpConstant %[[#int32]] 8
; CHECK-DAG: %[[#bytearr:]] = OpTypeArray %[[#int8]] %[[#size]]
; CHECK-DAG: %[[#boolarr:]] = OpTypeArray %[[#bool]] %[[#size]]
; CHECK-DAG: %[[#boolptr:]] = OpTypePointer Function %[[#boolarr]]
; CHECK-DAG: %[[#i8ptr:]] = OpTypePointer Function %[[#int8]]
; CHECK-DAG: %[[#one:]] = OpConstant %[[#]] 1

@bytes = internal global [8 x i8] zeroinitializer
@bools = internal global [8 x i1] zeroinitializer

; Byte addressing into a byte array lowers to a valid logical access chain.
; CHECK: OpFunction
; CHECK: %[[#var:]] = OpVariable %[[#]] Function
; CHECK: %[[#]] = OpInBoundsAccessChain %[[#i8ptr]] %[[#var]] %[[#one]]
define void @byte_array() {
entry:
  %gep = getelementptr inbounds i8, ptr @bytes, i32 1
  store i8 0, ptr %gep
  ret void
}

; Byte addressing into a subbyte array uses the element alloc size as the
; stride. This previously crashed with a division by zero because the element
; type size in bits (1 for i1) is not a whole number of bytes.
; CHECK: OpFunction
; CHECK: %[[#bvar:]] = OpVariable %[[#boolptr]] Function
; CHECK: %[[#]] = OpInBoundsAccessChain %[[#]] %[[#bvar]] %[[#one]]
define void @subbyte_array() {
entry:
  %gep = getelementptr inbounds i8, ptr @bools, i32 1
  store i1 true, ptr %gep
  ret void
}
