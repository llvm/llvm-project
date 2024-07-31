; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; The set of valid inputs for get_global_id depends on the runtime NDRange,
;; but inputs outside of [0, 2] always return 0.
;; Here we assume Itanium mangling for function name.
declare i64 @_Z13get_global_idj(i32)

define i64 @foo(i32 %dim) {
  %x = call i64 @_Z13get_global_idj(i32 0)
  %zero = call i64 @_Z13get_global_idj(i32 5)
  %unknown = call i64 @_Z13get_global_idj(i32 %dim)
  %acc = add i64 %x, %zero
  %ret = add i64 %acc, %unknown
  ret i64 %ret
}

;; Capabilities:
; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpCapability Int64

; CHECK-NOT: DAG-FENCE

;; Decorations:
; CHECK-DAG: OpDecorate %[[#GET_GLOBAL_ID:]] BuiltIn GlobalInvocationId
; CHECK-DAG: OpDecorate %[[#GET_GLOBAL_ID]] Constant

; CHECK-NOT: DAG-FENCE

;; Types, Constants and Variables:
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#VEC:]] = OpTypeVector %[[#I64]] 3
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Input %[[#VEC]]
; CHECK-DAG: %[[#FN:]] = OpTypeFunction %[[#I64]] %[[#I32]]
; CHECK-DAG: %[[#GET_GLOBAL_ID]] = OpVariable %[[#PTR]] Input
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#I64]]
; CHECK-DAG: %[[#THREE:]] = OpConstant %[[#I32]] 3

;; Functions:
; CHECK:     OpFunction %[[#I64]] None %[[#FN]]
; CHECK:     %[[#DIM:]] = OpFunctionParameter %[[#I32]]

;; get_global_id(0): OpLoad + OpCompositeExtract.
; CHECK:     %[[#TMP1:]] = OpLoad %[[#VEC]] %[[#GET_GLOBAL_ID]]
; CHECK:     %[[#X:]] = OpCompositeExtract %[[#I64]] %[[#TMP1]] 0

;; get_global_id(5): OpConstant (above) of zero.
;; get_global_id(dim): Here we assume a specific implementation using select.
; CHECK-DAG: %[[#TMP2:]] = OpLoad %[[#VEC]] %[[#GET_GLOBAL_ID]]
; CHECK-DAG: %[[#TMP3:]] = OpVectorExtractDynamic %[[#I64]] %[[#TMP2]] %[[#DIM]]
; CHECK-DAG: %[[#COND:]] = OpULessThan %[[#BOOL]] %[[#DIM]] %[[#THREE]]
; CHECK:     %[[#UNKNOWN:]] = OpSelect %[[#I64]] %[[#COND]] %[[#TMP3]] %[[#ZERO]]
