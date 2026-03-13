; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

;; The set of valid inputs for get_num_groups depends on the runtime NDRange,
;; but inputs outside of [0, 2] always return 1.
;; Here we assume Itanium mangling for function name.
declare i64 @_Z14get_num_groupsj(i32)

define i64 @foo(i32 %dim) {
  %x = call i64 @_Z14get_num_groupsj(i32 0)
  %y = call i64 @_Z14get_num_groupsj(i32 5)
  %acc = add i64 %x, %y
  %unknown = call i64 @_Z14get_num_groupsj(i32 %dim)
  %ret = add i64 %acc, %unknown
  ret i64 %ret
}

;; Capabilities:
; CHECK-DAG: OpCapability Kernel
; CHECK-DAG: OpCapability Int64

; CHECK-NOT: DAG-FENCE

;; Decorations:
; CHECK-DAG: OpDecorate %[[#GET_NUM_GROUPS:]] BuiltIn NumWorkgroups
; CHECK-DAG: OpDecorate %[[#GET_NUM_GROUPS]] Constant

; CHECK-NOT: DAG-FENCE

;; Types, Constants and Variables:
; CHECK-DAG: %[[#BOOL:]] = OpTypeBool
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#VEC:]] = OpTypeVector %[[#I64]] 3
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Input %[[#VEC]]
; CHECK-DAG: %[[#FN:]] = OpTypeFunction %[[#I64]] %[[#I32]]
; CHECK-DAG: %[[#GET_NUM_GROUPS]] = OpVariable %[[#PTR]] Input
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I64]] 1
; CHECK-DAG: %[[#THREE:]] = OpConstant %[[#I32]] 3

;; Functions:
; CHECK:     OpFunction %[[#I64]] None %[[#FN]]
; CHECK:     %[[#DIM:]] = OpFunctionParameter %[[#I32]]

;; get_num_groups(0): OpLoad + OpCompositeExtract.
; CHECK:     %[[#TMP1:]] = OpLoad %[[#VEC]] %[[#GET_NUM_GROUPS]]
; CHECK:     %[[#X:]] = OpCompositeExtract %[[#I64]] %[[#TMP1]] 0

;; get_num_groups(5): OpConstant of one.
; CHECK:     OpIAdd %[[#I64]] %[[#X]] %[[#ONE]]

;; get_num_groups(dim): Implementation using OpSelect.
; CHECK-DAG: %[[#TMP2:]] = OpLoad %[[#VEC]] %[[#GET_NUM_GROUPS]]
; CHECK-DAG: %[[#TMP3:]] = OpVectorExtractDynamic %[[#I64]] %[[#TMP2]] %[[#DIM]]
; CHECK-DAG: %[[#COND:]] = OpULessThan %[[#BOOL]] %[[#DIM]] %[[#THREE]]
; CHECK:     %[[#UNKNOWN:]] = OpSelect %[[#I64]] %[[#COND]] %[[#TMP3]] %[[#ONE]]
