; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[FOOBAR:%.+]] "foobar"
; CHECK-DAG: OpName [[PRODUCER:%.+]] "producer"
; CHECK-DAG: OpName [[CONSUMER:%.+]] "consumer"

; CHECK-NOT: DAG-FENCE

%ty1 = type {i16, i32}
%ty2 = type {%ty1, i64}

; CHECK-DAG: [[I16:%.+]] = OpTypeInt 16
; CHECK-DAG: [[I32:%.+]] = OpTypeInt 32
; CHECK-DAG: [[I64:%.+]] = OpTypeInt 64
; CHECK-DAG: [[TY1:%.+]] = OpTypeStruct [[I16]] [[I32]]
; CHECK-DAG: [[TY2:%.+]] = OpTypeStruct [[TY1]] [[I64]]
; CHECK-DAG: [[UNDEF_I16:%.+]] = OpUndef [[I16]]
; CHECK-DAG: [[UNDEF_I64:%.+]] = OpUndef [[I64]]
; CHECK-DAG: [[UNDEF_TY2:%.+]] = OpUndef [[TY2]]
; CHECK-DAG: [[CST_42:%.+]] = OpConstant [[I32]] 42

; CHECK-NOT: DAG-FENCE

define i32 @foobar() {
  %agg = call %ty2 @producer(i16 undef, i32 42, i64 undef)
  %ret = call i32 @consumer(%ty2 %agg)
  ret i32 %ret
}

; CHECK: [[FOOBAR]] = OpFunction
; CHECK: [[AGG:%.+]] = OpFunctionCall [[TY2]] [[PRODUCER]] [[UNDEF_I16]] [[CST_42]] [[UNDEF_I64]]
; CHECK: [[RET:%.+]] = OpFunctionCall [[I32]] [[CONSUMER]] [[AGG]]
; CHECK: OpReturnValue [[RET]]
; CHECK: OpFunctionEnd


define %ty2 @producer(i16 %a, i32 %b, i64 %c) {
  %agg1 = insertvalue %ty2 undef, i16 %a, 0, 0
  %agg2 = insertvalue %ty2 %agg1, i32 %b, 0, 1
  %agg3 = insertvalue %ty2 %agg2, i64 %c, 1
  ret %ty2 %agg3
}

; CHECK: [[PRODUCER]] = OpFunction
; CHECK: [[A:%.+]] = OpFunctionParameter [[I16]]
; CHECK: [[B:%.+]] = OpFunctionParameter [[I32]]
; CHECK: [[C:%.+]] = OpFunctionParameter [[I64]]
; CHECK: [[AGG1:%.+]] = OpCompositeInsert [[TY2]] [[A]] [[UNDEF_TY2]] 0 0
; CHECK: [[AGG2:%.+]] = OpCompositeInsert [[TY2]] [[B]] [[AGG1]] 0 1
; CHECK: [[AGG3:%.+]] = OpCompositeInsert [[TY2]] [[C]] [[AGG2]] 1
; CHECK: OpReturnValue [[AGG3]]
; CHECK: OpFunctionEnd


define i32 @consumer(%ty2 %agg) {
  %ret = extractvalue %ty2 %agg, 0, 1
  ret i32 %ret
}

; CHECK: [[CONSUMER]] = OpFunction
; CHECK: [[AGG:%.+]] = OpFunctionParameter [[TY2]]
; CHECK: [[RET:%.+]] = OpCompositeExtract [[I32]] [[AGG]] 0 1
; CHECK: OpReturnValue [[RET]]
; CHECK: OpFunctionEnd
