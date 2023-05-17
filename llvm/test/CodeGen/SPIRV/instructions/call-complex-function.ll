; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: OpName [[FUN:%.+]] "fun"
; CHECK-DAG: OpName [[FOO:%.+]] "foo"
; CHECK-DAG: OpName [[GOO:%.+]] "goo"

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[I16:%.+]] = OpTypeInt 16
; CHECK-DAG: [[I32:%.+]] = OpTypeInt 32
; CHECK-DAG: [[I64:%.+]] = OpTypeInt 64
; CHECK-DAG: [[FN3:%.+]] = OpTypeFunction [[I32]] [[I32]] [[I16]] [[I64]]
; CHECK-DAG: [[PAIR:%.+]] = OpTypeStruct [[I32]] [[I16]]
; CHECK-DAG: [[FN1:%.+]] = OpTypeFunction [[I32]] [[I32]]
; CHECK-DAG: [[FN2:%.+]] = OpTypeFunction [[I32]] [[PAIR]] [[I64]]
;; According to the Specification, the OpUndef can be defined in Function.
;; But the Specification also recommends defining it here. So we enforce that.
; CHECK-DAG: [[UNDEF:%.+]] = OpUndef [[PAIR]]


declare i32 @fun(i32 %value)

;; Check for @fun declaration
; CHECK:      [[FUN]] = OpFunction [[I32]] None [[FN1]]
; CHECK-NEXT: OpFunctionParameter [[I32]]
; CHECK-NEXT: OpFunctionEnd


define i32 @foo({i32, i16} %in, i64 %unused) {
  %first = extractvalue {i32, i16} %in, 0
  %bar = call i32 @fun(i32 %first)
  ret i32 %bar
}

; CHECK:      [[GOO]] = OpFunction [[I32]] None [[FN3]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[I32]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[I16]]
; CHECK-NEXT: [[C:%.+]] = OpFunctionParameter [[I64]]
; CHECK:      [[AGG1:%.+]] = OpCompositeInsert [[PAIR]] [[A]] [[UNDEF]] 0
; CHECK:      [[AGG2:%.+]] = OpCompositeInsert [[PAIR]] [[B]] [[AGG1]] 1
; CHECK:      [[RET:%.+]] = OpFunctionCall [[I32]] [[FOO]] [[AGG2]] [[C]]
; CHECK:      OpReturnValue [[RET]]
; CHECK:      OpFunctionEnd

; CHECK:      [[FOO]] = OpFunction [[I32]] None [[FN2]]
; CHECK-NEXT: [[IN:%.+]] = OpFunctionParameter [[PAIR]]
; CHECK-NEXT: OpFunctionParameter [[I64]]
; CHECK:      [[FIRST:%.+]] = OpCompositeExtract [[I32]] [[IN]] 0
; CHECK:      [[BAR:%.+]] = OpFunctionCall [[I32]] [[FUN]] [[FIRST]]
; CHECK:      OpReturnValue [[BAR]]
; CHECK:      OpFunctionEnd

define i32 @goo(i32 %a, i16 %b, i64 %c) {
  %agg1 = insertvalue {i32, i16} undef, i32 %a, 0
  %agg2 = insertvalue {i32, i16} %agg1, i16 %b, 1
  %ret = call i32 @foo({i32, i16} %agg2, i64 %c)
  ret i32 %ret
}

;; TODO: test tailcall?
