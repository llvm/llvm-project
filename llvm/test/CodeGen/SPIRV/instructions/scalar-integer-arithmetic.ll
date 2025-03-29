; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName [[BOOL_ADD:%.+]] "bool_add"
; CHECK-DAG: OpName [[BOOL_SUB:%.+]] "bool_sub"
; CHECK-DAG: OpName [[SCALAR_ADD:%.+]] "scalar_add"
; CHECK-DAG: OpName [[SCALAR_SUB:%.+]] "scalar_sub"
; CHECK-DAG: OpName [[SCALAR_MUL:%.+]] "scalar_mul"
; CHECK-DAG: OpName [[SCALAR_UDIV:%.+]] "scalar_udiv"
; CHECK-DAG: OpName [[SCALAR_SDIV:%.+]] "scalar_sdiv"
;; TODO: add tests for urem + srem
;; TODO: add test for OpSNegate

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[BOOL:%.+]] = OpTypeBool
; CHECK-DAG: [[SCALAR:%.+]] = OpTypeInt 32
; CHECK-DAG: [[SCALAR_FN:%.+]] = OpTypeFunction [[SCALAR]] [[SCALAR]] [[SCALAR]]
; CHECK-DAG: [[BOOL_FN:%.+]] = OpTypeFunction [[BOOL]] [[BOOL]] [[BOOL]]

; CHECK-NOT: DAG-FENCE


;; Test add on scalar:
define i1 @bool_add(i1 %a, i1 %b) {
    %c = add i1 %a, %b
    ret i1 %c
}

; CHECK:      [[BOOL_ADD]] = OpFunction [[BOOL]] None [[BOOL_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[BOOL]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[BOOL]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpLogicalNotEqual [[BOOL]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd

define i32 @scalar_add(i32 %a, i32 %b) {
    %c = add i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_ADD]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpIAdd [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test sub on scalar:
define i1 @bool_sub(i1 %a, i1 %b) {
    %c = sub i1 %a, %b
    ret i1 %c
}

; CHECK:      [[BOOL_SUB]] = OpFunction [[BOOL]] None [[BOOL_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[BOOL]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[BOOL]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpLogicalNotEqual [[BOOL]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd

define i32 @scalar_sub(i32 %a, i32 %b) {
    %c = sub i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_SUB]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpISub [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test mul on scalar:
define i32 @scalar_mul(i32 %a, i32 %b) {
    %c = mul i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_MUL]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpIMul [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test udiv on scalar:
define i32 @scalar_udiv(i32 %a, i32 %b) {
    %c = udiv i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_UDIV]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpUDiv [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test sdiv on scalar:
define i32 @scalar_sdiv(i32 %a, i32 %b) {
    %c = sdiv i32 %a, %b
    ret i32 %c
}

; CHECK:      [[SCALAR_SDIV]] = OpFunction [[SCALAR]] None [[SCALAR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[SCALAR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpSDiv [[SCALAR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
