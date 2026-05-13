; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName [[VECTOR_ADD:%.+]] "vector_add"
; CHECK-DAG: OpName [[VECTOR_SUB:%.+]] "vector_sub"
; CHECK-DAG: OpName [[VECTOR_MUL:%.+]] "vector_mul"
; CHECK-DAG: OpName [[VECTOR_UDIV:%.+]] "vector_udiv"
; CHECK-DAG: OpName [[VECTOR_SDIV:%.+]] "vector_sdiv"
; CHECK-DAG: OpName [[VECTOR_UREM:%.+]] "vector_urem"
; CHECK-DAG: OpName [[VECTOR_SREM:%.+]] "vector_srem"
; CHECK-DAG: OpName [[VECTOR_SNEGATE:%.+]] "vector_snegate"

; CHECK-NOT: DAG-FENCE

; CHECK-DAG: [[I16:%.+]] = OpTypeInt 16
; CHECK-DAG: [[VECTOR:%.+]] = OpTypeVector [[I16]]
; CHECK-DAG: [[VECTOR_FN:%.+]] = OpTypeFunction [[VECTOR]] [[VECTOR]] [[VECTOR]]
; CHECK-DAG: [[VECTOR_FN1:%.+]] = OpTypeFunction [[VECTOR]] [[VECTOR]]
; CHECK-DAG: [[ZERO:%.+]] = OpConstantNull [[VECTOR]]

; CHECK-NOT: DAG-FENCE


;; Test add on vector:
define <2 x i16> @vector_add(<2 x i16> %a, <2 x i16> %b) {
    %c = add <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_ADD]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpIAdd [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test sub on vector:
define <2 x i16> @vector_sub(<2 x i16> %a, <2 x i16> %b) {
    %c = sub <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_SUB]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpISub [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test mul on vector:
define <2 x i16> @vector_mul(<2 x i16> %a, <2 x i16> %b) {
    %c = mul <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_MUL]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpIMul [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test udiv on vector:
define <2 x i16> @vector_udiv(<2 x i16> %a, <2 x i16> %b) {
    %c = udiv <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_UDIV]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpUDiv [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test sdiv on vector:
define <2 x i16> @vector_sdiv(<2 x i16> %a, <2 x i16> %b) {
    %c = sdiv <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_SDIV]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpSDiv [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test urem on vector:
define <2 x i16> @vector_urem(<2 x i16> %a, <2 x i16> %b) {
    %c = urem <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_UREM]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpUMod [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test srem on vector:
define <2 x i16> @vector_srem(<2 x i16> %a, <2 x i16> %b) {
    %c = srem <2 x i16> %a, %b
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_SREM]] = OpFunction [[VECTOR]] None [[VECTOR_FN]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK-NEXT: [[B:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpSRem [[VECTOR]] [[A]] [[B]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd


;; Test snegate on vector:
define <2 x i16> @vector_snegate(<2 x i16> %a) {
    %c = sub <2 x i16> zeroinitializer, %a
    ret <2 x i16> %c
}

; CHECK:      [[VECTOR_SNEGATE]] = OpFunction [[VECTOR]] None [[VECTOR_FN1]]
; CHECK-NEXT: [[A:%.+]] = OpFunctionParameter [[VECTOR]]
; CHECK:      OpLabel
; CHECK:      [[C:%.+]] = OpISub [[VECTOR]] [[ZERO]] [[A]]
; CHECK:      OpReturnValue [[C]]
; CHECK-NEXT: OpFunctionEnd
