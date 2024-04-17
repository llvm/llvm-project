; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName [[ADD:%.*]] "test_add"
; CHECK-DAG: OpName [[SUB:%.*]] "test_sub"
; CHECK-DAG: OpName [[MIN:%.*]] "test_min"
; CHECK-DAG: OpName [[MAX:%.*]] "test_max"
; CHECK-DAG: OpName [[UMIN:%.*]] "test_umin"
; CHECK-DAG: OpName [[UMAX:%.*]] "test_umax"
; CHECK-DAG: OpName [[AND:%.*]] "test_and"
; CHECK-DAG: OpName [[OR:%.*]] "test_or"
; CHECK-DAG: OpName [[XOR:%.*]] "test_xor"

; CHECK-DAG: [[I32Ty:%.*]] = OpTypeInt 32 0
;; Device scope is encoded with constant 1
; CHECK-DAG: [[SCOPE:%.*]] = OpConstant [[I32Ty]] 1
;; "sequentially consistent" maps to constant 16
; CHECK-DAG: [[SEQ:%.*]] = OpConstant [[I32Ty]] 16

; CHECK:      [[ADD]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicIAdd [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_add(i32* %ptr, i32 %val) {
  %r = atomicrmw add i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[SUB]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicISub [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_sub(i32* %ptr, i32 %val) {
  %r = atomicrmw sub i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[MIN]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicSMin [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_min(i32* %ptr, i32 %val) {
  %r = atomicrmw min i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[MAX]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicSMax [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_max(i32* %ptr, i32 %val) {
  %r = atomicrmw max i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[UMIN]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicUMin [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_umin(i32* %ptr, i32 %val) {
  %r = atomicrmw umin i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[UMAX]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicUMax [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_umax(i32* %ptr, i32 %val) {
  %r = atomicrmw umax i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[AND]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicAnd [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_and(i32* %ptr, i32 %val) {
  %r = atomicrmw and i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[OR]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicOr [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_or(i32* %ptr, i32 %val) {
  %r = atomicrmw or i32* %ptr, i32 %val seq_cst
  ret i32 %r
}

; CHECK:      [[XOR]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[BC_A:%.*]] = OpBitcast %[[#]] [[A]]
; CHECK-NEXT: [[R:%.*]] = OpAtomicXor [[I32Ty]] [[BC_A]] [[SCOPE]] [[SEQ]] [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_xor(i32* %ptr, i32 %val) {
  %r = atomicrmw xor i32* %ptr, i32 %val seq_cst
  ret i32 %r
}
