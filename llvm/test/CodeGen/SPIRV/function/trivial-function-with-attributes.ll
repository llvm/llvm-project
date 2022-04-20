; RUN: llc -O0 %s -o - | FileCheck %s

target triple = "spirv32-unknown-unknown"

; FIXME: Are there any attributes that would make the IR invalid for SPIR-V?

; Names:
; CHECK-DAG: OpName [[FN1:%.+]] "fn1"
; CHECK-DAG: OpName [[FN2:%.+]] "fn2"
; CHECK-DAG: OpName [[FN3:%.+]] "fn3"
; CHECK-DAG: OpName [[FN4:%.+]] "fn4"
; CHECK-DAG: OpName [[FN5:%.+]] "fn5"
; CHECK-DAG: OpName [[FN6:%.+]] "fn6"
; CHECK-DAG: OpName [[FN7:%.+]] "fn7"

; CHECK-NOT: DAG-FENCE

; Types:
; CHECK: [[VOID:%.+]] = OpTypeVoid
; CHECK: [[FN:%.+]] = OpTypeFunction [[VOID]]


; Functions:

define void @fn1() noinline {
  ret void
}
; CHECK: [[FN1]] = OpFunction [[VOID]] DontInline [[FN]]
; CHECK-NOT: OpFunctionParameter
; CHECK: OpFunctionEnd


attributes #0 = { noinline }
define void @fn2() #0 {
  ret void
}
; CHECK: [[FN2]] = OpFunction [[VOID]] DontInline [[FN]]
; CHECK: OpFunctionEnd


define void @fn3() alwaysinline {
  ret void
}
; CHECK: [[FN3]] = OpFunction [[VOID]] Inline [[FN]]
; CHECK: OpFunctionEnd


; NOTE: inlinehint is not an actual requirement.
define void @fn4() inlinehint {
  ret void
}
; CHECK: [[FN4]] = OpFunction [[VOID]] None [[FN]]
; CHECK: OpFunctionEnd


define void @fn5() readnone {
  ret void
}
; CHECK: [[FN5]] = OpFunction [[VOID]] Pure [[FN]]
; CHECK: OpFunctionEnd


define void @fn6() readonly {
  ret void
}
; CHECK: [[FN6]] = OpFunction [[VOID]] Const [[FN]]
; CHECK: OpFunctionEnd


define void @fn7() alwaysinline readnone {
  ret void
}
; CHECK: [[FN7]] = OpFunction [[VOID]] Inline|Pure [[FN]]
; CHECK: OpFunctionEnd
