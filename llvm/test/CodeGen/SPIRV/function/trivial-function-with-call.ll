; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

;; Debug info:
; CHECK:      OpName [[FOO:%.+]] "foo"
; CHECK:      OpName [[BAR:%.+]] "bar"

;; Types:
; CHECK-DAG:  [[I32:%.+]] = OpTypeInt 32
; CHECK-DAG:  [[VOID:%.+]] = OpTypeVoid
; CHECK-DAG:  [[FNVOID:%.+]] = OpTypeFunction [[VOID]] [[I32]]
; CHECK-DAG:  [[FNI32:%.+]] = OpTypeFunction [[I32]] [[I32]]
;; Function decl:
; CHECK:      [[BAR]] = OpFunction [[I32]] None [[FNI32]]
; CHECK-NEXT: OpFunctionParameter [[I32]]
; CHECK-NEXT: OpFunctionEnd
declare i32 @bar(i32 %x)
;; Function def:
; CHECK:      [[FOO]] = OpFunction [[VOID]] None [[FNVOID]]
; CHECK:      OpFunctionParameter
; CHECK:      OpLabel
; CHECK:      OpFunctionCall [[I32]] [[BAR]]
; CHECK:      OpReturn
; CHECK-NOT:  OpLabel
; CHECK:      OpFunctionEnd
define spir_func void @foo(i32 %x) {
  %call1 = call spir_func i32 @bar(i32 %x)
  ret void
}
