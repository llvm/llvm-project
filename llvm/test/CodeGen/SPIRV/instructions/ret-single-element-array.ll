; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#One:]] = OpConstant %[[#]] 1
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Float]] %[[#One]]
; CHECK-DAG: %[[#Nested:]] = OpTypeArray %[[#Array]] %[[#One]]
; CHECK: OpFunction
; CHECK: %[[#X:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#NestedVal:]] = OpCompositeInsert %[[#Nested]] %[[#X]] %[[#]] 0 0
; CHECK: %[[#Ret:]] = OpCompositeExtract %[[#Array]] %[[#NestedVal]] 0
; CHECK: OpReturnValue %[[#Ret]]

define spir_func [1 x float] @single_element_array(float %x) {
entry:
  %nested = insertvalue [1 x [1 x float]] zeroinitializer, float %x, 0, 0
  %ret = extractvalue [1 x [1 x float]] %nested, 0
  ret [1 x float] %ret
}
