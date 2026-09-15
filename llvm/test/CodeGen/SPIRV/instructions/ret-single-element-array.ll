; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#One:]] = OpConstant %[[#]] 1
; CHECK-DAG: %[[#Array:]] = OpTypeArray %[[#Float]] %[[#One]]
; CHECK-DAG: %[[#ArrayInArray:]] = OpTypeArray %[[#Array]] %[[#One]]
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#Float]]
; CHECK-DAG: %[[#StructInStruct:]] = OpTypeStruct %[[#Struct]]
; CHECK-DAG: %[[#StructInArray:]] = OpTypeArray %[[#Struct]] %[[#One]]
; CHECK-DAG: %[[#ArrayInStruct:]] = OpTypeStruct %[[#Array]]

; CHECK: OpFunction
; CHECK: %[[#X1:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Agg1:]] = OpCompositeInsert %[[#ArrayInArray]] %[[#X1]] %[[#]] 0 0
; CHECK: %[[#Ret1:]] = OpCompositeExtract %[[#Array]] %[[#Agg1]] 0
; CHECK: OpReturnValue %[[#Ret1]]
define spir_func [1 x float] @single_element_array(float %x) {
entry:
  %nested = insertvalue [1 x [1 x float]] zeroinitializer, float %x, 0, 0
  %ret = extractvalue [1 x [1 x float]] %nested, 0
  ret [1 x float] %ret
}

; CHECK: OpFunction
; CHECK: %[[#X2:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Agg2:]] = OpCompositeInsert %[[#StructInStruct]] %[[#X2]] %[[#]] 0 0
; CHECK: %[[#Ret2:]] = OpCompositeExtract %[[#Struct]] %[[#Agg2]] 0
; CHECK: OpReturnValue %[[#Ret2]]
define spir_func { float } @single_element_struct(float %x) {
entry:
  %nested = insertvalue { { float } } zeroinitializer, float %x, 0, 0
  %ret = extractvalue { { float } } %nested, 0
  ret { float } %ret
}

; CHECK: OpFunction
; CHECK: %[[#X3:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Agg3:]] = OpCompositeInsert %[[#StructInArray]] %[[#X3]] %[[#]] 0 0
; CHECK: %[[#Ret3:]] = OpCompositeExtract %[[#Struct]] %[[#Agg3]] 0
; CHECK: OpReturnValue %[[#Ret3]]
define spir_func { float } @struct_in_array(float %x) {
entry:
  %nested = insertvalue [1 x { float }] zeroinitializer, float %x, 0, 0
  %ret = extractvalue [1 x { float }] %nested, 0
  ret { float } %ret
}

; CHECK: OpFunction
; CHECK: %[[#X4:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Agg4:]] = OpCompositeInsert %[[#ArrayInStruct]] %[[#X4]] %[[#]] 0 0
; CHECK: %[[#Ret4:]] = OpCompositeExtract %[[#Array]] %[[#Agg4]] 0
; CHECK: OpReturnValue %[[#Ret4]]
define spir_func [1 x float] @array_in_struct(float %x) {
entry:
  %nested = insertvalue { [1 x float] } zeroinitializer, float %x, 0, 0
  %ret = extractvalue { [1 x float] } %nested, 0
  ret [1 x float] %ret
}
