; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that <1 x T> nested inside aggregate types is scalarized to T.

; CHECK-NOT: OpTypeVector
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PtrFloat:]] = OpTypePointer Function %[[#FloatTy]]
; CHECK-DAG: %[[#Const8:]] = OpConstant %[[#IntTy]] 8
; CHECK-DAG: %[[#Const4:]] = OpConstant %[[#IntTy]] 4
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#IntTy]] 2
; CHECK-DAG: %[[#Float1:]] = OpConstant %[[#FloatTy]] 1
; CHECK-DAG: %[[#Float2:]] = OpConstant %[[#FloatTy]] 2
; CHECK-DAG: %[[#Float3:]] = OpConstant %[[#FloatTy]] 3
; CHECK-DAG: %[[#Int42:]] = OpConstant %[[#IntTy]] 42
; CHECK-DAG: %[[#Int7:]] = OpConstant %[[#IntTy]] 7
; CHECK-DAG: %[[#Arr8Float:]] = OpTypeArray %[[#FloatTy]] %[[#Const8]]
; CHECK-DAG: %[[#PtrArr8Float:]] = OpTypePointer Function %[[#Arr8Float]]
; CHECK-DAG: %[[#Arr4x8Float:]] = OpTypeArray %[[#Arr8Float]] %[[#Const4]]
; CHECK-DAG: %[[#Arr4x4x8Float:]] = OpTypeArray %[[#Arr4x8Float]] %[[#Const4]]
; CHECK-DAG: %[[#PtrArr4x4x8Float:]] = OpTypePointer Function %[[#Arr4x4x8Float]]
; CHECK-DAG: %[[#StructFloatInt:]] = OpTypeStruct %[[#FloatTy]] %[[#IntTy]]
; CHECK-DAG: %[[#PtrStructFloatInt:]] = OpTypePointer Function %[[#StructFloatInt]]
; CHECK-DAG: %[[#Arr4Float:]] = OpTypeArray %[[#FloatTy]] %[[#Const4]]
; CHECK-DAG: %[[#Arr2Int:]] = OpTypeArray %[[#IntTy]] %[[#Const2]]
; CHECK-DAG: %[[#StructFloatArr2Int:]] = OpTypeStruct %[[#FloatTy]] %[[#Arr2Int]]

; CHECK: OpFunction
; CHECK: %[[#ArrVar:]] = OpVariable %[[#PtrArr8Float]] Function
; CHECK: %[[#ArrGep:]] = OpPtrAccessChain %[[#PtrFloat]] %[[#ArrVar]]
; CHECK: OpStore %[[#ArrGep]] %[[#Float1]] Aligned 4
; CHECK: %[[#ArrLoad:]] = OpLoad %[[#FloatTy]] %[[#ArrGep]] Aligned 4
; CHECK: OpStore %[[#]] %[[#ArrLoad]] Aligned 4
; CHECK: OpFunctionEnd
define spir_kernel void @vec1_in_array(ptr addrspace(1) %out) {
entry:
  %v = alloca [8 x <1 x float>], align 4, addrspace(0)
  %p = getelementptr [8 x <1 x float>], ptr addrspace(0) %v, i32 0, i32 0
  store <1 x float> <float 1.0>, ptr addrspace(0) %p, align 4
  %r = load <1 x float>, ptr addrspace(0) %p, align 4
  %s = extractelement <1 x float> %r, i32 0
  store float %s, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#NestedVar:]] = OpVariable %[[#PtrArr4x4x8Float]] Function
; CHECK: %[[#NestedGep:]] = OpPtrAccessChain %[[#PtrFloat]] %[[#NestedVar]]
; CHECK: OpStore %[[#NestedGep]] %[[#Float2]] Aligned 4
; CHECK: %[[#NestedLoad:]] = OpLoad %[[#FloatTy]] %[[#NestedGep]] Aligned 4
; CHECK: OpStore %[[#]] %[[#NestedLoad]] Aligned 4
; CHECK: OpFunctionEnd
define spir_kernel void @vec1_in_nested_array(ptr addrspace(1) %out) {
entry:
  %v = alloca [4 x [4 x [8 x <1 x float>]]], align 4, addrspace(0)
  %p = getelementptr [4 x [4 x [8 x <1 x float>]]], ptr addrspace(0) %v, i32 0, i32 0, i32 0, i32 0
  store <1 x float> <float 2.0>, ptr addrspace(0) %p, align 4
  %r = load <1 x float>, ptr addrspace(0) %p, align 4
  %s = extractelement <1 x float> %r, i32 0
  store float %s, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#StructVar:]] = OpVariable %[[#PtrStructFloatInt]] Function
; CHECK: %[[#StructGep:]] = OpPtrAccessChain %[[#PtrFloat]] %[[#StructVar]]
; CHECK: OpStore %[[#StructGep]] %[[#Float1]] Aligned 4
; CHECK: %[[#StructLoad:]] = OpLoad %[[#FloatTy]] %[[#StructGep]] Aligned 4
; CHECK: OpStore %[[#]] %[[#StructLoad]] Aligned 4
; CHECK: OpFunctionEnd
define spir_kernel void @vec1_in_struct(ptr addrspace(1) %out) {
entry:
  %v = alloca {<1 x float>, <1 x i32>}, align 4, addrspace(0)
  %p = getelementptr {<1 x float>, <1 x i32>}, ptr addrspace(0) %v, i32 0, i32 0
  store <1 x float> <float 1.0>, ptr addrspace(0) %p, align 4
  %r = load <1 x float>, ptr addrspace(0) %p, align 4
  %s = extractelement <1 x float> %r, i32 0
  store float %s, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#ArrInsert1:]] = OpCompositeInsert %[[#Arr4Float]] %[[#Float1]] %[[#]] 0
; CHECK: %[[#ArrInsert2:]] = OpCompositeInsert %[[#Arr4Float]] %[[#Float2]] %[[#ArrInsert1]] 1
; CHECK: %[[#ArrExtract:]] = OpCompositeExtract %[[#FloatTy]] %[[#ArrInsert2]] 1
; CHECK: OpStore %[[#]] %[[#ArrExtract]] Aligned 4
; CHECK: OpFunctionEnd
define spir_kernel void @vec1_insertvalue_extractvalue_array(ptr addrspace(1) %out) {
entry:
  %a = insertvalue [4 x <1 x float>] poison, <1 x float> <float 1.0>, 0
  %a2 = insertvalue [4 x <1 x float>] %a, <1 x float> <float 2.0>, 1
  %v = extractvalue [4 x <1 x float>] %a2, 1
  %s = extractelement <1 x float> %v, i32 0
  store float %s, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#StructInsert1:]] = OpCompositeInsert %[[#StructFloatInt]] %[[#Float1]] %[[#]] 0
; CHECK: %[[#StructInsert2:]] = OpCompositeInsert %[[#StructFloatInt]] %[[#Int42]] %[[#StructInsert1]] 1
; CHECK: %[[#StructExtract:]] = OpCompositeExtract %[[#FloatTy]] %[[#StructInsert2]] 0
; CHECK: OpStore %[[#]] %[[#StructExtract]] Aligned 4
; CHECK: OpFunctionEnd
define spir_kernel void @vec1_insertvalue_extractvalue_struct(ptr addrspace(1) %out) {
entry:
  %a = insertvalue {<1 x float>, <1 x i32>} poison, <1 x float> <float 1.0>, 0
  %a2 = insertvalue {<1 x float>, <1 x i32>} %a, <1 x i32> <i32 42>, 1
  %v = extractvalue {<1 x float>, <1 x i32>} %a2, 0
  %s = extractelement <1 x float> %v, i32 0
  store float %s, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#MixedInsert1:]] = OpCompositeInsert %[[#StructFloatArr2Int]] %[[#Float3]] %[[#]] 0
; CHECK: %[[#MixedInsert2:]] = OpCompositeInsert %[[#StructFloatArr2Int]] %[[#Int7]] %[[#MixedInsert1]] 1 0
; CHECK: %[[#MixedExtract:]] = OpCompositeExtract %[[#FloatTy]] %[[#MixedInsert2]] 0
; CHECK: OpStore %[[#]] %[[#MixedExtract]] Aligned 4
; CHECK: OpFunctionEnd
define spir_kernel void @vec1_struct_with_nested_array(ptr addrspace(1) %out) {
entry:
  %s = insertvalue {<1 x float>, [2 x <1 x i32>]} poison, <1 x float> <float 3.0>, 0
  %s2 = insertvalue {<1 x float>, [2 x <1 x i32>]} %s, <1 x i32> <i32 7>, 1, 0
  %v = extractvalue {<1 x float>, [2 x <1 x i32>]} %s2, 0
  %sc = extractelement <1 x float> %v, i32 0
  store float %sc, ptr addrspace(1) %out, align 4
  ret void
}
