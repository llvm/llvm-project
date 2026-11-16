; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpTypeVector
; CHECK-DAG: %[[#FloatTy:]] = OpTypeFloat 32
; CHECK-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#One:]] = OpConstant %[[#IntTy]] 1
; CHECK-DAG: %[[#Vec1FloatTy:]] = OpTypeVectorIdEXT %[[#FloatTy]] %[[#One]]
; CHECK-DAG: %[[#Vec1IntTy:]] = OpTypeVectorIdEXT %[[#IntTy]] %[[#One]]
; CHECK-DAG: %[[#PtrFloat:]] = OpTypePointer Function %[[#Vec1FloatTy]]
; CHECK-DAG: %[[#Const8:]] = OpConstant %[[#IntTy]] 8
; CHECK-DAG: %[[#Const4:]] = OpConstant %[[#IntTy]] 4
; CHECK-DAG: %[[#Const2:]] = OpConstant %[[#IntTy]] 2
; CHECK-DAG: %[[#Float1:]] = OpConstant %[[#FloatTy]] 1
; CHECK-DAG: %[[#Vec1Float1:]] = OpConstantComposite %[[#Vec1FloatTy]] %[[#Float1]]
; CHECK-DAG: %[[#Float2:]] = OpConstant %[[#FloatTy]] 2
; CHECK-DAG: %[[#Vec1Float2:]] = OpConstantComposite %[[#Vec1FloatTy]] %[[#Float2]]
; CHECK-DAG: %[[#Float3:]] = OpConstant %[[#FloatTy]] 3
; CHECK-DAG: %[[#Vec1Float3:]] = OpConstantComposite %[[#Vec1FloatTy]] %[[#Float3]]
; CHECK-DAG: %[[#Int42:]] = OpConstant %[[#IntTy]] 42
; CHECK-DAG: %[[#Vec1Int42:]] = OpConstantComposite %[[#Vec1IntTy]] %[[#Int42]]
; CHECK-DAG: %[[#Int7:]] = OpConstant %[[#IntTy]] 7
; CHECK-DAG: %[[#Vec1Int7:]] = OpConstantComposite %[[#Vec1IntTy]] %[[#Int7]]
; CHECK-DAG: %[[#Arr8Float:]] = OpTypeArray %[[#Vec1FloatTy]] %[[#Const8]]
; CHECK-DAG: %[[#PtrArr8Float:]] = OpTypePointer Function %[[#Arr8Float]]
; CHECK-DAG: %[[#Arr4x8Float:]] = OpTypeArray %[[#Arr8Float]] %[[#Const4]]
; CHECK-DAG: %[[#Arr4x4x8Float:]] = OpTypeArray %[[#Arr4x8Float]] %[[#Const4]]
; CHECK-DAG: %[[#PtrArr4x4x8Float:]] = OpTypePointer Function %[[#Arr4x4x8Float]]
; CHECK-DAG: %[[#StructFloatInt:]] = OpTypeStruct %[[#Vec1FloatTy]] %[[#Vec1IntTy]]
; CHECK-DAG: %[[#PtrStructFloatInt:]] = OpTypePointer Function %[[#StructFloatInt]]
; CHECK-DAG: %[[#Arr4Float:]] = OpTypeArray %[[#Vec1FloatTy]] %[[#Const4]]
; CHECK-DAG: %[[#Arr2Int:]] = OpTypeArray %[[#Vec1IntTy]] %[[#Const2]]
; CHECK-DAG: %[[#StructFloatArr2Int:]] = OpTypeStruct %[[#Vec1FloatTy]] %[[#Arr2Int]]

; CHECK: OpFunction
; CHECK: %[[#ArrVar:]] = OpVariable %[[#PtrArr8Float]] Function
; CHECK: %[[#ArrGep:]] = OpPtrAccessChain %[[#PtrFloat]] %[[#ArrVar]]
; CHECK: OpStore %[[#ArrGep]] %[[#Vec1Float1]] Aligned 4
; CHECK: %[[#ArrLoad:]] = OpLoad %[[#Vec1FloatTy]] %[[#ArrGep]] Aligned 4
; CHECK: %[[#ExtractElt:]] = OpCompositeExtract %[[#FloatTy]] %[[#ArrLoad]] 0
; CHECK: OpStore %[[#]] %[[#ExtractElt]] Aligned 4
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
; CHECK: OpStore %[[#NestedGep]] %[[#Vec1Float2]] Aligned 4
; CHECK: %[[#NestedLoad:]] = OpLoad %[[#Vec1FloatTy]] %[[#NestedGep]] Aligned 4
; CHECK: %[[#ExtractElt1:]] = OpCompositeExtract %[[#FloatTy]] %[[#NestedLoad]] 0
; CHECK: OpStore %[[#]] %[[#ExtractElt1]] Aligned 4
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
; CHECK: OpStore %[[#StructGep]] %[[#Vec1Float1]] Aligned 4
; CHECK: %[[#StructLoad:]] = OpLoad %[[#Vec1FloatTy]] %[[#StructGep]] Aligned 4
; CHECK: %[[#ExtractElt2:]] = OpCompositeExtract %[[#FloatTy]] %[[#StructLoad]] 0
; CHECK: OpStore %[[#]] %[[#ExtractElt2]] Aligned 4
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
; CHECK: %[[#ArrInsert1:]] = OpCompositeInsert %[[#Arr4Float]] %[[#Vec1Float1]] %[[#]] 0
; CHECK: %[[#ArrInsert2:]] = OpCompositeInsert %[[#Arr4Float]] %[[#Vec1Float2]] %[[#ArrInsert1]] 1
; CHECK: %[[#ArrExtract:]] = OpCompositeExtract %[[#Vec1FloatTy]] %[[#ArrInsert2]] 1
; CHECK: %[[#ExtractElt3:]] = OpCompositeExtract %[[#FloatTy]] %[[#ArrExtract]] 0
; CHECK: OpStore %[[#]] %[[#ExtractElt3]] Aligned 4
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
; CHECK: %[[#StructInsert1:]] = OpCompositeInsert %[[#StructFloatInt]] %[[#Vec1Float1]] %[[#]] 0
; CHECK: %[[#StructInsert2:]] = OpCompositeInsert %[[#StructFloatInt]] %[[#Vec1Int42]] %[[#StructInsert1]] 1
; CHECK: %[[#StructExtract:]] = OpCompositeExtract %[[#Vec1FloatTy]] %[[#StructInsert2]] 0
; CHECK: %[[#ExtractElt4:]] = OpCompositeExtract %[[#FloatTy]] %[[#StructExtract]] 0
; CHECK: OpStore %[[#]] %[[#ExtractElt4]] Aligned 4
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
; CHECK: %[[#MixedInsert1:]] = OpCompositeInsert %[[#StructFloatArr2Int]] %[[#Vec1Float3]] %[[#]] 0
; CHECK: %[[#MixedInsert2:]] = OpCompositeInsert %[[#StructFloatArr2Int]] %[[#Vec1Int7]] %[[#MixedInsert1]] 1 0
; CHECK: %[[#MixedExtract:]] = OpCompositeExtract %[[#Vec1FloatTy]] %[[#MixedInsert2]] 0
; CHECK: %[[#ExtractElt5:]] = OpCompositeExtract %[[#FloatTy]] %[[#MixedExtract]] 0
; CHECK: OpStore %[[#]] %[[#ExtractElt5]] Aligned 4
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
