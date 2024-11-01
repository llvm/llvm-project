; RUN: llc -O0 -opaque-pointers=0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; Check 'LLVM ==> SPIR-V' conversion of extractvalue/insertvalue.

%struct.arr = type { [7 x float] }
%struct.st = type { %struct.inner }
%struct.inner = type { float }

; CHECK-SPIRV:     %[[#float_ty:]] = OpTypeFloat 32
; CHECK-SPIRV:     %[[#int_ty:]] = OpTypeInt 32
; CHECK-SPIRV:     %[[#arr_size:]] = OpConstant %[[#int_ty]] 7
; CHECK-SPIRV:     %[[#array_ty:]] = OpTypeArray %[[#float_ty]] %[[#arr_size]]
; CHECK-SPIRV:     %[[#struct_ty:]] = OpTypeStruct %[[#array_ty]]
; CHECK-SPIRV:     %[[#struct_ptr_ty:]] = OpTypePointer CrossWorkgroup %[[#struct_ty]]
; CHECK-SPIRV:     %[[#array_ptr_ty:]] = OpTypePointer CrossWorkgroup %[[#array_ty]]
; CHECK-SPIRV:     %[[#struct1_in_ty:]] = OpTypeStruct %[[#float_ty]]
; CHECK-SPIRV:     %[[#struct1_ty:]] = OpTypeStruct %[[#struct1_in_ty]]
; CHECK-SPIRV:     %[[#struct1_ptr_ty:]] = OpTypePointer CrossWorkgroup %[[#struct1_ty]]
; CHECK-SPIRV:     %[[#struct1_in_ptr_ty:]] = OpTypePointer CrossWorkgroup %[[#struct1_in_ty]]
; CHECK-SPIRV-NOT: OpConstant %{{.*}} 2
; CHECK-SPIRV-NOT: OpConstant %{{.*}} 4
; CHECK-SPIRV-NOT: OpConstant %{{.*}} 5

; CHECK-SPIRV-LABEL:  OpFunction
; CHECK-SPIRV-NEXT:   %[[#object:]] = OpFunctionParameter %[[#struct_ptr_ty]]
; CHECK-SPIRV:        %[[#store_ptr:]] = OpInBoundsPtrAccessChain %[[#array_ptr_ty]] %[[#object]] %[[#]] %[[#]]
; CHECK-SPIRV:        %[[#extracted_array:]] = OpLoad %[[#array_ty]] %[[#store_ptr]] Aligned 4
; CHECK-SPIRV:        %[[#elem_4:]] = OpCompositeExtract %[[#float_ty]] %[[#extracted_array]] 4
; CHECK-SPIRV:        %[[#elem_2:]] = OpCompositeExtract %[[#float_ty]] %[[#extracted_array]] 2
; CHECK-SPIRV:        %[[#add:]] = OpFAdd %[[#float_ty]] %[[#elem_4]] %[[#elem_2]]
; CHECK-SPIRV:        %[[#inserted_array:]] = OpCompositeInsert %[[#array_ty]] %[[#add]] %[[#extracted_array]] 5
; CHECK-SPIRV:        OpStore %[[#store_ptr]] %[[#inserted_array]]
; CHECK-SPIRV-LABEL:  OpFunctionEnd

define spir_func void @array_test(%struct.arr addrspace(1)* %object) {
entry:
  %0 = getelementptr inbounds %struct.arr, %struct.arr addrspace(1)* %object, i32 0, i32 0
  %1 = load [7 x float], [7 x float] addrspace(1)* %0, align 4
  %2 = extractvalue [7 x float] %1, 4
  %3 = extractvalue [7 x float] %1, 2
  %4 = fadd float %2, %3
  %5 = insertvalue [7 x float] %1, float %4, 5
  store [7 x float] %5, [7 x float] addrspace(1)* %0
  ret void
}

; CHECK-SPIRV-LABEL:  OpFunction
; CHECK-SPIRV-NEXT:   %[[#object:]] = OpFunctionParameter %[[#struct1_ptr_ty]]
; CHECK-SPIRV:        %[[#store1_ptr:]] = OpInBoundsPtrAccessChain %[[#struct1_in_ptr_ty]] %[[#object]] %[[#]] %[[#]]
; CHECK-SPIRV:        %[[#extracted_struct:]] = OpLoad %[[#struct1_in_ty]] %[[#store1_ptr]] Aligned 4
; CHECK-SPIRV:        %[[#elem:]] = OpCompositeExtract %[[#float_ty]] %[[#extracted_struct]] 0
; CHECK-SPIRV:        %[[#add:]] = OpFAdd %[[#float_ty]] %[[#elem]] %[[#]]
; CHECK-SPIRV:        %[[#inserted_struct:]] = OpCompositeInsert %[[#struct1_in_ty]] %[[#add]] %[[#extracted_struct]] 0
; CHECK-SPIRV:        OpStore %[[#store1_ptr]] %[[#inserted_struct]]
; CHECK-SPIRV-LABEL:  OpFunctionEnd

define spir_func void @struct_test(%struct.st addrspace(1)* %object) {
entry:
  %0 = getelementptr inbounds %struct.st, %struct.st addrspace(1)* %object, i32 0, i32 0
  %1 = load %struct.inner, %struct.inner addrspace(1)* %0, align 4
  %2 = extractvalue %struct.inner %1, 0
  %3 = fadd float %2, 1.000000e+00
  %4 = insertvalue %struct.inner %1, float %3, 0
  store %struct.inner %4, %struct.inner addrspace(1)* %0
  ret void
}
