; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | not spirv-val 2>&1 | FileCheck --check-prefix=SPIRV-VAL %s %}

; spirv-val poorly supports the SPV_INTEL_function_pointers extension.
; In this case the function is declared after the constant so it fails.
; SPIRV-VAL: ID '{{.*}}[%f0]' has not been defined
; SPIRV-VAL: OpConstantFunctionPointerINTEL %_ptr_CodeSectionINTEL_11 %f0

; CHECK: OpCapability Kernel
; CHECK-DAG: OpCapability FunctionPointersINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_function_pointers"
; CHECK-DAG: OpName %[[#fArray:]] "array"
; CHECK-DAG: OpName %[[#fStruct:]] "struct"

; CHECK-DAG: %[[#Int8Ty:]] = OpTypeInt 8 0
; CHECK: %[[#GlobalInt8PtrTy:]] = OpTypePointer CrossWorkgroup %[[#Int8Ty]]
; CHECK: %[[#VoidTy:]] = OpTypeVoid
; CHECK: %[[#TestFnTy:]] = OpTypeFunction %[[#VoidTy]] %[[#GlobalInt8PtrTy]]
; CHECK: %[[#F16Ty:]] = OpTypeFloat 16
; CHECK: %[[#t_halfTy:]] = OpTypeStruct %[[#F16Ty]]
; CHECK: %[[#FnTy:]] = OpTypeFunction %[[#t_halfTy]] %[[#GlobalInt8PtrTy]] %[[#t_halfTy]]
; CHECK: %[[#IntelFnPtrTy:]] = OpTypePointer CodeSectionINTEL %[[#FnTy]]
; CHECK: %[[#Int8PtrTy:]] = OpTypePointer Function %[[#Int8Ty]]
; CHECK: %[[#Int32Ty:]] = OpTypeInt 32 0
; CHECK: %[[#I32Const3:]] = OpConstant %[[#Int32Ty]] 3
; CHECK: %[[#FnArrTy:]] = OpTypeArray %[[#Int8PtrTy]] %[[#I32Const3]]
; CHECK: %[[#GlobalFnArrPtrTy:]] = OpTypePointer CrossWorkgroup %[[#FnArrTy]]
; CHECK: %[[#GlobalFnPtrTy:]] = OpTypePointer CrossWorkgroup %[[#FnTy]]
; CHECK: %[[#FnPtrTy:]] = OpTypePointer Function %[[#FnTy]]
; CHECK: %[[#StructWithPfnTy:]] = OpTypeStruct %[[#FnPtrTy]] %[[#FnPtrTy]] %[[#FnPtrTy]]
; CHECK: %[[#ArrayOfPfnTy:]] = OpTypeArray %[[#FnPtrTy]] %[[#I32Const3]]
; CHECK: %[[#Int64Ty:]] = OpTypeInt 64 0
; CHECK: %[[#GlobalStructWithPfnPtrTy:]] = OpTypePointer CrossWorkgroup %[[#StructWithPfnTy]]
; CHECK: %[[#GlobalArrOfPfnPtrTy:]] = OpTypePointer CrossWorkgroup %[[#ArrayOfPfnTy]]
; CHECK: %[[#I64Const2:]] = OpConstant %[[#Int64Ty]] 2
; CHECK: %[[#I64Const1:]] = OpConstant %[[#Int64Ty]] 1
; CHECK: %[[#I64Const0:]] = OpConstantNull %[[#Int64Ty]]
; CHECK: %[[#f0Pfn:]] = OpConstantFunctionPointerINTEL %[[#IntelFnPtrTy]] %28
; CHECK: %[[#f1Pfn:]] = OpConstantFunctionPointerINTEL %[[#IntelFnPtrTy]] %32
; CHECK: %[[#f2Pfn:]] = OpConstantFunctionPointerINTEL %[[#IntelFnPtrTy]] %36
; CHECK: %[[#f0Cast:]] = OpSpecConstantOp %[[#FnPtrTy]] Bitcast %[[#f0Pfn]]
; CHECK: %[[#f1Cast:]] = OpSpecConstantOp %[[#FnPtrTy]] Bitcast %[[#f1Pfn]]
; CHECK: %[[#f2Cast:]] = OpSpecConstantOp %[[#FnPtrTy]] Bitcast %[[#f2Pfn]]
; CHECK: %[[#fnptrTy:]] = OpConstantComposite %[[#ArrayOfPfnTy]] %[[#f0Cast]] %[[#f1Cast]] %[[#f2Cast]]
; CHECK: %[[#fnptr:]] = OpVariable %[[#GlobalArrOfPfnPtrTy]] CrossWorkgroup %[[#fnptrTy]]
; CHECK: %[[#fnstructTy:]] = OpConstantComposite %[[#StructWithPfnTy]] %[[#f0Cast]] %[[#f1Cast]] %[[#f2Cast]]
; CHECK: %[[#fnstruct:]] = OpVariable %[[#GlobalStructWithPfnPtrTy:]] CrossWorkgroup %[[#fnstructTy]]
; CHECK-DAG: %[[#GlobalInt8PtrPtrTy:]] = OpTypePointer CrossWorkgroup %[[#Int8PtrTy]]
; CHECK: %[[#StructWithPtrTy:]] = OpTypeStruct %[[#Int8PtrTy]] %[[#Int8PtrTy]] %[[#Int8PtrTy]]
; CHECK: %[[#GlobalStructWithPtrPtrTy:]] = OpTypePointer CrossWorkgroup %[[#StructWithPtrTy]]
; CHECK: %[[#I32Const2:]] = OpConstant %[[#Int32Ty]] 2
; CHECK: %[[#I32Const1:]] = OpConstant %[[#Int32Ty]] 1
; CHECK: %[[#I32Const0:]] = OpConstantNull %[[#Int32Ty]]
; CHECK: %[[#GlobalFnPtrPtrTy:]] = OpTypePointer CrossWorkgroup %[[#FnPtrTy]]
%t_half = type { half }
%struct.anon = type { ptr, ptr, ptr }

declare spir_func %t_half @f0(ptr addrspace(1) %a, %t_half %b)
declare spir_func %t_half @f1(ptr addrspace(1) %a, %t_half %b)
declare spir_func %t_half @f2(ptr addrspace(1) %a, %t_half %b)

@fnptr = addrspace(1) constant [3 x ptr] [ptr @f0, ptr @f1, ptr @f2]
@fnstruct = addrspace(1) constant %struct.anon { ptr @f0, ptr @f1, ptr @f2 }, align 8

; CHECK-DAG: %[[#fArray]] = OpFunction %[[#VoidTy]] None %[[#TestFnTy]]
;	CHECK-DAG: %[[#fnptrCast:]] = OpBitcast %[[#GlobalFnArrPtrTy]] %[[#fnptr]]
; CHECK: %[[#f0GEP:]] = OpInBoundsPtrAccessChain %[[#GlobalFnArrPtrTy]] %[[#fnptrCast]] %[[#I64Const0]]
; CHECK: %[[#f0GEPCast:]] = OpBitcast %[[#GlobalFnPtrTy]] %[[#f0GEP]]
; CHECK: %[[#f1GEP:]] = OpInBoundsPtrAccessChain %[[#GlobalFnArrPtrTy]] %[[#fnptrCast]] %[[#I64Const1]]
; CHECK: %[[#f1GEPCast:]] = OpBitcast %[[#GlobalFnPtrTy]] %[[#f1GEP]]
; CHECK: %[[#f2GEP:]] = OpInBoundsPtrAccessChain %[[#GlobalFnArrPtrTy]] %[[#fnptrCast]] %[[#I64Const2]]
; CHECK: %[[#f2GEPCast:]] = OpBitcast %[[#GlobalFnPtrTy]] %[[#f2GEP]]
; CHECK: %{{.*}} = OpFunctionPointerCallINTEL %[[#t_halfTy]] %[[#f0GEPCast]]
; CHECK: %{{.*}} = OpFunctionPointerCallINTEL %[[#t_halfTy]] %[[#f1GEPCast]]
; CHECK: %{{.*}} = OpFunctionPointerCallINTEL %[[#t_halfTy]] %[[#f2GEPCast]]
define spir_func void @array(ptr addrspace(1) %p) {
entry:
  %f = getelementptr inbounds [3 x ptr], ptr addrspace(1) @fnptr, i64 0
  %g = getelementptr inbounds [3 x ptr], ptr addrspace(1) @fnptr, i64 1
  %h = getelementptr inbounds [3 x ptr], ptr addrspace(1) @fnptr, i64 2
  %0 = call spir_func addrspace(1) %t_half %f(ptr addrspace(1) %p, %t_half poison)
  %1 = call spir_func addrspace(1) %t_half %g(ptr addrspace(1) %p, %t_half %0)
  %2 = call spir_func addrspace(1) %t_half %h(ptr addrspace(1) %p, %t_half %1)

  ret void
}

; CHECK-DAG: %[[#fStruct]] = OpFunction %[[#VoidTy]] None %[[#TestFnTy]]
; CHECK-DAG: %[[#fnStructCast0:]] = OpBitcast %[[#GlobalInt8PtrPtrTy]] %[[#fnstruct]]
; CHECK: %[[#fnStructCast1:]] = OpBitcast %[[#GlobalFnPtrPtrTy]] %[[#fnStructCast0]]
; CHECK: %[[#f0Load:]] = OpLoad %[[#FnPtrTy]] %[[#fnStructCast1]]
; CHECK: %[[#fnStructCast2:]] = OpBitcast %[[#GlobalStructWithPtrPtrTy]] %[[#fnstruct]]
; CHECK: %[[#f1GEP:]] = OpInBoundsPtrAccessChain %[[#GlobalInt8PtrPtrTy]] %[[#fnStructCast2]] %[[#I32Const0]] %[[#I32Const1]]
; CHECK: %[[#f1GEPCast:]] = OpBitcast %[[#GlobalFnPtrPtrTy]] %[[#f1GEP]]
; CHECK: %[[#f1Load:]] = OpLoad %[[#FnPtrTy]] %[[#f1GEPCast]]
; CHECK: %[[#f2GEP:]] = OpInBoundsPtrAccessChain %[[#GlobalInt8PtrPtrTy]] %[[#fnStructCast2]] %[[#I32Const0]] %[[#I32Const2]]
; CHECK: %[[#f2GEPCast:]] = OpBitcast %[[#GlobalFnPtrPtrTy]] %[[#f2GEP]]
; CHECK: %[[#f2Load:]] = OpLoad %[[#FnPtrTy]] %[[#f2GEPCast]]
; CHECK: %{{.*}} = OpFunctionPointerCallINTEL %[[#t_halfTy]] %[[#f0Load]]
; CHECK: %{{.*}} = OpFunctionPointerCallINTEL %[[#t_halfTy]] %[[#f1Load]]
; CHECK: %{{.*}} = OpFunctionPointerCallINTEL %[[#t_halfTy]] %[[#f2Load]]
define spir_func void @struct(ptr addrspace(1) %p) {
entry:
  %f = load ptr, ptr addrspace(1) @fnstruct
  %g = load ptr, ptr addrspace(1) getelementptr inbounds (%struct.anon, ptr addrspace(1) @fnstruct, i32 0, i32 1)
  %h = load ptr, ptr addrspace(1) getelementptr inbounds (%struct.anon, ptr addrspace(1) @fnstruct, i32 0, i32 2)
  %0 = call spir_func noundef %t_half %f(ptr addrspace(1) %p, %t_half poison)
  %1 = call spir_func noundef %t_half %g(ptr addrspace(1) %p, %t_half %0)
  %2 = call spir_func noundef %t_half %h(ptr addrspace(1) %p, %t_half %1)

  ret void
}
