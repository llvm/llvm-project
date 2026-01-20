; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for generic_cast_to_ptr_explicit are lowered correctly.

; CHECK: %[[#Char:]] = OpTypeInt 8 0
; CHECK: %[[#GenericPtr:]] = OpTypePointer Generic %[[#Char]]
; CHECK: %[[#GlobalPtr:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK: %[[#LocalPtr:]] = OpTypePointer Workgroup %[[#Char]]
; CHECK: %[[#PrivatePtr:]] = OpTypePointer Function %[[#Char]]

; CHECK: OpFunction %[[#GlobalPtr]]
; CHECK-NEXT: %[[#Arg:]] = OpFunctionParameter %[[#GenericPtr]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpGenericCastToPtrExplicit %[[#GlobalPtr]] %[[#Arg]] CrossWorkgroup
define ptr addrspace(1) @test_to_global(ptr addrspace(4) noundef %ptr) {
entry:
  %cast = call spir_func noundef ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit.p1(ptr addrspace(4) noundef %ptr)
  ret ptr addrspace(1) %cast
}

; CHECK: OpFunction %[[#LocalPtr]]
; CHECK-NEXT: %[[#Arg:]] = OpFunctionParameter %[[#GenericPtr]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpGenericCastToPtrExplicit %[[#LocalPtr]] %[[#Arg]] Workgroup
define ptr addrspace(3) @test_to_local(ptr addrspace(4) noundef %ptr) {
entry:
  %cast = call spir_func noundef ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit.p3(ptr addrspace(4) noundef %ptr)
  ret ptr addrspace(3) %cast
}

; CHECK: OpFunction %[[#PrivatePtr]]
; CHECK-NEXT: %[[#Arg:]] = OpFunctionParameter %[[#GenericPtr]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpGenericCastToPtrExplicit %[[#PrivatePtr]] %[[#Arg]] Function
define ptr @test_to_private(ptr addrspace(4) noundef %ptr) {
entry:
  %cast = call spir_func noundef ptr @llvm.spv.generic.cast.to.ptr.explicit.p0(ptr addrspace(4) noundef %ptr)
  ret ptr %cast
}

declare noundef ptr @llvm.spv.generic.cast.to.ptr.explicit.p0(ptr addrspace(4))
declare noundef ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit.p1(ptr addrspace(4))
declare noundef ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit.p3(ptr addrspace(4))
