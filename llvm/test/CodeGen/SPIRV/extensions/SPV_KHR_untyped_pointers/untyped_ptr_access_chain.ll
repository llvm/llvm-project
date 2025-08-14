; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; XFAIL: *

; CHECK-SPIRV: OpCapability UntypedPointersKHR
; CHECK-SPIRV: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-SPIRV: OpTypeInt [[#IntTy:]] 32
; CHECK-SPIRV: OpConstant [[#IntTy]] [[#Const0:]] 0
; CHECK-SPIRV: OpConstant [[#IntTy]] [[#Const1:]] 1
; CHECK-SPIRV: OpTypeUntypedPointerKHR [[#UntypedPtrTy:]] 7
; CHECK-SPIRV: OpTypeStruct [[#StructTy:]] [[#IntTy]] [[#IntTy]]
; CHECK-SPIRV: OpUntypedVariableKHR [[#UntypedPtrTy]] [[#StructVarId:]] 7 [[#StructTy]]
; CHECK-SPIRV: OpUntypedInBoundsPtrAccessChainKHR [[#UntypedPtrTy]] [[#PtrAccessId:]] [[#StructTy]] [[#StructVarId]] [[#Const0]] [[#Const1]]

%struct.Example = type { i32, i32 }

define spir_func void @test(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca %struct.Example, align 4
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = getelementptr inbounds nuw %struct.Example, ptr %3, i32 0, i32 1
  store i32 %4, ptr %5, align 4
  ret void
}
