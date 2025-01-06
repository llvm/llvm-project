; The goal of the test is to ensure that type inference doesn't break validity of the generated SPIR-V code.
; The only pass criterion is that spirv-val considers output valid.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Int:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrChar:]] = OpTypePointer Function %[[#Char]]
; CHECK-DAG: %[[#PtrCharCW:]] = OpTypePointer CrossWorkgroup %[[#Char]]
; CHECK-DAG: %[[#PtrCharGen:]] = OpTypePointer Generic %[[#Char]]
; CHECK-DAG: %[[#Struct:]] = OpTypeStruct %[[#]] %[[#]] %[[#]]
; CHECK-DAG: %[[#PtrInt:]] = OpTypePointer Function %[[#Int]]
; CHECK-DAG: %[[#PtrPtrCharGen:]] = OpTypePointer Function %[[#PtrCharGen]]
; CHECK-DAG: %[[#PtrStruct:]] = OpTypePointer Function %[[#Struct]]
; CHECK: OpFunction
; CHECK: %[[#Arg1:]] = OpFunctionParameter %[[#Int]]
; CHECK: %[[#Arg2:]] = OpFunctionParameter %[[#PtrCharCW]]
; CHECK: %[[#Kernel:]] = OpVariable %[[#PtrStruct]] Function
; CHECK: %[[#IntKernel:]] = OpBitcast %[[#PtrInt]] %[[#Kernel]]
; CHECK: OpStore %[[#IntKernel]] %[[#Arg1]]
; CHECK: %[[#CharKernel:]] = OpBitcast %[[#PtrChar]] %[[#Kernel]]
; CHECK: %[[#P:]] = OpInBoundsPtrAccessChain %[[#PtrChar]] %[[#CharKernel]] %[[#]]
; CHECK: %[[#R0:]] = OpPtrCastToGeneric %[[#PtrCharGen]] %[[#Arg2]]
; CHECK: %[[#P2:]] = OpBitcast %[[#PtrPtrCharGen]] %[[#P]]
; CHECK: OpStore %[[#P2]] %[[#R0]]
; CHECK: %[[#P3:]] = OpBitcast %[[#PtrPtrCharGen]] %[[#P]]
; CHECK: %[[#]] = OpLoad %[[#PtrCharGen]] %[[#P3]]

%"class.std::complex" = type { { double, double } }
%class.anon = type { i32, ptr addrspace(4), [2 x [2 x %"class.std::complex"]] }

define weak_odr dso_local spir_kernel void @foo(i32 noundef %_arg_N, ptr addrspace(1) noundef align 8 %_arg_p) {
entry:
  %Kernel = alloca %class.anon, align 8
  store i32 %_arg_N, ptr %Kernel, align 8
  %p = getelementptr inbounds i8, ptr %Kernel, i64 8
  %r0 = addrspacecast ptr addrspace(1) %_arg_p to ptr addrspace(4)
  store ptr addrspace(4) %r0, ptr %p, align 8
  %r3 = load ptr addrspace(4), ptr %p, align 8
  ret void
}
