; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; byval/byref/sret arguments must keep a typed OpTypePointer even with the
; untyped pointers extension on, so the aggregate layout survives in the pointee
; type. An untyped pointer here loses the type and breaks the argument ABI.

; CHECK: OpCapability UntypedPointersKHR
; CHECK: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#F32]] %[[#F32]] %[[#F32]]
; The by-value struct argument keeps a typed pointer to the struct...
; CHECK-DAG: %[[#STRUCT_PTR:]] = OpTypePointer Function %[[#STRUCT]]
; ...while ordinary pointer arguments become untyped.
; CHECK-DAG: %[[#CROSS_PTR:]] = OpTypeUntypedPointerKHR CrossWorkgroup

%struct.Params = type { float, float, float }

; CHECK: OpFunction
; CHECK: OpFunctionParameter %[[#STRUCT_PTR]]
; CHECK: OpFunctionParameter %[[#CROSS_PTR]]
define spir_kernel void @test_byval(ptr byval(%struct.Params) %params,
                                    ptr addrspace(1) %out) {
entry:
  %f1.addr = getelementptr inbounds %struct.Params, ptr %params, i32 0, i32 1
  %f1 = load float, ptr %f1.addr, align 4
  store float %f1, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: OpFunction
; CHECK: OpFunctionParameter %[[#STRUCT_PTR]]
define spir_kernel void @test_byref(ptr byref(%struct.Params) %params,
                                    ptr addrspace(1) %out) {
entry:
  %f2.addr = getelementptr inbounds %struct.Params, ptr %params, i32 0, i32 2
  %f2 = load float, ptr %f2.addr, align 4
  store float %f2, ptr addrspace(1) %out, align 4
  ret void
}
