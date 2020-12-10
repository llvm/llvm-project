; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces -o - %s | FileCheck %s

@c0 = addrspace(4) global float* undef

; CHECK-LABEL: @generic_ptr_from_constant
; CHECK: addrspacecast float* %p to float addrspace(1)*
; CHECK-NEXT: load float, float addrspace(1)*
define float @generic_ptr_from_constant() {
  %p = load float*, float* addrspace(4)* @c0
  %v = load float, float* %p
  ret float %v
}

%struct.S = type { i32*, float* }

; CHECK-LABEL: @generic_ptr_from_aggregate_argument
; CHECK: addrspacecast i32* %p0 to i32 addrspace(1)*
; CHECK: addrspacecast float* %p1 to float addrspace(1)*
; CHECK: load i32, i32 addrspace(1)*
; CHECK: store float %v1, float addrspace(1)*
; CHECK: ret
define amdgpu_kernel void @generic_ptr_from_aggregate_argument(%struct.S addrspace(4)* byref(%struct.S) align 8 %0) {
  %f0 = getelementptr inbounds %struct.S, %struct.S addrspace(4)* %0, i64 0, i32 0
  %p0 = load i32*, i32* addrspace(4)* %f0
  %f1 = getelementptr inbounds %struct.S, %struct.S addrspace(4)* %0, i64 0, i32 1
  %p1 = load float*, float* addrspace(4)* %f1
  %v0 = load i32, i32* %p0
  %v1 = sitofp i32 %v0 to float
  store float %v1, float* %p1
  ret void
}
