; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces -o - %s | FileCheck %s

@c0 = addrspace(4) global ptr undef

; CHECK-LABEL: @generic_ptr_from_constant
; CHECK: addrspacecast ptr %p to ptr addrspace(1)
; CHECK-NEXT: load float, ptr addrspace(1)
define float @generic_ptr_from_constant() {
  %p = load ptr, ptr addrspace(4) @c0
  %v = load float, ptr %p
  ret float %v
}

%struct.S = type { ptr, ptr }

; CHECK-LABEL: @generic_ptr_from_aggregate_argument
; CHECK: addrspacecast ptr %p0 to ptr addrspace(1)
; CHECK: addrspacecast ptr %p1 to ptr addrspace(1)
; CHECK: load i32, ptr addrspace(1)
; CHECK: store float %v1, ptr addrspace(1)
; CHECK: ret
define amdgpu_kernel void @generic_ptr_from_aggregate_argument(ptr addrspace(4) byref(%struct.S) align 8 %0) {
  %p0 = load ptr, ptr addrspace(4) %0
  %f1 = getelementptr inbounds %struct.S, ptr addrspace(4) %0, i64 0, i32 1
  %p1 = load ptr, ptr addrspace(4) %f1
  %v0 = load i32, ptr %p0
  %v1 = sitofp i32 %v0 to float
  store float %v1, ptr %p1
  ret void
}
