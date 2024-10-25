; RUN: opt -S -mtriple=spirv64-amd-amdhsa -passes=infer-address-spaces -o - %s | FileCheck %s

@c0 = addrspace(2) global ptr undef

; CHECK-LABEL: @generic_ptr_from_constant
; CHECK: addrspacecast ptr addrspace(4) %p to ptr addrspace(1)
; CHECK-NEXT: load float, ptr addrspace(1)
define spir_func float @generic_ptr_from_constant() {
  %p = load ptr addrspace(4), ptr addrspace(2) @c0
  %v = load float, ptr addrspace(4) %p
  ret float %v
}

%struct.S = type { ptr addrspace(4), ptr addrspace(4) }

; CHECK-LABEL: @generic_ptr_from_aggregate_argument
; CHECK: addrspacecast ptr addrspace(4) %p0 to ptr addrspace(1)
; CHECK: addrspacecast ptr addrspace(4) %p1 to ptr addrspace(1)
; CHECK: load i32, ptr addrspace(1)
; CHECK: store float %v1, ptr addrspace(1)
; CHECK: ret
define spir_kernel void @generic_ptr_from_aggregate_argument(ptr addrspace(2) byval(%struct.S) align 8 %0) {
  %p0 = load ptr addrspace(4), ptr addrspace(2) %0
  %f1 = getelementptr inbounds %struct.S, ptr addrspace(2) %0, i64 0, i32 1
  %p1 = load ptr addrspace(4), ptr addrspace(2) %f1
  %v0 = load i32, ptr addrspace(4) %p0
  %v1 = sitofp i32 %v0 to float
  store float %v1, ptr addrspace(4) %p1
  ret void
}
