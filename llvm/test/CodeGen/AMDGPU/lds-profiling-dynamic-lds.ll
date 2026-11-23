; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s

; The profiling clone of a kernel with dynamic LDS uses the dynamic LDS of the
; kernel, and is passed its size like the kernel is.

@dynamic = external addrspace(3) global [0 x i32], align 4

; CHECK-LABEL: {{^}}k:
; CHECK-LABEL: {{^}}k.prof:
; The size of the allocation is not known, so no lane is filtered out.
; CHECK-NOT: v_cmp_gt_u32
; CHECK: v_readlane_b32
; CHECK: ds_store_b32

; CHECK: .value_kind: hidden_dynamic_lds_size
; CHECK: .name: k
; CHECK: .value_kind: hidden_dynamic_lds_size
; CHECK: .name: k.prof
define amdgpu_kernel void @k(i32 %i, i32 %v) {
  %p = getelementptr inbounds [0 x i32], ptr addrspace(3) @dynamic, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}
