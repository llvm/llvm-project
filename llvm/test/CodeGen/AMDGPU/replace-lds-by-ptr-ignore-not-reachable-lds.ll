; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION ;
;
; LDS global @not-reachable-lds is used within non-kernel function @f0, but @f0 is *not*
; reachable from kernel @k, hence pointer replacement does not take place.
;

; CHECK: @not-reachable-lds = internal addrspace(3) global [4 x i32] undef, align 4
@not-reachable-lds = internal addrspace(3) global [4 x i32] undef, align 4

; CHECK-NOT: @not-reachable-lds.ptr

define internal void @f0() {
; CHECK-LABEL: entry:
; CHECK:   %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @not-reachable-lds, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @not-reachable-lds, i32 0, i32 0
  ret void
}

define protected amdgpu_kernel void @k0() {
; CHECK-LABEL: entry:
; CHECK:   ret void
entry:
  ret void
}
