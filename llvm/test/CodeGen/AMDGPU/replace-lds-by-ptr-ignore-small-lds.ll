; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION ;
;
; LDS global @small_lds is used within non-kernel function @f0, and @f0 is reachable
; from kernel @k0, but since @small_lds too small for pointer replacement, pointer
; replacement does not take place.
;

; CHECK: @small_lds = addrspace(3) global i8 undef, align 1
@small_lds = addrspace(3) global i8 undef, align 1

; CHECK-NOT: @small_lds.ptr

define void @f0() {
; CHECK-LABEL: entry:
; CHECK:   store i8 1, i8 addrspace(3)* @small_lds, align 1
; CHECK:   ret void
entry:
  store i8 1, i8 addrspace(3)* @small_lds, align 1
  ret void
}

define amdgpu_kernel void @k0() {
; CHECK-LABEL: entry:
; CHECK:   call void @f0()
; CHECK:   ret void
entry:
  call void @f0()
  ret void
}
