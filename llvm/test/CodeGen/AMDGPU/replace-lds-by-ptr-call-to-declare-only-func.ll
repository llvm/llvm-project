; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; The kernel 'kern' makes a call to declared only function `foo`, hence `foo`
; is not considered as reachable callee, and is ignored. The function `goo`
; which uses LDS is not called from kernel 'kern', hence it is also ignored.
;

; Original LDS should exist.
; CHECK: @lds = internal local_unnamed_addr addrspace(3) global i32 undef, align 4
@lds = internal local_unnamed_addr addrspace(3) global i32 undef, align 4

; Pointer should not be created.
; CHECK-NOT: @lds.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2

; CHECK: declare i32 @foo()
declare i32 @foo()

; No change
define internal void @goo() {
; CHECK-LABEL: entry:
; CHECK:   store i32 undef, i32 addrspace(3)* @lds, align 4
; CHECK:   ret void
entry:
  store i32 undef, i32 addrspace(3)* @lds, align 4
  ret void
}

; No change
define weak amdgpu_kernel void @kern() {
; CHECK-LABEL: entry:
; CHECK-LABEL:   %nt = call i32 @foo()
; CHECK-LABEL:   ret void
entry:
  %nt = call i32 @foo()
  ret void
}
