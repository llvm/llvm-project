; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION ;
;
; LDS global @used_only_within_kern is used only within kernel @k0, hence pointer replacement
; does not take place for @used_only_within_kern.
;

; CHECK: @used_only_within_kern = addrspace(3) global [4 x i32] undef, align 4
@used_only_within_kern = addrspace(3) global [4 x i32] undef, align 4

; CHECK-NOT: @used_only_within_kern.ptr

define amdgpu_kernel void @k0() {
; CHECK-LABEL: entry:
; CHECK:   %ld = load i32, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64)) to ptr), align 4
; CHECK:   %mul = mul i32 %ld, 2
; CHECK:   store i32 %mul, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64)) to ptr), align 4
; CHECK:   ret void
entry:
  %ld = load i32, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64)) to ptr), align 4
  %mul = mul i32 %ld, 2
  store i32 %mul, ptr inttoptr (i64 add (i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64), i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @used_only_within_kern to ptr) to i64)) to ptr), align 4
  ret void
}
