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
; CHECK:   %ld = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @used_only_within_kern, i32 0, i32 0) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @used_only_within_kern, i32 0, i32 0) to i32*) to i64)) to i32*), align 4
; CHECK:   %mul = mul i32 %ld, 2
; CHECK:   store i32 %mul, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @used_only_within_kern, i32 0, i32 0) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @used_only_within_kern, i32 0, i32 0) to i32*) to i64)) to i32*), align 4
; CHECK:   ret void
entry:
  %ld = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast ([4 x i32] addrspace(3)* @used_only_within_kern to i32 addrspace(3)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast ([4 x i32] addrspace(3)* @used_only_within_kern to i32 addrspace(3)*) to i32*) to i64)) to i32*), align 4
  %mul = mul i32 %ld, 2
  store i32 %mul, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast ([4 x i32] addrspace(3)* @used_only_within_kern to i32 addrspace(3)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(3)* bitcast ([4 x i32] addrspace(3)* @used_only_within_kern to i32 addrspace(3)*) to i32*) to i64)) to i32*), align 4
  ret void
}
