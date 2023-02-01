; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; There are three lds globals defined here, and these three lds are used within a single
; non-kernel function, and this non-kernel function is reachable from kernel. Hence pointer
; replacement is required for all three lds globals.
;

; Original LDS should exist.
; CHECK: @lds1 = internal addrspace(3) global [1 x i32] undef, align 4
; CHECK: @lds2 = internal addrspace(3) global [2 x i32] undef, align 4
; CHECK: @lds3 = internal addrspace(3) global [3 x i32] undef, align 4
@lds1 = internal addrspace(3) global [1 x i32] undef, align 4
@lds2 = internal addrspace(3) global [2 x i32] undef, align 4
@lds3 = internal addrspace(3) global [3 x i32] undef, align 4

; Pointers should be created.
; CHECK: @lds1.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; CHECK: @lds2.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; CHECK: @lds3.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2

; Pointer replacement code should be added.
define internal void @function() {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, ptr addrspace(3) @lds3.ptr, align 2
; CHECK:   %1 = getelementptr i8, ptr addrspace(3) null, i16 %0
; CHECK:   %2 = load i16, ptr addrspace(3) @lds2.ptr, align 2
; CHECK:   %3 = getelementptr i8, ptr addrspace(3) null, i16 %2
; CHECK:   %4 = load i16, ptr addrspace(3) @lds1.ptr, align 2
; CHECK:   %5 = getelementptr i8, ptr addrspace(3) null, i16 %4
; CHECK:   %gep1 = getelementptr inbounds [1 x i32], ptr addrspace(3) %5, i32 0, i32 0
; CHECK:   %gep2 = getelementptr inbounds [2 x i32], ptr addrspace(3) %3, i32 0, i32 0
; CHECK:   %gep3 = getelementptr inbounds [3 x i32], ptr addrspace(3) %1, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep1 = getelementptr inbounds [1 x i32], ptr addrspace(3) @lds1, i32 0, i32 0
  %gep2 = getelementptr inbounds [2 x i32], ptr addrspace(3) @lds2, i32 0, i32 0
  %gep3 = getelementptr inbounds [3 x i32], ptr addrspace(3) @lds3, i32 0, i32 0
  ret void
}

; Pointer initialization code shoud be added;
define protected amdgpu_kernel void @kernel() {
; CHECK-LABEL: entry:
; CHECK:   %0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   br i1 %1, label %2, label %3
;
; CHECK-LABEL: 2:
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds3 to i16), ptr addrspace(3) @lds3.ptr, align 2
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds2 to i16), ptr addrspace(3) @lds2.ptr, align 2
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds1 to i16), ptr addrspace(3) @lds1.ptr, align 2
; CHECK:   br label %3
;
; CHECK-LABEL: 3:
; CHECK:   call void @llvm.amdgcn.wave.barrier()
; CHECK:   call void @function()
; CHECK:   ret void
entry:
  call void @function()
  ret void
}
