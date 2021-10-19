; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
; There is one lds global defined here, and this lds is used within a single non-kernel
; function, as an operand of nested constant expression, and this non-kernel function is
; reachable from kernel. Hence nested constant expression should to be converted into a
; series of instructons and pointer replacement should take place. But, important note
; is - only constant expression operands which uses lds should be converted into
; instructions, other constant expression operands which do not use lds should be left
; untouched.
;

; Original LDS should exist.
; CHECK: @lds_used_within_function = internal addrspace(3) global [4 x i32] undef, align 4
@lds_used_within_function = internal addrspace(3) global [4 x i32] undef, align 4

; Non-LDS global should exist as it is.
; CHECK: @global_var = internal addrspace(1) global [4 x i32] undef, align 4
@global_var = internal addrspace(1) global [4 x i32] undef, align 4

; Pointer should be created.
; CHECK: @lds_used_within_function.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2

; Pointer replacement code should be added.
define internal void @function() {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, i16 addrspace(3)* @lds_used_within_function.ptr, align 2
; CHECK:   %1 = getelementptr i8, i8 addrspace(3)* null, i16 %0
; CHECK:   %2 = bitcast i8 addrspace(3)* %1 to [4 x i32] addrspace(3)*
; CHECK:   %3 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* %2, i32 0, i32 2
; CHECK:   %4 = addrspacecast i32 addrspace(3)* %3 to i32*
; CHECK:   %5 = ptrtoint i32* %4 to i32
; CHECK:   %6 = add i32 %5, ptrtoint (i32 addrspace(1)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(1)* @global_var, i32 0, i32 2) to i32)
; CHECK:   ret void
entry:
  %0 = add i32 ptrtoint (i32* addrspacecast (i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @lds_used_within_function, i32 0, i32 2) to i32*) to i32), ptrtoint (i32 addrspace(1)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(1)* @global_var, i32 0, i32 2) to i32)
  ret void
}

; Pointer initialization code shoud be added
define protected amdgpu_kernel void @kernel() {
; CHECK-LABEL: entry:
; CHECK:   %0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   br i1 %1, label %2, label %3
;
; CHECK-LABEL: 2:
; CHECK:   store i16 ptrtoint ([4 x i32] addrspace(3)* @lds_used_within_function to i16), i16 addrspace(3)* @lds_used_within_function.ptr, align 2
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
