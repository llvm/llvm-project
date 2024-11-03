; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; There are three lds globals defined here, and these three lds are used respectively within
; three non-kernel functions. There are three kernels, which call two of the non-kernel functions.
; Hence pointer replacement should take place for all three lds, and pointer initialization within
; kernel should selectively happen depending on which lds is reachable from the kernel.
;

; Original LDS should exist.
; CHECK: @lds_used_within_function_1 = internal addrspace(3) global [1 x i32] undef, align 4
; CHECK: @lds_used_within_function_2 = internal addrspace(3) global [2 x i32] undef, align 4
; CHECK: @lds_used_within_function_3 = internal addrspace(3) global [3 x i32] undef, align 4
@lds_used_within_function_1 = internal addrspace(3) global [1 x i32] undef, align 4
@lds_used_within_function_2 = internal addrspace(3) global [2 x i32] undef, align 4
@lds_used_within_function_3 = internal addrspace(3) global [3 x i32] undef, align 4

; Pointers should be created.
; CHECK: @lds_used_within_function_1.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; CHECK: @lds_used_within_function_2.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; CHECK: @lds_used_within_function_3.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2

; Pointer replacement code should be added.
define internal void @function_3() {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, ptr addrspace(3) @lds_used_within_function_3.ptr, align 2
; CHECK:   %1 = getelementptr i8, ptr addrspace(3) null, i16 %0
; CHECK:   %gep = getelementptr inbounds [3 x i32], ptr addrspace(3) %1, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [3 x i32], ptr addrspace(3) @lds_used_within_function_3, i32 0, i32 0
  ret void
}

; Pointer replacement code should be added.
define internal void @function_2() {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, ptr addrspace(3) @lds_used_within_function_2.ptr, align 2
; CHECK:   %1 = getelementptr i8, ptr addrspace(3) null, i16 %0
; CHECK:   %gep = getelementptr inbounds [2 x i32], ptr addrspace(3) %1, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [2 x i32], ptr addrspace(3) @lds_used_within_function_2, i32 0, i32 0
  ret void
}

; Pointer replacement code should be added.
define internal void @function_1() {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, ptr addrspace(3) @lds_used_within_function_1.ptr, align 2
; CHECK:   %1 = getelementptr i8, ptr addrspace(3) null, i16 %0
; CHECK:   %gep = getelementptr inbounds [1 x i32], ptr addrspace(3) %1, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [1 x i32], ptr addrspace(3) @lds_used_within_function_1, i32 0, i32 0
  ret void
}

; Pointer initialization code shoud be added
define protected amdgpu_kernel void @kernel_calls_function_3_and_1() {
; CHECK-LABEL: entry:
; CHECK:   %0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   br i1 %1, label %2, label %3
;
; CHECK-LABEL: 2:
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds_used_within_function_3 to i16), ptr addrspace(3) @lds_used_within_function_3.ptr, align 2
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds_used_within_function_1 to i16), ptr addrspace(3) @lds_used_within_function_1.ptr, align 2
; CHECK:   br label %3
;
; CHECK-LABEL: 3:
; CHECK:   call void @llvm.amdgcn.wave.barrier()
; CHECK:   call void @function_3()
; CHECK:   call void @function_1()
; CHECK:   ret void
entry:
  call void @function_3()
  call void @function_1()
  ret void
}

; Pointer initialization code shoud be added
define protected amdgpu_kernel void @kernel_calls_function_2_and_3() {
; CHECK-LABEL: entry:
; CHECK:   %0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   br i1 %1, label %2, label %3
;
; CHECK-LABEL: 2:
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds_used_within_function_3 to i16), ptr addrspace(3) @lds_used_within_function_3.ptr, align 2
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds_used_within_function_2 to i16), ptr addrspace(3) @lds_used_within_function_2.ptr, align 2
; CHECK:   br label %3
;
; CHECK-LABEL: 3:
; CHECK:   call void @llvm.amdgcn.wave.barrier()
; CHECK:   call void @function_2()
; CHECK:   call void @function_3()
; CHECK:   ret void
entry:
  call void @function_2()
  call void @function_3()
  ret void
}

; Pointer initialization code shoud be added
define protected amdgpu_kernel void @kernel_calls_function_1_and_2() {
; CHECK-LABEL: entry:
; CHECK:   %0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   br i1 %1, label %2, label %3
;
; CHECK-LABEL: 2:
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds_used_within_function_2 to i16), ptr addrspace(3) @lds_used_within_function_2.ptr, align 2
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds_used_within_function_1 to i16), ptr addrspace(3) @lds_used_within_function_1.ptr, align 2
; CHECK:   br label %3
;
; CHECK-LABEL: 3:
; CHECK:   call void @llvm.amdgcn.wave.barrier()
; CHECK:   call void @function_1()
; CHECK:   call void @function_2()
; CHECK:   ret void
entry:
  call void @function_1()
  call void @function_2()
  ret void
}
