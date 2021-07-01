; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; There are three lds globals defined here, and these three lds are used respectively within
; three non-kernel functions. There is one kernel which *indirectly* calls one of the non-kernel
; functions. But since all the three non-kernel functions have same signature, all three
; non-kernel functions are resolved as potential callees for indirect call-site. Hence we land-up
; pointer replacement for three lds globals.
;

; Original LDS should exist.
; CHECK: @lds_used_within_function_1 = internal addrspace(3) global [4 x i32] undef, align 4
; CHECK: @lds_used_within_function_2 = internal addrspace(3) global [4 x i32] undef, align 4
; CHECK: @lds_used_within_function_3 = internal addrspace(3) global [4 x i32] undef, align 4
@lds_used_within_function_1 = internal addrspace(3) global [4 x i32] undef, align 4
@lds_used_within_function_2 = internal addrspace(3) global [4 x i32] undef, align 4
@lds_used_within_function_3 = internal addrspace(3) global [4 x i32] undef, align 4

; Function pointers should exist.
; CHECK: @ptr_to_func1 = internal local_unnamed_addr externally_initialized global void (i16)* @function_1, align 8
; CHECK: @ptr_to_func2 = internal local_unnamed_addr externally_initialized global void (i16)* @function_2, align 8
; CHECK: @ptr_to_func3 = internal local_unnamed_addr externally_initialized global void (i16)* @function_3, align 8
@ptr_to_func1 = internal local_unnamed_addr externally_initialized global void (i16)* @function_1, align 8
@ptr_to_func2 = internal local_unnamed_addr externally_initialized global void (i16)* @function_2, align 8
@ptr_to_func3 = internal local_unnamed_addr externally_initialized global void (i16)* @function_3, align 8

; Pointers should be created.
; CHECK: @lds_used_within_function_1.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; CHECK: @lds_used_within_function_2.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; CHECK: @lds_used_within_function_3.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2

; Pointer replacement code should be added.
define internal void @function_3(i16 %i) {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, i16 addrspace(3)* @lds_used_within_function_3.ptr, align 2
; CHECK:   %1 = getelementptr i8, i8 addrspace(3)* null, i16 %0
; CHECK:   %2 = bitcast i8 addrspace(3)* %1 to [4 x i32] addrspace(3)*
; CHECK:   %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* %2, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @lds_used_within_function_3, i32 0, i32 0
  ret void
}

; Pointer replacement code should be added.
define internal void @function_2(i16 %i) {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, i16 addrspace(3)* @lds_used_within_function_2.ptr, align 2
; CHECK:   %1 = getelementptr i8, i8 addrspace(3)* null, i16 %0
; CHECK:   %2 = bitcast i8 addrspace(3)* %1 to [4 x i32] addrspace(3)*
; CHECK:   %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* %2, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @lds_used_within_function_2, i32 0, i32 0
  ret void
}

; Pointer replacement code should be added.
define internal void @function_1(i16 %i) {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, i16 addrspace(3)* @lds_used_within_function_1.ptr, align 2
; CHECK:   %1 = getelementptr i8, i8 addrspace(3)* null, i16 %0
; CHECK:   %2 = bitcast i8 addrspace(3)* %1 to [4 x i32] addrspace(3)*
; CHECK:   %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* %2, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @lds_used_within_function_1, i32 0, i32 0
  ret void
}

; Pointer initialization code shoud be added
define protected amdgpu_kernel void @kernel_indirectly_calls_function_1() {
; CHECK-LABEL: entry:
; CHECK:   %0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   br i1 %1, label %2, label %3
;
; CHECK-LABEL: 2:
; CHECK:   store i16 ptrtoint ([4 x i32] addrspace(3)* @lds_used_within_function_3 to i16), i16 addrspace(3)* @lds_used_within_function_3.ptr, align 2
; CHECK:   store i16 ptrtoint ([4 x i32] addrspace(3)* @lds_used_within_function_2 to i16), i16 addrspace(3)* @lds_used_within_function_2.ptr, align 2
; CHECK:   store i16 ptrtoint ([4 x i32] addrspace(3)* @lds_used_within_function_1 to i16), i16 addrspace(3)* @lds_used_within_function_1.ptr, align 2
; CHECK:   br label %3
;
; CHECK-LABEL: 3:
; CHECK:   call void @llvm.amdgcn.wave.barrier()
; CHECK:   %fptr1 = load void (i16)*, void (i16)** @ptr_to_func1, align 8
; CHECK:   call void %fptr1(i16 6)
; CHECK:   ret void
entry:
  %fptr1 = load void (i16)*, void (i16)** @ptr_to_func1, align 8
  call void %fptr1(i16 6)
  ret void
}
