; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; The lds global @lds_used_within_func is used within non-kernel function @func_uses_lds
; which is recheable from kernel @kernel_reaches_lds, hence pointer replacement takes place
; for @lds_used_within_func.
;

; Original LDS should exist.
; CHECK: @lds_used_within_func = internal addrspace(3) global [4 x i32] undef, align 4
@lds_used_within_func = internal addrspace(3) global [4 x i32] undef, align 4

; Pointer should be created.
; CHECK: @lds_used_within_func.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2

; Pointer replacement code should be added.
define internal void @func_uses_lds() {
; CHECK-LABEL: entry:
; CHECK:   %0 = load i16, i16 addrspace(3)* @lds_used_within_func.ptr, align 2
; CHECK:   %1 = getelementptr i8, i8 addrspace(3)* null, i16 %0
; CHECK:   %2 = bitcast i8 addrspace(3)* %1 to [4 x i32] addrspace(3)*
; CHECK:   %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* %2, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @lds_used_within_func, i32 0, i32 0
  ret void
}

; No change
define internal void @func_does_not_use_lds_3() {
; CHECK-LABEL: entry:
; CHECK:   call void @func_uses_lds()
; CHECK:   ret void
entry:
  call void @func_uses_lds()
  ret void
}

; No change
define internal void @func_does_not_use_lds_2() {
; CHECK-LABEL: entry:
; CHECK:   call void @func_uses_lds()
; CHECK:   ret void
entry:
  call void @func_uses_lds()
  ret void
}

; No change
define internal void @func_does_not_use_lds_1() {
; CHECK-LABEL: entry:
; CHECK:   call void @func_does_not_use_lds_2()
; CHECK:   call void @func_does_not_use_lds_3()
; CHECK:   ret void
entry:
  call void @func_does_not_use_lds_2()
  call void @func_does_not_use_lds_3()
  ret void
}

; Pointer initialization code shoud be added
define protected amdgpu_kernel void @kernel_reaches_lds() {
; CHECK-LABEL: entry:
; CHECK:   %0 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   br i1 %1, label %2, label %3
;
; CHECK-LABEL: 2:
; CHECK:   store i16 ptrtoint ([4 x i32] addrspace(3)* @lds_used_within_func to i16), i16 addrspace(3)* @lds_used_within_func.ptr, align 2
; CHECK:   br label %3
;
; CHECK-LABEL: 3:
; CHECK:   call void @llvm.amdgcn.wave.barrier()
; CHECK:   call void @func_does_not_use_lds_1()
; CHECK:   ret void
entry:
  call void @func_does_not_use_lds_1()
  ret void
}

; No change here since this kernel does not reach @func_uses_lds which uses lds.
define protected amdgpu_kernel void @kernel_does_not_reach_lds() {
; CHECK-LABEL: entry:
; CHECK:   ret void
entry:
  ret void
}
