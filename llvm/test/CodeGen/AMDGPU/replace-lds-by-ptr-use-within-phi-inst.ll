; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; Replace lds globals used within phi instruction.
;

; Original LDS should exist.
; CHECK: @lds.1 = addrspace(3) global i32 undef, align 4
; CHECK: @lds.2 = addrspace(3) global i32 undef, align 4
@lds.1 = addrspace(3) global i32 undef, align 4
@lds.2 = addrspace(3) global i32 undef, align 4

; Pointers should be created.
; CHECK: @lds.1.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
; CHECK: @lds.2.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2

define void @f0(i32 %arg) {
; CHECK-LABEL: bb:
; CHECK:   %0 = load i16, ptr addrspace(3) @lds.2.ptr, align 2
; CHECK:   %1 = getelementptr i8, ptr addrspace(3) null, i16 %0
; CHECK:   %2 = load i16, ptr addrspace(3) @lds.1.ptr, align 2
; CHECK:   %3 = getelementptr i8, ptr addrspace(3) null, i16 %2
; CHECK:   %id = call i32 @llvm.amdgcn.workitem.id.x()
; CHECK:   %my.tmp = sub i32 %id, %arg
; CHECK:   br label %bb1
bb:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %my.tmp = sub i32 %id, %arg
  br label %bb1

; CHECK-LABEL: bb1:
; CHECK:   %lsr.iv = phi i32 [ undef, %bb ], [ %my.tmp2, %Flow ]
; CHECK:   %4 = icmp ne ptr addrspace(3) inttoptr (i32 4 to ptr addrspace(3)), %3
; CHECK:   %lsr.iv.next = add i32 %lsr.iv, 1
; CHECK:   %cmp0 = icmp slt i32 %lsr.iv.next, 0
; CHECK:   br i1 %cmp0, label %bb4, label %Flow
bb1:
  %lsr.iv = phi i32 [ undef, %bb ], [ %my.tmp2, %Flow ]
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp0 = icmp slt i32 %lsr.iv.next, 0
  br i1 %cmp0, label %bb4, label %Flow

; CHECK-LABEL: bb4:
; CHECK:   %load = load volatile i32, ptr addrspace(1) undef, align 4
; CHECK:   %cmp1 = icmp sge i32 %my.tmp, %load
; CHECK:   br label %Flow
bb4:
  %load = load volatile i32, ptr addrspace(1) undef, align 4
  %cmp1 = icmp sge i32 %my.tmp, %load
  br label %Flow

; CHECK-LABEL: Flow:
; CHECK:   %my.tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
; CHECK:   %my.tmp3 = phi ptr addrspace(3) [ %1, %bb4 ], [ %3, %bb1 ]
; CHECK:   %my.tmp4 = phi i1 [ %cmp1, %bb4 ], [ %4, %bb1 ]
; CHECK:   br i1 %my.tmp4, label %bb9, label %bb1
Flow:
  %my.tmp2 = phi i32 [ %lsr.iv.next, %bb4 ], [ undef, %bb1 ]
  %my.tmp3 = phi ptr addrspace(3) [@lds.2, %bb4 ], [ @lds.1, %bb1 ]
  %my.tmp4 = phi i1 [ %cmp1, %bb4 ], [ icmp ne (ptr addrspace(3) inttoptr (i32 4 to ptr addrspace(3)), ptr addrspace(3) @lds.1), %bb1 ]
  br i1 %my.tmp4, label %bb9, label %bb1

; CHECK-LABEL: bb9:
; CHECK:   store volatile i32 7, ptr addrspace(3) undef, align 4
; CHECK:   ret void
bb9:
  store volatile i32 7, ptr addrspace(3) undef
  ret void
}

; CHECK-LABEL: @k0
; CHECK:   %1 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
; CHECK:   %2 = icmp eq i32 %1, 0
; CHECK:   br i1 %2, label %3, label %4
;
; CHECK-LABEL: 3:
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds.2 to i16), ptr addrspace(3) @lds.2.ptr, align 2
; CHECK:   store i16 ptrtoint (ptr addrspace(3) @lds.1 to i16), ptr addrspace(3) @lds.1.ptr, align 2
; CHECK:   br label %4
;
; CHECK-LABEL: 4:
; CHECK:   call void @llvm.amdgcn.wave.barrier()
; CHECK:   call void @f0(i32 %arg)
; CHECK:   ret void
define amdgpu_kernel void @k0(i32 %arg) {
  call void @f0(i32 %arg)
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
