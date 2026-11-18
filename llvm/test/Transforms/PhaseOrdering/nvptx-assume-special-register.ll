; RUN: opt -passes='default<O2>' -S -mtriple=nvptx64-nvidia-cuda < %s | FileCheck %s
; REQUIRES: nvptx-registered-target

; Check that assumptions about CUDA built-in variables constrain subsequent
; reads of the corresponding NVVM special registers. The control functions
; show that the result cannot be folded without the assumptions.

target triple = "nvptx64-nvidia-cuda"

declare void @llvm.assume(i1 noundef)
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()

define i32 @assume_thread_idx_x_eq() {
; CHECK-LABEL: @assume_thread_idx_x_eq(
; CHECK-NEXT:    ret i32 8
;
  %tid.assume = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %condition = icmp eq i32 %tid.assume, 7
  call void @llvm.assume(i1 %condition)
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %result = add i32 %tid, 1
  ret i32 %result
}

define i32 @no_assume_thread_idx_x_eq() {
; CHECK-LABEL: @no_assume_thread_idx_x_eq(
; CHECK-NEXT:    [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NEXT:    [[RESULT:%.*]] = add nuw nsw i32 [[TID]], 1
; CHECK-NEXT:    ret i32 [[RESULT]]
;
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %result = add i32 %tid, 1
  ret i32 %result
}

define i32 @assume_block_idx_x_eq() {
; CHECK-LABEL: @assume_block_idx_x_eq(
; CHECK-NEXT:    ret i32 8
;
  %ctaid.assume = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %condition = icmp eq i32 %ctaid.assume, 7
  call void @llvm.assume(i1 %condition)
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %result = add i32 %ctaid, 1
  ret i32 %result
}

define i32 @no_assume_block_idx_x_eq() {
; CHECK-LABEL: @no_assume_block_idx_x_eq(
; CHECK-NEXT:    [[CTAID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK-NEXT:    [[RESULT:%.*]] = add nuw nsw i32 [[CTAID]], 1
; CHECK-NEXT:    ret i32 [[RESULT]]
;
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %result = add i32 %ctaid, 1
  ret i32 %result
}

define i1 @assume_block_dim_x_range() {
; CHECK-LABEL: @assume_block_dim_x_range(
; CHECK-NEXT:    ret i1 true
;
  %ntid.assume = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %condition = icmp ule i32 %ntid.assume, 32
  call void @llvm.assume(i1 %condition)
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %result = icmp ule i32 %ntid, 64
  ret i1 %result
}

define i1 @no_assume_block_dim_x_range() {
; CHECK-LABEL: @no_assume_block_dim_x_range(
; CHECK-NEXT:    [[NTID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
; CHECK-NEXT:    [[RESULT:%.*]] = icmp samesign ult i32 [[NTID]], 65
; CHECK-NEXT:    ret i1 [[RESULT]]
;
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %result = icmp ule i32 %ntid, 64
  ret i1 %result
}

define i1 @assume_grid_dim_x_range() {
; CHECK-LABEL: @assume_grid_dim_x_range(
; CHECK-NEXT:    ret i1 true
;
  %nctaid.assume = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %condition = icmp ule i32 %nctaid.assume, 32
  call void @llvm.assume(i1 %condition)
  %nctaid = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %result = icmp ule i32 %nctaid, 64
  ret i1 %result
}

define i1 @no_assume_grid_dim_x_range() {
; CHECK-LABEL: @no_assume_grid_dim_x_range(
; CHECK-NEXT:    [[NCTAID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
; CHECK-NEXT:    [[RESULT:%.*]] = icmp samesign ult i32 [[NCTAID]], 65
; CHECK-NEXT:    ret i1 [[RESULT]]
;
  %nctaid = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %result = icmp ule i32 %nctaid, 64
  ret i1 %result
}
