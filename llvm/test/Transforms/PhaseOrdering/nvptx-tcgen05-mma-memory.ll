; RUN: opt -passes='dse,gvn' -S < %s | FileCheck %s
; REQUIRES: nvptx-registered-target

target triple = "nvptx64-nvidia-cuda"

@shared = addrspace(3) global [4096 x i32] undef, align 16
@global = addrspace(1) global i32 0

; The accumulator is read when enable_input_d is true.
define void @reads_accumulator(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @reads_accumulator(
; CHECK: store i32 1, ptr addrspace(6) %d
; CHECK: call void @llvm.nvvm.tcgen05.mma.shared(
; CHECK: store i32 2, ptr addrspace(6) %d
  store i32 1, ptr addrspace(6) %d
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 true, i32 0, i32 1, i32 0, i32 0)
  store i32 2, ptr addrspace(6) %d
  ret void
}

; The descriptors can refer to shared memory absent from pointer arguments.
define void @reads_shared_memory(ptr addrspace(6) noalias %d, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @reads_shared_memory(
; CHECK: store i32 1, ptr addrspace(3) @shared
; CHECK: call void @llvm.nvvm.tcgen05.mma.shared(
; CHECK: store i32 2, ptr addrspace(3) @shared
  store i32 1, ptr addrspace(3) @shared
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 false, i32 0, i32 1, i32 0, i32 0)
  store i32 2, ptr addrspace(3) @shared
  ret void
}

; .ashift changes A even when the destination does not alias it.
define i32 @ashift_writes_a(ptr addrspace(6) noalias %d, ptr addrspace(6) %a, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @ashift_writes_a(
; CHECK: %before = load i32, ptr addrspace(6) %a
; CHECK: call void @llvm.nvvm.tcgen05.mma.tensor.ashift(
; CHECK: %after = load i32, ptr addrspace(6) %a
; CHECK: %difference = sub i32 %after, %before
; CHECK: ret i32 %difference
  %before = load i32, ptr addrspace(6) %a
  call void @llvm.nvvm.tcgen05.mma.tensor.ashift(ptr addrspace(6) %d, ptr addrspace(6) %a, i64 %bdesc, i32 %idesc, i1 false, i32 0, i32 1, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %a
  %difference = sub i32 %after, %before
  ret i32 %difference
}

; A can be modified through the destination even when A itself is read-only.
define i32 @destination_aliases_a(ptr addrspace(6) %a, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @destination_aliases_a(
; CHECK: %before = load i32, ptr addrspace(6) %a
; CHECK: call void @llvm.nvvm.tcgen05.mma.tensor(
; CHECK: %after = load i32, ptr addrspace(6) %a
; CHECK: %difference = sub i32 %after, %before
; CHECK: ret i32 %difference
  %before = load i32, ptr addrspace(6) %a
  call void @llvm.nvvm.tcgen05.mma.tensor(ptr addrspace(6) %a, ptr addrspace(6) %a, i64 %bdesc, i32 %idesc, i1 true, i32 0, i32 1, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %a
  %difference = sub i32 %after, %before
  ret i32 %difference
}

; MMA does not write global or shared memory.
define i32 @preserves_other_memory(ptr addrspace(6) noalias %d, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @preserves_other_memory(
; CHECK: call void @llvm.nvvm.tcgen05.mma.shared(
; CHECK: ret i32 0
  %global.before = load i32, ptr addrspace(1) @global
  %shared.before = load i32, ptr addrspace(3) @shared
  call void @llvm.nvvm.tcgen05.mma.shared(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 true, i32 0, i32 1, i32 0, i32 0)
  %global.after = load i32, ptr addrspace(1) @global
  %shared.after = load i32, ptr addrspace(3) @shared
  %global.difference = sub i32 %global.after, %global.before
  %shared.difference = sub i32 %shared.after, %shared.before
  %difference = or i32 %global.difference, %shared.difference
  ret i32 %difference
}

; Aliasing the read-only inputs requires each argument to be read-only.
define i32 @preserves_a_and_sparse_metadata(ptr addrspace(6) noalias %d, ptr addrspace(6) %input, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @preserves_a_and_sparse_metadata(
; CHECK: call void @llvm.nvvm.tcgen05.mma.sp.tensor(
; CHECK: ret i32 0
  %before = load i32, ptr addrspace(6) %input
  call void @llvm.nvvm.tcgen05.mma.sp.tensor(ptr addrspace(6) %d, ptr addrspace(6) %input, i64 %bdesc, i32 %idesc, i1 true, ptr addrspace(6) %input, i32 0, i32 1, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %input
  %difference = sub i32 %after, %before
  ret i32 %difference
}

define i32 @disable_output_lane_preserves_a(ptr addrspace(6) noalias %d, ptr addrspace(6) %a, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @disable_output_lane_preserves_a(
; CHECK: call void @llvm.nvvm.tcgen05.mma.tensor.disable_output_lane.cg1(
; CHECK: ret i32 0
  %before = load i32, ptr addrspace(6) %a
  call void @llvm.nvvm.tcgen05.mma.tensor.disable_output_lane.cg1(ptr addrspace(6) %d, ptr addrspace(6) %a, i64 %bdesc, i32 %idesc, i1 true, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %a
  %difference = sub i32 %after, %before
  ret i32 %difference
}

define i32 @disable_output_lane_ashift_writes_a(ptr addrspace(6) noalias %d, ptr addrspace(6) %a, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @disable_output_lane_ashift_writes_a(
; CHECK: %before = load i32, ptr addrspace(6) %a
; CHECK: call void @llvm.nvvm.tcgen05.mma.tensor.disable_output_lane.cg1.ashift(
; CHECK: %after = load i32, ptr addrspace(6) %a
; CHECK: %difference = sub i32 %after, %before
; CHECK: ret i32 %difference
  %before = load i32, ptr addrspace(6) %a
  call void @llvm.nvvm.tcgen05.mma.tensor.disable_output_lane.cg1.ashift(ptr addrspace(6) %d, ptr addrspace(6) %a, i64 %bdesc, i32 %idesc, i1 true, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %a
  %difference = sub i32 %after, %before
  ret i32 %difference
}

define i32 @preserves_scales(ptr addrspace(6) noalias %d, ptr addrspace(6) %scales, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @preserves_scales(
; CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(
; CHECK: ret i32 0
  %before = load i32, ptr addrspace(6) %scales
  call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 true, ptr addrspace(6) %scales, ptr addrspace(6) %scales, i32 1, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %scales
  %difference = sub i32 %after, %before
  ret i32 %difference
}

; Sparse metadata shifts both scale arguments by one position.
define i32 @preserves_sparse_scales(ptr addrspace(6) noalias %d, ptr addrspace(6) %input, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @preserves_sparse_scales(
; CHECK: call void @llvm.nvvm.tcgen05.mma.sp.shared.mxf8f6f4.block_scale(
; CHECK: ret i32 0
  %before = load i32, ptr addrspace(6) %input
  call void @llvm.nvvm.tcgen05.mma.sp.shared.mxf8f6f4.block_scale(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 true, ptr addrspace(6) %input, ptr addrspace(6) %input, ptr addrspace(6) %input, i32 1, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %input
  %difference = sub i32 %after, %before
  ret i32 %difference
}

define i32 @preserves_decompression_table(ptr addrspace(6) noalias %d, ptr addrspace(6) %table, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @preserves_decompression_table(
; CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(
; CHECK: ret i32 0
  %before = load i32, ptr addrspace(6) %table
  call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.decompress_b(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 true, ptr addrspace(6) %table, i32 1, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %table
  %difference = sub i32 %after, %before
  ret i32 %difference
}

define i32 @disable_output_lane_preserves_decompression_table(ptr addrspace(6) noalias %d, ptr addrspace(6) %table, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @disable_output_lane_preserves_decompression_table(
; CHECK: call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(
; CHECK: ret i32 0
  %before = load i32, ptr addrspace(6) %table
  call void @llvm.nvvm.tcgen05.mma.shared.f8f6f4.disable_output_lane.cg1.decompress_b(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 true, ptr addrspace(6) %table, <4 x i32> zeroinitializer, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %table
  %difference = sub i32 %after, %before
  ret i32 %difference
}

define i32 @preserves_scales_and_decompression_table(ptr addrspace(6) noalias %d, ptr addrspace(6) %input, i64 %adesc, i64 %bdesc, i32 %idesc) {
; CHECK-LABEL: @preserves_scales_and_decompression_table(
; CHECK: call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32.decompress_b(
; CHECK: ret i32 0
  %before = load i32, ptr addrspace(6) %input
  call void @llvm.nvvm.tcgen05.mma.shared.mxf8f6f4.block_scale.block32.decompress_b(ptr addrspace(6) %d, i64 %adesc, i64 %bdesc, i32 %idesc, i1 true, ptr addrspace(6) %input, ptr addrspace(6) %input, ptr addrspace(6) %input, i32 1, i32 0, i32 0)
  %after = load i32, ptr addrspace(6) %input
  %difference = sub i32 %after, %before
  ret i32 %difference
}
