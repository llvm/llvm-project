; RUN: opt -passes='dse,gvn' -S < %s | FileCheck %s
; REQUIRES: nvptx-registered-target

target triple = "nvptx64-nvidia-cuda"

@shared = addrspace(3) global [4096 x i32] undef, align 16

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
