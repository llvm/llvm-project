; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; Two threadgroup barriers with nothing between them that touches memory are
; one barrier; a memory access between them keeps both.

@tg = internal addrspace(3) global [64 x i32] undef

declare void @llvm.agx.threadgroup.barrier()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; CHECK-LABEL: repeated:
; CHECK:         threadgroup_barrier
; CHECK-NEXT:    threadgroup_load
; CHECK:         threadgroup_barrier
; CHECK-NOT:     threadgroup_barrier
; CHECK:         stop
define agx_kernel void @repeated(ptr addrspace(1) %out) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %p = getelementptr [64 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  call void @llvm.agx.threadgroup.barrier()
  %x = load i32, ptr addrspace(3) %p
  call void @llvm.agx.threadgroup.barrier()
  call void @llvm.agx.threadgroup.barrier()
  %y = add i32 %x, 1
  call void @llvm.agx.threadgroup.barrier()
  store i32 %y, ptr addrspace(1) %out
  ret void
}
