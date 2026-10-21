; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=G14
; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s --check-prefix=TG
; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-strict -verify-machineinstrs < %s | FileCheck %s --check-prefix=TG

; llvm.agx.block.barrier is the G14 block barrier, which orders memory by
; itself; elsewhere (and in strict mode) it is a threadgroup barrier. The
; compiler keeps memory accesses on their side of it.

@tg = internal addrspace(3) global [128 x i32] undef

declare void @llvm.agx.block.barrier()
declare i32 @llvm.agx.thread.position.in.threadgroup.x()

; G14-LABEL: exchange:
; G14:         threadgroup_store
; G14-NEXT:    block_barrier
; G14-NOT:     memory_barrier
; G14:         threadgroup_load
; G14:         stop
; TG-LABEL:  exchange:
; TG:          threadgroup_store
; TG-NOT:      block_barrier
; TG:          threadgroup_barrier
; TG:          threadgroup_load
; TG:          stop
define agx_kernel void @exchange(ptr addrspace(1) %out) {
  %t = call i32 @llvm.agx.thread.position.in.threadgroup.x()
  %p = getelementptr [128 x i32], ptr addrspace(3) @tg, i32 0, i32 %t
  store i32 %t, ptr addrspace(3) %p
  call void @llvm.agx.block.barrier()
  %u = xor i32 %t, 1
  %q = getelementptr [128 x i32], ptr addrspace(3) @tg, i32 0, i32 %u
  %x = load i32, ptr addrspace(3) %q
  store i32 %x, ptr addrspace(1) %out
  ret void
}
