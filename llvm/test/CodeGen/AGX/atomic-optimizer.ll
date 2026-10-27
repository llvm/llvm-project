; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs -agx-atomic-optimizer < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s --check-prefix=OFF

; With -agx-atomic-optimizer, atomics of a SIMD-group's threads to one uniform
; address are one atomic by the first active thread: of the active count times
; a uniform operand, or of the SIMD-group's sum; each thread's old value is the
; first thread's result plus an exclusive scan.

; CHECK-LABEL: counter:
; CHECK:         icmp_ballot [[B:r[0-9]+]]
; CHECK:         popcount [[N:r[0-9]+]], [[B]]
; CHECK:         atomic add, 0, [[N]], u2_u3,
; CHECK:         simd_shuffle
; OFF-LABEL: counter:
; OFF-NOT:       popcount
; OFF:           atomic add, 0, r{{[0-9]+}}, u2_u3,
define agx_kernel void @counter(ptr addrspace(1) %out, ptr addrspace(1) %count, float %v) {
  %slot = atomicrmw add ptr addrspace(1) %count, i32 1 monotonic
  %p = getelementptr float, ptr addrspace(1) %out, i32 %slot
  store float %v, ptr addrspace(1) %p
  ret void
}

; A thread-varying operand: the SIMD-group's sum, no result needed.
; CHECK-LABEL: sum:
; CHECK:         simd_iadd
; CHECK:         atomic add, 0, _, u0_u1,
define agx_kernel void @sum(ptr addrspace(1) %total, ptr addrspace(1) %x) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr i32, ptr addrspace(1) %x, i32 %i
  %v = load i32, ptr addrspace(1) %p
  %r = atomicrmw add ptr addrspace(1) %total, i32 %v monotonic
  ret void
}

; Thread-varying addresses and float adds are left alone.
; CHECK-LABEL: scatter:
; CHECK-NOT:     popcount
; CHECK-NOT:     simd_
; CHECK:         atomic add
define agx_kernel void @scatter(ptr addrspace(1) %h, ptr addrspace(1) %f, float %v) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %p = getelementptr i32, ptr addrspace(1) %h, i32 %i
  %r = atomicrmw add ptr addrspace(1) %p, i32 1 monotonic
  %s = atomicrmw fadd ptr addrspace(1) %f, float %v monotonic
  ret void
}

declare i32 @llvm.agx.thread.position.in.grid.x()
