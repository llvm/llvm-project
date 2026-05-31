; Lower `atomicrmw` instructions to AIR `air.atomic.*` intrinsic calls.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: atomicrmw
; CHECK: call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 1, i1 true)
; CHECK: call i32 @air.atomic.local.xchg.i32(ptr addrspace(3) %l, i32 %v, i32 0, i32 1, i1 true)
; CHECK: ret void

define void @kernel(ptr addrspace(1) %g, ptr addrspace(3) %l, i32 %v) {
entry:
  %a = atomicrmw add ptr addrspace(1) %g, i32 %v seq_cst
  %b = atomicrmw xchg ptr addrspace(3) %l, i32 %v seq_cst
  ret void
}
