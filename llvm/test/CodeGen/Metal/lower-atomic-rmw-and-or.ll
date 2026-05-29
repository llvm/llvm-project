; The lower-atomic-rmw pass also covers logical RMW ops (and/or/xor) — pin
; their AIR-intrinsic names so a regression in the opcode -> name table is
; caught immediately.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: atomicrmw
; CHECK: call i32 @air.atomic.global.and.s.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 1, i1 true)
; CHECK: call i32 @air.atomic.global.or.s.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 1, i1 true)
; CHECK: call i32 @air.atomic.global.xor.s.i32(ptr addrspace(1) %g, i32 %v, i32 0, i32 1, i1 true)
; CHECK: ret void

define void @kernel(ptr addrspace(1) %g, i32 %v) {
entry:
  %a = atomicrmw and ptr addrspace(1) %g, i32 %v seq_cst
  %b = atomicrmw or  ptr addrspace(1) %g, i32 %v seq_cst
  %c = atomicrmw xor ptr addrspace(1) %g, i32 %v seq_cst
  ret void
}
