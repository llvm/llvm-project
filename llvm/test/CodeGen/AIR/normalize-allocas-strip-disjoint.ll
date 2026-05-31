; Focused fixture for one side of AIRNormalizeAllocas: the `disjoint` flag
; on `or` is unconditionally stripped (AIR v1 bitcode rejects it).
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: or disjoint
; CHECK: %r = or i32 %a, %b
; CHECK: ret void

define void @kernel(i32 %a, i32 %b, ptr addrspace(1) %out) {
entry:
  %r = or disjoint i32 %a, %b
  store i32 %r, ptr addrspace(1) %out
  ret void
}
