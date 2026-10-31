; RUN: llc -mtriple=sparc -verify-machineinstrs < %s | FileCheck %s

; Leaf conversion must be decided from the registers that are actually used
; after allocation, so that a call-free function under register pressure keeps
; its register window instead of being forced to spill.

@g = global [8 x i32] zeroinitializer

; Eight simultaneously-live values plus two addresses do not fit in the
; registers a leaf may use. The function must retain its register window and
; not spill.
; CHECK-LABEL: high_pressure_no_call:
; CHECK: save %sp, -{{[0-9]+}}, %sp
; CHECK-NOT: %sp
; CHECK: ret
; CHECK-NEXT: restore
define void @high_pressure_no_call() {
entry:
  %p0 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 0
  %p1 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 1
  %p2 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 2
  %p3 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 3
  %p4 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 4
  %p5 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 5
  %p6 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 6
  %p7 = getelementptr inbounds [8 x i32], ptr @g, i32 0, i32 7
  %v0 = load volatile i32, ptr %p0
  %v1 = load volatile i32, ptr %p1
  %v2 = load volatile i32, ptr %p2
  %v3 = load volatile i32, ptr %p3
  %v4 = load volatile i32, ptr %p4
  %v5 = load volatile i32, ptr %p5
  %v6 = load volatile i32, ptr %p6
  %v7 = load volatile i32, ptr %p7
  store volatile i32 %v7, ptr %p0
  store volatile i32 %v6, ptr %p1
  store volatile i32 %v5, ptr %p2
  store volatile i32 %v4, ptr %p3
  store volatile i32 %v3, ptr %p4
  store volatile i32 %v2, ptr %p5
  store volatile i32 %v1, ptr %p6
  store volatile i32 %v0, ptr %p7
  ret void
}

; A small call-free function must still be converted to a leaf.
; CHECK-LABEL: low_pressure_no_call:
; CHECK-NOT: save
; CHECK: retl
; CHECK-NEXT: add %o0, %o1, %o0
define i32 @low_pressure_no_call(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
