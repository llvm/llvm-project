; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b < %s | FileCheck %s

; When a function needs dynamic stack realignment, AP (here r16) is
; initialized by PS_aligna in the prologue. AP is a callee-saved register, so it
; must be saved before PS_aligna clobbers it and restored on the way out.
; PS_aligna is created during prologue emission (after callee-saved registers
; are determined), so HexagonFrameLowering::determineCalleeSaves must explicitly
; add AP to the save set; otherwise the caller's r16 is corrupted.

; CHECK-LABEL: f:
; CHECK: r16 = and(r30,#-128)
; CHECK: memd(r30+#-8) = r17:16
; CHECK: r17:16 = memd(r30+#-8)
; CHECK: dealloc_return

declare void @use(ptr, ptr)

define void @f(i32 %n) {
entry:
  ; Variable-sized alloca -> hasVarSizedObjects -> needsAligna.
  %vla = alloca i32, i32 %n, align 4
  ; Over-aligned local -> MaxAlign > stack alignment -> real realignment.
  %buf = alloca [16 x i32], align 128
  call void @use(ptr %vla, ptr %buf)
  ret void
}
