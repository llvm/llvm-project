; RUN: llc < %s -mtriple=ve | FileCheck %s

@A = dso_local global i64 0, align 8

define i64 @leam(i64 %x) nounwind {
; CHECK-LABEL: leam:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, A@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, A@hi(, %s0)
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s0, (%s0)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    b.l.t (, %s10)
  %asmtmp = tail call i64 asm "lea $0, $1", "=r,*m"(ptr elementtype(i64) @A) nounwind
  ret i64 %asmtmp
}
