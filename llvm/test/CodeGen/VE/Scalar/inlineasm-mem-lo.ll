; RUN: llc < %s -mtriple=ve | FileCheck %s

define i64 @leam(i64 %x) nounwind {
; CHECK-LABEL: leam:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s0, 8(%s11)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %z = alloca i64, align 8
  %asmtmp = tail call i64 asm "lea $0, $1", "=r,*m"(i64* elementtype(i64) %z) nounwind
  ret i64 %asmtmp
}
