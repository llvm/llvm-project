; RUN: llc < %s -mtriple=ve | FileCheck %s

define i64 @lea1a(i64 %x) nounwind {
; CHECK-LABEL: lea1a:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s0, (%s0)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    or %s11, 0, %s9
  %asmtmp = tail call i64 asm "lea $0, ($1)", "=r,r"(i64 %x) nounwind
  ret i64 %asmtmp
}

define i64 @lea1b(i64 %x) nounwind {
; CHECK-LABEL: lea1b:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s0, (, %s0)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    or %s11, 0, %s9
  %asmtmp = tail call i64 asm "lea $0, (, $1)", "=r,r"(i64 %x) nounwind
  ret i64 %asmtmp
}

define i64 @lea2(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: lea2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s0, (%s0, %s1)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    or %s11, 0, %s9
  %asmtmp = tail call i64 asm "lea $0, ($1, $2)", "=r,r,r"(i64 %x, i64 %y) nounwind
  ret i64 %asmtmp
}

define i64 @lea3(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: lea3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea %s0, 2048(%s0, %s1)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    or %s11, 0, %s9
  %asmtmp = tail call i64 asm "lea $0, 2048($1, $2)", "=r,r,r"(i64 %x, i64 %y) nounwind
  ret i64 %asmtmp
}

define i64 @leasl3(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: leasl3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    lea.sl %s0, 2048(%s1, %s0)
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    or %s11, 0, %s9
  %asmtmp = tail call i64 asm "lea.sl $0, 2048($1, $2)", "=r,r,r"(i64 %y, i64 %x) nounwind
  ret i64 %asmtmp
}
