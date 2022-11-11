; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define zeroext i1 @_Z3fooPb(ptr nocapture %x) {
entry:
  %a = load i8, ptr %x, align 1, !range !0
  %b = and i8 %a, 1
  %tobool = icmp ne i8 %b, 0
  ret i1 %tobool
}

; CHECK: %a = load i8, ptr %x, align 1, !range !0
; CHECK-NEXT: %tobool = icmp ne i8 %a, 0
; CHECK-NEXT: ret i1 %tobool

!0 = !{i8 0, i8 2}
