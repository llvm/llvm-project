; RUN: llc -march=mipsel < %s | FileCheck %s

@foo12.d4 = internal unnamed_addr global double 0.000000e+00, align 8

define double @foo12(i32 %a, i32, i64 %b) nounwind {
entry:
; check that this transformation doesn't happen:
; (sint_to_fp (setcc x, y, cc)) -> (select_cc x, y, -1.0, 0.0,, cc)
;
; CHECK-NOT:   # double -1

  %tobool1 = icmp ne i32 %a, 0
  %not.tobool = icmp ne i64 %b, 0
  %tobool1. = or i1 %tobool1, %not.tobool
  %lor.ext = zext i1 %tobool1. to i32
  %conv = sitofp i32 %lor.ext to double
  %1 = load double, ptr @foo12.d4, align 8
  %add = fadd double %conv, %1
  store double %add, ptr @foo12.d4, align 8
  ret double %add
}
