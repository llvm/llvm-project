; RUN: opt -S -passes=normalize < %s | FileCheck %s

define double @foo(double %a0, double %a1) {
; CHECK-LABEL: foo(
entry:
; CHECK: %b
; CHECK: %d
; CHECK: %a
; CHECK: %c
  %a = fmul double %a0, %a1
  %b = fmul double %a0, 2.000000e+00
  %c = fmul double %a, 6.000000e+00
  %d = fmul double %b, 6.000000e+00
  ret double %d
}

declare double @bir()

declare double @bar()

define double @baz(double %x) {
; CHECK-LABEL: baz(
entry:
  %ifcond = fcmp one double %x, 0.000000e+00
  br i1 %ifcond, label %then, label %else

then:       ; preds = %entry
  %calltmp = call double @bir()
  br label %ifcont

else:       ; preds = %entry
  %calltmp1 = call double @bar()
  br label %ifcont

ifcont:     ; preds = %else, %then
; CHECK: %iftmp = phi double [ %calltmp1, %else ], [ %calltmp, %then ]
  %iftmp = phi double [ %calltmp, %then ], [ %calltmp1, %else ]
  ret double %iftmp
}

