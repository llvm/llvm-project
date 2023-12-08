; RUN: llc < %s -fast-isel -fast-isel-abort=2 -verify-machineinstrs -mtriple=x86_64-apple-darwin10

; Just make sure these don't abort when lowering the arguments.
define i32 @t1(i32 %a, i32 %b, i32 %c) {
entry:
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %add, %c
  ret i32 %add1
}

define i64 @t2(i64 %a, i64 %b, i64 %c) {
entry:
  %add = add nsw i64 %b, %a
  %add1 = add nsw i64 %add, %c
  ret i64 %add1
}

define i64 @t3(i32 %a, i64 %b, i32 %c) {
entry:
  %conv = sext i32 %a to i64
  %add = add nsw i64 %conv, %b
  %conv1 = sext i32 %c to i64
  %add2 = add nsw i64 %add, %conv1
  ret i64 %add2
}

define float @t4(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h) {
entry:
  %add1 = fadd float %a, %b
  %add2 = fadd float %c, %d
  %add3 = fadd float %e, %f
  %add4 = fadd float %g, %h
  %add5 = fadd float %add1, %add2
  %add6 = fadd float %add3, %add4
  %add7 = fadd float %add5, %add6
  ret float %add7
}

define double @t5(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h) {
entry:
  %add1 = fadd double %a, %b
  %add2 = fadd double %c, %d
  %add3 = fadd double %e, %f
  %add4 = fadd double %g, %h
  %add5 = fadd double %add1, %add2
  %add6 = fadd double %add3, %add4
  %add7 = fadd double %add5, %add6
  ret double %add7
}
