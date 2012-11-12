; RUN: opt < %s -instcombine -S | grep "add nsw i32"

define double @x(i32 %a, i32 %b) nounwind {
  %m = lshr i32 %a, 24
  %n = and i32 %m, %b
  %o = sitofp i32 %n to double
  %p = fadd double %o, 1.0
  ret double %p
}
