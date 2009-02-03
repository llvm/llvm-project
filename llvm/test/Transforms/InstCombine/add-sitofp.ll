; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {add i32}

define double @x(i32 %a, i32 %b) nounwind {
  %m = lshr i32 %a, 24
  %n = and i32 %m, %b
  %o = sitofp i32 %n to double
  %p = add double %o, 1.0
  ret double %p
}
