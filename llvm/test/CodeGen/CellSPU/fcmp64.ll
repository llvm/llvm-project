; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s

define i1 @fcmp_eq_setcc_f64(double %arg1, double %arg2) nounwind {
entry:
       %A = fcmp oeq double %arg1, %arg2
       ret i1 %A
}
