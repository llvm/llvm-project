; RUN: llc -march=sparc -filetype=obj < %s > /dev/null 2> %t2

define void @mul_double_cc({ double, double }* noalias sret({ double, double }) %agg.result, double %a, double %b, double %c, double %d) {
entry:
  call void @__muldc3({ double, double }* sret({ double, double }) %agg.result, double %a, double %b, double %c, double %d)
  ret void
}

declare void @__muldc3({ double, double }*, double, double, double, double)
