; RUN: llc -mtriple=sparc -filetype=obj < %s > /dev/null 2> %t2

define void @mul_double_cc(ptr noalias sret({ double, double }) %agg.result, double %a, double %b, double %c, double %d) {
entry:
  call void @__muldc3(ptr sret({ double, double }) %agg.result, double %a, double %b, double %c, double %d)
  ret void
}

declare void @__muldc3(ptr, double, double, double, double)
