; RUN: llc -mtriple=aarch64 < %s

define i256 @doubletosi256(double %a) {
  %conv = fptosi double %a to i256
  ret i256 %conv
}

define double @si256todouble(i256 %a) {
  %conv = sitofp i256 %a to double
  ret double %conv
}
