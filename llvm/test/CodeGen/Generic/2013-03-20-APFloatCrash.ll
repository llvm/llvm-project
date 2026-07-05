; RUN: llc < %s

define internal i1 @f(float %s) {
entry:
  %c = fcmp ogt float %s, 0x41EFFFFFE0000000
  ret i1 %c
}
