; RUN: llc < %s -march=x86 -mattr=+sse2 | grep movs | count 2
; RUN: llc < %s -march=x86 -mattr=+sse2 | grep fld | count 2
; check 'inreg' attribute for sse_regparm

define inreg double @foo1()  nounwind {
  ret double 1.0
}

define inreg float @foo2()  nounwind {
  ret float 1.0
}

define double @bar() nounwind {
  ret double 1.0
}

define float @bar2() nounwind {
  ret float 1.0
}
