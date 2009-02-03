; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movs | count 2
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep fld | count 2
; check 'inreg' attribute for sse_regparm

define double @foo1() inreg nounwind {
  ret double 1.0
}

define float @foo2() inreg nounwind {
  ret float 1.0
}

define double @bar() nounwind {
  ret double 1.0
}

define float @bar2() nounwind {
  ret float 1.0
}
