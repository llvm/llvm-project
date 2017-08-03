; RUN: llc < %s -march=x86 -mattr=sse2 | grep movsd | count 5
; RUN: llc < %s -march=x86 -mattr=sse2 -O0 | grep -v esp | grep movsd | count 5

@x = external global double

define void @foo() nounwind  {
  %a = load volatile double, double* @x
  store volatile double 0.0, double* @x
  store volatile double 0.0, double* @x
  %b = load volatile double, double* @x
  ret void
}

define void @bar() nounwind  {
  %c = load volatile double, double* @x
  ret void
}
