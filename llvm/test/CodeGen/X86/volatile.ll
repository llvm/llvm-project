; RUN: llvm-as < %s | llc -march=x86 -mattr=sse2 | grep movsd | count 5
; RUN: llvm-as < %s | llc -march=x86 -mattr=sse2 -fast | grep movsd | count 5

@x = external global double

define void @foo() nounwind  {
  %a = volatile load double* @x
  volatile store double 0.0, double* @x
  volatile store double 0.0, double* @x
  %b = volatile load double* @x
  ret void
}

define void @bar() nounwind  {
  %c = volatile load double* @x
  ret void
}
