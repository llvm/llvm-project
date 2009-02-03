; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: grep adcl %t | count 7
; RUN: grep sbbl %t | count 7

define void @add(i256* %p, i256* %q) nounwind {
  %a = load i256* %p
  %b = load i256* %q
  %c = add i256 %a, %b
  store i256 %c, i256* %p
  ret void
}
define void @sub(i256* %p, i256* %q) nounwind {
  %a = load i256* %p
  %b = load i256* %q
  %c = sub i256 %a, %b
  store i256 %c, i256* %p
  ret void
}
