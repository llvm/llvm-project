; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movsd | count 1
; rdar://6504833

define float @f(i32 %x) nounwind readnone {
entry:
	%0 = uitofp i32 %x to float
	ret float %0
}
