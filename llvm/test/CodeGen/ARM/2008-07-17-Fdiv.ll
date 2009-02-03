; RUN: llvm-as < %s | llc -march=arm

define float @f(float %a, float %b) nounwind  {
	%tmp = fdiv float %a, %b
	ret float %tmp
}
