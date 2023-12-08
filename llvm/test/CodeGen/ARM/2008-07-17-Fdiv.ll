; RUN: llc -mtriple=arm-eabi %s -o /dev/null

define float @f(float %a, float %b) nounwind  {
	%tmp = fdiv float %a, %b
	ret float %tmp
}
