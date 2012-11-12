; RUN: llc < %s

define void @foo(double %a, double %b, float* %fp) {
	%c = fadd double %a, %b
	%d = fptrunc double %c to float
	store float %d, float* %fp
	ret void
}
