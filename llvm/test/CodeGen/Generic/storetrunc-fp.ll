; RUN: llvm-as < %s | llc

define void @foo(double %a, double %b, float* %fp) {
	%c = add double %a, %b
	%d = fptrunc double %c to float
	store float %d, float* %fp
	ret void
}
