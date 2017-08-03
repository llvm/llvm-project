; RUN: llc < %s -mtriple=i686-- | not grep flds

define void @foo(x86_fp80 %a, x86_fp80 %b, float* %fp) {
	%c = fadd x86_fp80 %a, %b
	%d = fptrunc x86_fp80 %c to float
	store float %d, float* %fp
	ret void
}
