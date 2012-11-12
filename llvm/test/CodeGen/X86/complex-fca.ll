; RUN: llc < %s -march=x86 | grep mov | count 2

define void @ccosl({ x86_fp80, x86_fp80 }* noalias sret %agg.result, { x86_fp80, x86_fp80 } %z) nounwind {
entry:
	%z8 = extractvalue { x86_fp80, x86_fp80 } %z, 0
	%z9 = extractvalue { x86_fp80, x86_fp80 } %z, 1
	%0 = fsub x86_fp80 0xK80000000000000000000, %z9
	%insert = insertvalue { x86_fp80, x86_fp80 } undef, x86_fp80 %0, 0
	%insert7 = insertvalue { x86_fp80, x86_fp80 } %insert, x86_fp80 %z8, 1
	call void @ccoshl({ x86_fp80, x86_fp80 }* noalias sret %agg.result, { x86_fp80, x86_fp80 } %insert7) nounwind
	ret void
}

declare void @ccoshl({ x86_fp80, x86_fp80 }* noalias sret, { x86_fp80, x86_fp80 }) nounwind
