; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1
; RUN: grep "callq	__fixunsxfti@PLT" %t1

define i128 @f(x86_fp80 %a) nounwind {
entry:
	%tmp78 = fptoui x86_fp80 %a to i128
	ret i128 %tmp78
}
