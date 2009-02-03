; RUN: llvm-as < %s | llc -march=xcore -mcpu=xs1b-generic > %t1.s
; RUN: grep ladd %t1.s | count 2
; RUN: grep lsub %t1.s | count 2
define i64 @add64(i64 %a, i64 %b) {
	%result = add i64 %a, %b
	ret i64 %result
}

define i64 @sub64(i64 %a, i64 %b) {
	%result = sub i64 %a, %b
	ret i64 %result
}
