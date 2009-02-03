; RUN: llvm-as < %s | opt -aa-eval -print-all-alias-modref-info -disable-output |& grep {MustAlias:.*%R,.*%r}
; Make sure that basicaa thinks R and r are must aliases.

define i32 @test(i8 * %P) {
entry:
	%Q = bitcast i8* %P to {i32, i32}*
	%R = getelementptr {i32, i32}* %Q, i32 0, i32 1
	%S = load i32* %R

	%q = bitcast i8* %P to {i32, i32}*
	%r = getelementptr {i32, i32}* %q, i32 0, i32 1
	%s = load i32* %r

	%t = sub i32 %S, %s
	ret i32 %t
}
