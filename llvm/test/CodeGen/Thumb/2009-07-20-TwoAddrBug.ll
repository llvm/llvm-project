; RUN: llc < %s -mtriple=thumbv6-apple-darwin10

@Time.2535 = external global i64		; <ptr> [#uses=2]

define i64 @millisecs() nounwind {
entry:
	%0 = load i64, ptr @Time.2535, align 4		; <i64> [#uses=2]
	%1 = add i64 %0, 1		; <i64> [#uses=1]
	store i64 %1, ptr @Time.2535, align 4
	ret i64 %0
}
