; RUN: opt < %s -ipsccp -S | not grep global

@G = internal global i66 undef



define void @foo() {
	%X = load i66, i66* @G
	store i66 %X, i66* @G
	ret void
}

define i66 @bar() {
	%V = load i66, i66* @G
	%C = icmp eq i66 %V, 17
	br i1 %C, label %T, label %F
T:
	store i66 17, i66* @G
	ret i66 %V
F:
	store i66 123, i66* @G
	ret i66 0
}
