; RUN: opt < %s -globalopt

@g = external global i32

@a = alias bitcast (i32* @g to i8*)

define void @f() {
	%tmp = load i8* @a
	ret void
}
