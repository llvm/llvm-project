; RUN: opt < %s -passes=globalopt

@g = global i32 0

@a = alias i8, ptr @g

define void @f() {
	%tmp = load i8, ptr @a
	ret void
}
