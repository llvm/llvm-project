; RUN: opt < %s -argpromotion

declare void @llvm.gcroot(ptr, ptr)

define i32 @g() {
entry:
	%var = alloca i32
	store i32 1, ptr %var
	%x = call i32 @f(ptr %var)
	ret i32 %x
}

define internal i32 @f(ptr %xp) gc "example" {
entry:
	%var = alloca ptr
	call void @llvm.gcroot(ptr %var, ptr null)
	%x = load i32, ptr %xp
	ret i32 %x
}
