; RUN: opt < %s -argpromotion

declare void @llvm.gcroot(i8**, i8*)

define i32 @g() {
entry:
	%var = alloca i32
	store i32 1, i32* %var
	%x = call i32 @f(i32* %var)
	ret i32 %x
}

define internal i32 @f(i32* %xp) gc "example" {
entry:
	%var = alloca i8*
	call void @llvm.gcroot(i8** %var, i8* null)
	%x = load i32* %xp
	ret i32 %x
}
