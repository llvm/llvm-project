; RUN: llvm-as < %s | opt -deadargelim

declare void @llvm.gcroot(i8**, i8*)

define void @g() {
entry:
	call void @f(i32 0)
	ret void
}

define internal void @f(i32 %unused) gc "example" {
entry:
	%var = alloca i8*
	call void @llvm.gcroot(i8** %var, i8* null)
	ret void
}
