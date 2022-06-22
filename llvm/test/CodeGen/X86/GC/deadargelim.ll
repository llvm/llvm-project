; RUN: opt < %s -deadargelim

declare void @llvm.gcroot(ptr, ptr)

define void @g() {
entry:
	call void @f(i32 0)
	ret void
}

define internal void @f(i32 %unused) gc "example" {
entry:
	%var = alloca ptr
	call void @llvm.gcroot(ptr %var, ptr null)
	ret void
}
