; RUN: not llvm-as < %s > /dev/null 2>&1

declare void @llvm.gcroot(ptr, ptr)

define void @f(ptr %x) {
	%root = alloca ptr
	call void @llvm.gcroot(ptr %root, ptr null)
	store ptr %x, ptr %root
	ret void
}
