; RUN: not llvm-as < %s >& /dev/null

declare void @llvm.gcroot(i8**, i8*)

define void @f(i8* %x) {
	%root = alloca i8*
	call void @llvm.gcroot(i8** %root, i8* null)
	store i8* %x, i8** %root
	ret void
}
