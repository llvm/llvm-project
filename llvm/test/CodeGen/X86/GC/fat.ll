; RUN: not llvm-as < %s > /dev/null 2>&1

declare void @llvm.gcroot(i8**, i8*) nounwind

define void @f() gc "x" {
	%st = alloca { i8*, i1 }		; <{ i8*, i1 }*> [#uses=1]
	%st_ptr = bitcast { i8*, i1 }* %st to i8**		; <i8**> [#uses=1]
	call void @llvm.gcroot(i8** %st_ptr, i8* null)
	ret void
}
