; RUN: llvm-as < %s | llc -march=x86 | \
; RUN:   ignore grep {movl..0} | count 0

%struct.obj = type { i8*, %struct.obj* }

declare void @g() gc "shadow-stack"

define void @f(i8* %o) gc "shadow-stack" {
entry:
	%root = alloca i8*
	call void @llvm.gcroot(i8** %root, i8* null)
	store i8* %o, i8** %root
	call void @g()
	ret void
}

declare void @llvm.gcroot(i8**, i8*)
