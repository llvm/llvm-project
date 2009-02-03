; RUN: llvm-as < %s | llc -asm-verbose | grep {frame size} | grep -v 0x0

declare void @llvm.gcroot(i8** %value, i8* %tag)
declare void @g() gc "ocaml"

define void @f(i8* %arg.0, void()* %arg.1) gc "ocaml" {
entry:
	%gcroot.0 = alloca i8*
	call void @llvm.gcroot(i8** %gcroot.0, i8* null)
	store i8* %arg.0, i8** %gcroot.0
	call void @g()
	call void %arg.1()
	ret void
}
