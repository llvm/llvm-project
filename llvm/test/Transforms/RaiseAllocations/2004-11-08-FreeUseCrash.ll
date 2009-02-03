; RUN: llvm-as < %s | opt -raiseallocs -disable-output

define void @main() {
	%tmp.13 = call i32 (...)* @free( i32 32 )		; <i32> [#uses=1]
	%tmp.14 = inttoptr i32 %tmp.13 to i32*		; <i32*> [#uses=0]
	ret void
}

declare i32 @free(...)

