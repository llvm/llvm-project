; RUN: llvm-as < %s | opt -strip | llvm-dis | grep foo | count 2
; RUN: llvm-as < %s | opt -strip | llvm-dis | grep bar | count 2
@llvm.used = appending global [2 x i8*] [ i8* bitcast (i32* @foo to i8*), i8* bitcast (i32 ()* @bar to i8*) ], section "llvm.metadata"		; <[2 x i8*]*> [#uses=0]
@foo = internal constant i32 41		; <i32*> [#uses=1]

define internal i32 @bar() nounwind  {
entry:
	ret i32 42
}

define i32 @main() nounwind  {
entry:
	ret i32 0
}

