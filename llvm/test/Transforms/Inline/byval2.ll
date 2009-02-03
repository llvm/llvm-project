; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep {llvm.memcpy}

; Inlining a byval struct should NOT cause an explicit copy 
; into an alloca if the function is readonly

	%struct.ss = type { i32, i64 }
@.str = internal constant [10 x i8] c"%d, %lld\0A\00"		; <[10 x i8]*> [#uses=1]

define internal i32 @f(%struct.ss* byval  %b) nounwind readonly {
entry:
	%tmp = getelementptr %struct.ss* %b, i32 0, i32 0		; <i32*> [#uses=2]
	%tmp1 = load i32* %tmp, align 4		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=1]
	ret i32 %tmp2
}

declare i32 @printf(i8*, ...) nounwind 

define i32 @main() nounwind  {
entry:
	%S = alloca %struct.ss		; <%struct.ss*> [#uses=4]
	%tmp1 = getelementptr %struct.ss* %S, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 8
	%tmp4 = getelementptr %struct.ss* %S, i32 0, i32 1		; <i64*> [#uses=1]
	store i64 2, i64* %tmp4, align 4
	%X = call i32 @f( %struct.ss* byval  %S ) nounwind 
	ret i32 %X
}
