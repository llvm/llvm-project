; RUN: llc < %s -mtriple=x86_64-apple-darwin -o %t1
; RUN: grep GOTPCREL %t1 | count 4
; RUN: grep %%rip      %t1 | count 6
; RUN: grep movq     %t1 | count 6
; RUN: grep leaq     %t1 | count 1
; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=static -o %t2
; RUN: grep movl %t2 | count 2
; RUN: grep movq %t2 | count 2

@ptr = external global i32*		; <i32**> [#uses=1]
@src = external global [0 x i32]		; <[0 x i32]*> [#uses=1]
@dst = external global [0 x i32]		; <[0 x i32]*> [#uses=1]
@lptr = internal global i32* null		; <i32**> [#uses=1]
@ldst = internal global [500 x i32] zeroinitializer, align 32		; <[500 x i32]*> [#uses=1]
@lsrc = internal global [500 x i32] zeroinitializer, align 32		; <[500 x i32]*> [#uses=0]
@bsrc = internal global [500000 x i32] zeroinitializer, align 32		; <[500000 x i32]*> [#uses=0]
@bdst = internal global [500000 x i32] zeroinitializer, align 32		; <[500000 x i32]*> [#uses=0]

define void @test1() nounwind {
	%tmp = load i32* getelementptr ([0 x i32]* @src, i32 0, i32 0)		; <i32> [#uses=1]
	store i32 %tmp, i32* getelementptr ([0 x i32]* @dst, i32 0, i32 0)
	ret void
}

define void @test2() nounwind {
	store i32* getelementptr ([0 x i32]* @dst, i32 0, i32 0), i32** @ptr
	ret void
}

define void @test3() nounwind {
	store i32* getelementptr ([500 x i32]* @ldst, i32 0, i32 0), i32** @lptr
	br label %return

return:		; preds = %0
	ret void
}
