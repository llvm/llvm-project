; RUN: llc < %s -frame-pointer=all -O0 -mcpu=i486 | grep 1082126238 | count 3
; RUN: llc < %s -frame-pointer=all -O0 -mcpu=i486 | grep -- -1236950581 | count 1
;; magic constants are 3.999f and half of 3.999
; ModuleID = '1489.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-apple-darwin8"
@.str = internal constant [13 x i8] c"%d %d %d %d\0A\00"		; <[13 x i8]*> [#uses=1]

define i32 @quux() nounwind {
entry:
	%tmp1 = tail call i32 @lrintf( float 0x400FFDF3C0000000 )		; <i32> [#uses=1]
	%tmp2 = icmp slt i32 %tmp1, 1		; <i1> [#uses=1]
	%tmp23 = zext i1 %tmp2 to i32		; <i32> [#uses=1]
	ret i32 %tmp23
}

declare i32 @lrintf(float)

define i32 @foo() nounwind {
entry:
	%tmp1 = tail call i32 @lrint( double 3.999000e+00 )		; <i32> [#uses=1]
	%tmp2 = icmp slt i32 %tmp1, 1		; <i1> [#uses=1]
	%tmp23 = zext i1 %tmp2 to i32		; <i32> [#uses=1]
	ret i32 %tmp23
}

declare i32 @lrint(double)

define i32 @bar() nounwind {
entry:
	%tmp1 = tail call i32 @lrintf( float 0x400FFDF3C0000000 )		; <i32> [#uses=1]
	%tmp2 = icmp slt i32 %tmp1, 1		; <i1> [#uses=1]
	%tmp23 = zext i1 %tmp2 to i32		; <i32> [#uses=1]
	ret i32 %tmp23
}

define i32 @baz() nounwind {
entry:
	%tmp1 = tail call i32 @lrintf( float 0x400FFDF3C0000000 )		; <i32> [#uses=1]
	%tmp2 = icmp slt i32 %tmp1, 1		; <i1> [#uses=1]
	%tmp23 = zext i1 %tmp2 to i32		; <i32> [#uses=1]
	ret i32 %tmp23
}

define i32 @main() nounwind {
entry:
	%tmp = tail call i32 @baz( )		; <i32> [#uses=1]
	%tmp1 = tail call i32 @bar( )		; <i32> [#uses=1]
	%tmp2 = tail call i32 @foo( )		; <i32> [#uses=1]
	%tmp3 = tail call i32 @quux( )		; <i32> [#uses=1]
	%tmp5 = tail call i32 (i8*, ...) @printf( i8* getelementptr ([13 x i8], [13 x i8]* @.str, i32 0, i32 0), i32 %tmp3, i32 %tmp2, i32 %tmp1, i32 %tmp )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @printf(i8*, ...)
