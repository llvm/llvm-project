; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {icmp eq i32 %indvar, 0}
; PR1978
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
	%struct.x = type <{ i8 }>
@.str = internal constant [6 x i8] c"Main!\00"		; <[6 x i8]*> [#uses=1]
@.str1 = internal constant [12 x i8] c"destroy %p\0A\00"		; <[12 x i8]*> [#uses=1]

define i32 @main() nounwind  {
entry:
	%orientations = alloca [1 x [1 x %struct.x]]		; <[1 x [1 x %struct.x]]*> [#uses=2]
	%tmp3 = call i32 @puts( i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0) ) nounwind 		; <i32> [#uses=0]
	%tmp45 = getelementptr [1 x [1 x %struct.x]]* %orientations, i32 1, i32 0, i32 0		; <%struct.x*> [#uses=1]
	%orientations62 = getelementptr [1 x [1 x %struct.x]]* %orientations, i32 0, i32 0, i32 0		; <%struct.x*> [#uses=1]
	br label %bb10

bb10:		; preds = %bb10, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb10 ]		; <i32> [#uses=2]
	%tmp.0.reg2mem.0.rec = mul i32 %indvar, -1		; <i32> [#uses=1]
	%tmp12.rec = add i32 %tmp.0.reg2mem.0.rec, -1		; <i32> [#uses=1]
	%tmp12 = getelementptr %struct.x* %tmp45, i32 %tmp12.rec		; <%struct.x*> [#uses=2]
	%tmp16 = call i32 (i8*, ...)* @printf( i8* getelementptr ([12 x i8]* @.str1, i32 0, i32 0), %struct.x* %tmp12 ) nounwind 		; <i32> [#uses=0]
	%tmp84 = icmp eq %struct.x* %tmp12, %orientations62		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp84, label %bb17, label %bb10

bb17:		; preds = %bb10
	ret i32 0
}

declare i32 @puts(i8*)

declare i32 @printf(i8*, ...)
