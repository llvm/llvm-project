; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {call.*llvm.stackrestore}
; PR2488
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@p = weak global i8* null		; <i8**> [#uses=2]

define i32 @main() nounwind  {
entry:
	%tmp248 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%tmp2752 = alloca i32		; <i32*> [#uses=2]
	%tmpcast53 = bitcast i32* %tmp2752 to i8*		; <i8*> [#uses=1]
	store i32 2, i32* %tmp2752, align 4
	volatile store i8* %tmpcast53, i8** @p, align 4
	br label %bb44

bb:		; preds = %bb44
	ret i32 0

bb44:		; preds = %bb44, %entry
	%indvar = phi i32 [ 0, %entry ], [ %tmp3857, %bb44 ]		; <i32> [#uses=1]
	%tmp249 = phi i8* [ %tmp248, %entry ], [ %tmp2, %bb44 ]		; <i8*> [#uses=1]
	%tmp3857 = add i32 %indvar, 1		; <i32> [#uses=3]
	call void @llvm.stackrestore( i8* %tmp249 )
	%tmp2 = call i8* @llvm.stacksave( )		; <i8*> [#uses=1]
	%tmp4 = srem i32 %tmp3857, 1000		; <i32> [#uses=2]
	%tmp5 = add i32 %tmp4, 1		; <i32> [#uses=1]
	%tmp27 = alloca i32, i32 %tmp5		; <i32*> [#uses=3]
	%tmpcast = bitcast i32* %tmp27 to i8*		; <i8*> [#uses=1]
	store i32 1, i32* %tmp27, align 4
	%tmp34 = getelementptr i32* %tmp27, i32 %tmp4		; <i32*> [#uses=1]
	store i32 2, i32* %tmp34, align 4
	volatile store i8* %tmpcast, i8** @p, align 4
	%exitcond = icmp eq i32 %tmp3857, 999999		; <i1> [#uses=1]
	br i1 %exitcond, label %bb, label %bb44
}

declare i8* @llvm.stacksave() nounwind 

declare void @llvm.stackrestore(i8*) nounwind 
