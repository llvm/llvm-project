; RUN: opt < %s -passes=instcombine -S | grep "call.*llvm.stackrestore"
; PR2488
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@p = weak global ptr null		; <ptr> [#uses=2]

define i32 @main() nounwind  {
entry:
	%tmp248 = call ptr @llvm.stacksave( )		; <ptr> [#uses=1]
	%tmp2752 = alloca i32		; <ptr> [#uses=2]
	store i32 2, ptr %tmp2752, align 4
	store volatile ptr %tmp2752, ptr @p, align 4
	br label %bb44

bb:		; preds = %bb44
	ret i32 0

bb44:		; preds = %bb44, %entry
	%indvar = phi i32 [ 0, %entry ], [ %tmp3857, %bb44 ]		; <i32> [#uses=1]
	%tmp249 = phi ptr [ %tmp248, %entry ], [ %tmp2, %bb44 ]		; <ptr> [#uses=1]
	%tmp3857 = add i32 %indvar, 1		; <i32> [#uses=3]
	call void @llvm.stackrestore( ptr %tmp249 )
	%tmp2 = call ptr @llvm.stacksave( )		; <ptr> [#uses=1]
	%tmp4 = srem i32 %tmp3857, 1000		; <i32> [#uses=2]
	%tmp5 = add i32 %tmp4, 1		; <i32> [#uses=1]
	%tmp27 = alloca i32, i32 %tmp5		; <ptr> [#uses=3]
	store i32 1, ptr %tmp27, align 4
	%tmp34 = getelementptr i32, ptr %tmp27, i32 %tmp4		; <ptr> [#uses=1]
	store i32 2, ptr %tmp34, align 4
	store volatile ptr %tmp27, ptr @p, align 4
	%exitcond = icmp eq i32 %tmp3857, 999999		; <i1> [#uses=1]
	br i1 %exitcond, label %bb, label %bb44
}

declare ptr @llvm.stacksave() nounwind 

declare void @llvm.stackrestore(ptr) nounwind 
