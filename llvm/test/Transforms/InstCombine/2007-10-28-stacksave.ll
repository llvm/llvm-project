; RUN: opt < %s -passes=instcombine -S | grep "call.*stacksave"
; PR1745
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
@p = weak global ptr null		; <ptr> [#uses=1]

define i32 @main() {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	br label %lab

lab:		; preds = %cleanup31, %entry
	%n.0 = phi i32 [ 0, %entry ], [ %tmp25, %cleanup31 ]		; <i32> [#uses=2]
	%tmp2 = call ptr @llvm.stacksave( )		; <ptr> [#uses=2]
	%tmp4 = srem i32 %n.0, 47		; <i32> [#uses=1]
	%tmp5 = add i32 %tmp4, 1		; <i32> [#uses=5]
	%tmp7 = sub i32 %tmp5, 1		; <i32> [#uses=0]
	%tmp89 = zext i32 %tmp5 to i64		; <i64> [#uses=1]
	%tmp10 = mul i64 %tmp89, 32		; <i64> [#uses=0]
	%tmp12 = mul i32 %tmp5, 4		; <i32> [#uses=0]
	%tmp1314 = zext i32 %tmp5 to i64		; <i64> [#uses=1]
	%tmp15 = mul i64 %tmp1314, 32		; <i64> [#uses=0]
	%tmp17 = mul i32 %tmp5, 4		; <i32> [#uses=1]
	%tmp18 = alloca i8, i32 %tmp17		; <ptr> [#uses=1]
	%tmp21 = getelementptr i32, ptr %tmp18, i32 0		; <ptr> [#uses=1]
	store i32 1, ptr %tmp21, align 4
	store volatile ptr %tmp18, ptr @p, align 4
	%tmp25 = add i32 %n.0, 1		; <i32> [#uses=2]
	%tmp27 = icmp sle i32 %tmp25, 999999		; <i1> [#uses=1]
	%tmp2728 = zext i1 %tmp27 to i8		; <i8> [#uses=1]
	%toBool = icmp ne i8 %tmp2728, 0		; <i1> [#uses=1]
	br i1 %toBool, label %cleanup31, label %cond_next

cond_next:		; preds = %lab
	call void @llvm.stackrestore( ptr %tmp2 )
	ret i32 0

cleanup31:		; preds = %lab
	call void @llvm.stackrestore( ptr %tmp2 )
	br label %lab
}

declare ptr @llvm.stacksave()

declare void @llvm.stackrestore(ptr)
