; RUN: llvm-as < %s | opt -predsimplify | llvm-dis | grep br | grep return.i.bb8_crit_edge | grep false
@str = external global [4 x i8]		; <[4 x i8]*> [#uses=1]

declare i32 @sprintf(i8*, i8*, ...)

define i32 @main() {
entry:
	br label %cond_true.outer
cond_true.outer:		; preds = %cond_true.i, %entry
	%i.0.0.ph = phi i32 [ 0, %entry ], [ %tmp5, %cond_true.i ]		; <i32> [#uses=1]
	%j.0.0.ph = phi i32 [ 0, %entry ], [ %tmp312, %cond_true.i ]		; <i32> [#uses=2]
	br label %cond_true
cond_true:		; preds = %return.i, %cond_true.outer
	%indvar.ui = phi i32 [ 0, %cond_true.outer ], [ %indvar.next, %return.i ]		; <i32> [#uses=2]
	%indvar = bitcast i32 %indvar.ui to i32		; <i32> [#uses=1]
	%i.0.0 = add i32 %indvar, %i.0.0.ph		; <i32> [#uses=3]
	%savedstack = call i8* @llvm.stacksave( )		; <i8*> [#uses=2]
	%tmp.i = icmp eq i32 %i.0.0, 0		; <i1> [#uses=1]
	%tmp5 = add i32 %i.0.0, 1		; <i32> [#uses=3]
	br i1 %tmp.i, label %return.i, label %cond_true.i
cond_true.i:		; preds = %cond_true
	%tmp.i.upgrd.1 = alloca [1000 x i8]		; <[1000 x i8]*> [#uses=1]
	%tmp.sub.i = getelementptr [1000 x i8]* %tmp.i.upgrd.1, i32 0, i32 0		; <i8*> [#uses=2]
	%tmp4.i = call i32 (i8*, i8*, ...)* @sprintf( i8* %tmp.sub.i, i8* getelementptr ([4 x i8]* @str, i32 0, i64 0), i32 %i.0.0 )		; <i32> [#uses=0]
	%tmp.i.upgrd.2 = load i8* %tmp.sub.i		; <i8> [#uses=1]
	%tmp7.i = sext i8 %tmp.i.upgrd.2 to i32		; <i32> [#uses=1]
	call void @llvm.stackrestore( i8* %savedstack )
	%tmp312 = add i32 %tmp7.i, %j.0.0.ph		; <i32> [#uses=2]
	%tmp19 = icmp sgt i32 %tmp5, 9999		; <i1> [#uses=1]
	br i1 %tmp19, label %bb8, label %cond_true.outer
return.i:		; preds = %cond_true
	call void @llvm.stackrestore( i8* %savedstack )
	%tmp21 = icmp sgt i32 %tmp5, 9999		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar.ui, 1		; <i32> [#uses=1]
	br i1 %tmp21, label %bb8, label %cond_true
bb8:		; preds = %return.i, %cond_true.i
	%j.0.1 = phi i32 [ %j.0.0.ph, %return.i ], [ %tmp312, %cond_true.i ]		; <i32> [#uses=1]
	%tmp10 = call i32 (i8*, ...)* @printf( i8* getelementptr ([4 x i8]* @str, i32 0, i64 0), i32 %j.0.1 )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @printf(i8*, ...)

declare i8* @llvm.stacksave()

declare void @llvm.stackrestore(i8*)
