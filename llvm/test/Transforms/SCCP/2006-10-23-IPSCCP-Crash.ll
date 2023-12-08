; RUN: opt < %s -passes=sccp -disable-output
; END.
target datalayout = "E-p:32:32"
target triple = "powerpc-unknown-linux-gnu"
	%struct.pat_list = type { i32, ptr }
@JUMP = external global i32		; <ptr> [#uses=1]
@old_D_pat = external global [16 x i8]		; <ptr> [#uses=0]

define void @asearch1(i32 %D) {
entry:
	%tmp80 = icmp ult i32 0, %D		; <i1> [#uses=1]
	br i1 %tmp80, label %bb647.preheader, label %cond_true81.preheader
cond_true81.preheader:		; preds = %entry
	ret void
bb647.preheader:		; preds = %entry
	%tmp3.i = call i32 @read( )		; <i32> [#uses=1]
	%tmp6.i = add i32 %tmp3.i, 0		; <i32> [#uses=1]
	%tmp653 = icmp sgt i32 %tmp6.i, 0		; <i1> [#uses=1]
	br i1 %tmp653, label %cond_true654, label %UnifiedReturnBlock
cond_true612:		; preds = %cond_true654
	ret void
cond_next624:		; preds = %cond_true654
	ret void
cond_true654:		; preds = %bb647.preheader
	br i1 undef, label %cond_true612, label %cond_next624
UnifiedReturnBlock:		; preds = %bb647.preheader
	ret void
}

define void @bitap(i32 %D) {
entry:
	%tmp29 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp29, label %cond_next50, label %cond_next37
cond_next37:		; preds = %entry
	ret void
cond_next50:		; preds = %entry
	%tmp52 = icmp sgt i32 %D, 0		; <i1> [#uses=1]
	br i1 %tmp52, label %cond_true53, label %cond_next71
cond_true53:		; preds = %cond_next50
	%tmp54 = load i32, ptr @JUMP		; <i32> [#uses=1]
	%tmp55 = icmp eq i32 %tmp54, 1		; <i1> [#uses=1]
	br i1 %tmp55, label %cond_true56, label %cond_next63
cond_true56:		; preds = %cond_true53
	%tmp57 = bitcast i32 %D to i32		; <i32> [#uses=1]
	call void @asearch1( i32 %tmp57 )
	ret void
cond_next63:		; preds = %cond_true53
	ret void
cond_next71:		; preds = %cond_next50
	ret void
}

declare i32 @read()

define void @initial_value() {
entry:
	ret void
}

define void @main() {
entry:
	br label %cond_next252
cond_next208:		; preds = %cond_true260
	%tmp229 = call i32 @atoi( )		; <i32> [#uses=1]
	br label %cond_next252
bb217:		; preds = %cond_true260
	ret void
cond_next252:		; preds = %cond_next208, %entry
	%D.0.0 = phi i32 [ 0, %entry ], [ %tmp229, %cond_next208 ]		; <i32> [#uses=1]
	%tmp254 = getelementptr ptr, ptr null, i32 1		; <ptr> [#uses=1]
	%tmp256 = load ptr, ptr %tmp254		; <ptr> [#uses=1]
	%tmp258 = load i8, ptr %tmp256		; <i8> [#uses=1]
	%tmp259 = icmp eq i8 %tmp258, 45		; <i1> [#uses=1]
	br i1 %tmp259, label %cond_true260, label %bb263
cond_true260:		; preds = %cond_next252
	%tmp205818 = icmp sgt i8 0, -1		; <i1> [#uses=1]
	br i1 %tmp205818, label %cond_next208, label %bb217
bb263:		; preds = %cond_next252
	%tmp265 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp265, label %cond_next276, label %cond_true266
cond_true266:		; preds = %bb263
	ret void
cond_next276:		; preds = %bb263
	%tmp278 = icmp eq i32 0, 0		; <i1> [#uses=1]
	br i1 %tmp278, label %cond_next298, label %cond_true279
cond_true279:		; preds = %cond_next276
	ret void
cond_next298:		; preds = %cond_next276
	call void @bitap( i32 %D.0.0 )
	ret void
}

declare i32 @atoi()

define void @subset_pset() {
entry:
	ret void
}

define void @strcmp() {
entry:
	ret void
}
