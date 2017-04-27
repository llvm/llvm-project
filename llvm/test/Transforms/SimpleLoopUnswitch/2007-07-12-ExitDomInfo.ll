; RUN: opt < %s -simple-loop-unswitch -instcombine -disable-output

@str3 = external constant [3 x i8]		; <[3 x i8]*> [#uses=1]

define i32 @stringSearch_Clib(i32 %count) {
entry:
	%ttmp25 = icmp sgt i32 %count, 0		; <i1> [#uses=1]
	br i1 %ttmp25, label %bb36.preheader, label %bb44

bb36.preheader:		; preds = %entry
	%ttmp33 = icmp slt i32 0, 250		; <i1> [#uses=1]
	br label %bb36.outer

bb36.outer:		; preds = %bb41, %bb36.preheader
	br i1 %ttmp33, label %bb.nph, label %bb41

bb.nph:		; preds = %bb36.outer
	%ttmp8 = icmp eq i8* null, null		; <i1> [#uses=1]
	%ttmp6 = icmp eq i8* null, null		; <i1> [#uses=1]
	%tmp31 = call i32 @strcspn( i8* null, i8* getelementptr ([3 x i8], [3 x i8]* @str3, i64 0, i64 0) )		; <i32> [#uses=1]
	br i1 %ttmp8, label %cond_next, label %cond_true

cond_true:		; preds = %bb.nph
	ret i32 0

cond_next:		; preds = %bb.nph
	br i1 %ttmp6, label %cond_next28, label %cond_true20

cond_true20:		; preds = %cond_next
	ret i32 0

cond_next28:		; preds = %cond_next
	%tmp33 = add i32 %tmp31, 0		; <i32> [#uses=1]
	br label %bb41

bb41:		; preds = %cond_next28, %bb36.outer
	%c.2.lcssa = phi i32 [ 0, %bb36.outer ], [ %tmp33, %cond_next28 ]		; <i32> [#uses=1]
	br i1 false, label %bb36.outer, label %bb44

bb44:		; preds = %bb41, %entry
	%c.01.1 = phi i32 [ 0, %entry ], [ %c.2.lcssa, %bb41 ]		; <i32> [#uses=1]
	ret i32 %c.01.1
}

declare i32 @strcspn(i8*, i8*)
