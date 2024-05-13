; RUN: opt < %s -passes=instcombine -S | \
; RUN:   grep "icmp sgt"
; END.
target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"
	%struct.point = type { i32, i32 }

define i32 @visible(i32 %direction, i64 %p1.0, i64 %p2.0, i64 %p3.0) {
entry:
	%p1_addr = alloca %struct.point		; <ptr> [#uses=2]
	%p2_addr = alloca %struct.point		; <ptr> [#uses=2]
	%p3_addr = alloca %struct.point		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp.upgrd.1 = getelementptr { i64 }, ptr %p1_addr, i64 0, i32 0		; <ptr> [#uses=1]
	store i64 %p1.0, ptr %tmp.upgrd.1
	%tmp2 = getelementptr { i64 }, ptr %p2_addr, i64 0, i32 0		; <ptr> [#uses=1]
	store i64 %p2.0, ptr %tmp2
	%tmp4 = getelementptr { i64 }, ptr %p3_addr, i64 0, i32 0		; <ptr> [#uses=1]
	store i64 %p3.0, ptr %tmp4
	%tmp.upgrd.2 = icmp eq i32 %direction, 0		; <i1> [#uses=1]
	%tmp6 = getelementptr { i64 }, ptr %p1_addr, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp.upgrd.3 = load i64, ptr %tmp6		; <i64> [#uses=1]
	%tmp8 = getelementptr { i64 }, ptr %p2_addr, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp9 = load i64, ptr %tmp8		; <i64> [#uses=1]
	%tmp11 = getelementptr { i64 }, ptr %p3_addr, i64 0, i32 0		; <ptr> [#uses=1]
	%tmp12 = load i64, ptr %tmp11		; <i64> [#uses=1]
	%tmp13 = call i32 @determinant( i64 %tmp.upgrd.3, i64 %tmp9, i64 %tmp12 )		; <i32> [#uses=2]
	br i1 %tmp.upgrd.2, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	%tmp14 = icmp slt i32 %tmp13, 0		; <i1> [#uses=1]
	%tmp14.upgrd.4 = zext i1 %tmp14 to i32		; <i32> [#uses=1]
	br label %return

cond_false:		; preds = %entry
	%tmp26 = icmp sgt i32 %tmp13, 0		; <i1> [#uses=1]
	%tmp26.upgrd.5 = zext i1 %tmp26 to i32		; <i32> [#uses=1]
	br label %return

return:		; preds = %cond_false, %cond_true
	%retval.0 = phi i32 [ %tmp14.upgrd.4, %cond_true ], [ %tmp26.upgrd.5, %cond_false ]		; <i32> [#uses=1]
	ret i32 %retval.0
}

declare i32 @determinant(i64, i64, i64)
