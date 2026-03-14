; RUN: llc < %s -mtriple=x86_64-apple-darwin10
; PR4587
; rdar://7072590

	%struct.re_pattern_buffer = type <{ ptr, i64, i64, i64, ptr, i64, i64, i8, i8, i8, i8, i8, i8, i8, i8 }>

define fastcc i32 @regex_compile(ptr %pattern, i64 %size, i64 %syntax, ptr nocapture %bufp) nounwind ssp {
entry:
	br i1 undef, label %return, label %if.end

if.end:		; preds = %entry
	%tmp35 = getelementptr %struct.re_pattern_buffer, ptr %bufp, i64 0, i32 3		; <ptr> [#uses=1]
	store i64 %syntax, ptr %tmp35
	store i32 undef, ptr undef
	br i1 undef, label %if.then66, label %if.end102

if.then66:		; preds = %if.end
	br i1 false, label %if.else, label %if.then70

if.then70:		; preds = %if.then66
	%call74 = call ptr @xrealloc(ptr undef, i64 32) nounwind ssp		; <ptr> [#uses=0]
	unreachable

if.else:		; preds = %if.then66
	br i1 false, label %do.body86, label %if.end99

do.body86:		; preds = %if.else
	br i1 false, label %do.end, label %if.then90

if.then90:		; preds = %do.body86
	unreachable

do.end:		; preds = %do.body86
	ret i32 12

if.end99:		; preds = %if.else
	br label %if.end102

if.end102:		; preds = %if.end99, %if.end
	br label %while.body

while.body:		; preds = %if.end1126, %sw.bb532, %while.body, %if.end102
	%laststart.2 = phi ptr [ null, %if.end102 ], [ %laststart.7.ph, %if.end1126 ], [ %laststart.2, %sw.bb532 ], [ %laststart.2, %while.body ]		; <ptr> [#uses=6]
	%b.1 = phi ptr [ undef, %if.end102 ], [ %ctg29688, %if.end1126 ], [ %b.1, %sw.bb532 ], [ %b.1, %while.body ]		; <ptr> [#uses=5]
	br i1 undef, label %while.body, label %if.end127

if.end127:		; preds = %while.body
	switch i32 undef, label %sw.bb532 [
		i32 123, label %handle_interval
		i32 92, label %do.body3527
	]

sw.bb532:		; preds = %if.end127
	br i1 undef, label %while.body, label %if.end808

if.end808:		; preds = %sw.bb532
	br i1 undef, label %while.cond1267.preheader, label %if.then811

while.cond1267.preheader:		; preds = %if.end808
	br i1 false, label %return, label %if.end1294

if.then811:		; preds = %if.end808
	%call817 = call fastcc ptr @skip_one_char(ptr %laststart.2) ssp		; <ptr> [#uses=0]
	br i1 undef, label %cond.end834, label %lor.lhs.false827

lor.lhs.false827:		; preds = %if.then811
	br label %cond.end834

cond.end834:		; preds = %lor.lhs.false827, %if.then811
	br i1 undef, label %land.lhs.true838, label %while.cond979.preheader

land.lhs.true838:		; preds = %cond.end834
	br i1 undef, label %if.then842, label %while.cond979.preheader

if.then842:		; preds = %land.lhs.true838
	%conv851 = trunc i64 undef to i32		; <i32> [#uses=1]
	br label %while.cond979.preheader

while.cond979.preheader:		; preds = %if.then842, %land.lhs.true838, %cond.end834
	%startoffset.0.ph = phi i32 [ 0, %cond.end834 ], [ 0, %land.lhs.true838 ], [ %conv851, %if.then842 ]		; <i32> [#uses=2]
	%laststart.7.ph = phi ptr [ %laststart.2, %cond.end834 ], [ %laststart.2, %land.lhs.true838 ], [ %laststart.2, %if.then842 ]		; <ptr> [#uses=3]
	%b.4.ph = phi ptr [ %b.1, %cond.end834 ], [ %b.1, %land.lhs.true838 ], [ %b.1, %if.then842 ]		; <ptr> [#uses=3]
	%ctg29688 = getelementptr i8, ptr %b.4.ph, i64 6		; <ptr> [#uses=1]
	br label %while.cond979

while.cond979:		; preds = %if.end1006, %while.cond979.preheader
	%cmp991 = icmp ugt i64 undef, 0		; <i1> [#uses=1]
	br i1 %cmp991, label %do.body994, label %while.end1088

do.body994:		; preds = %while.cond979
	br i1 undef, label %return, label %if.end1006

if.end1006:		; preds = %do.body994
	%cmp1014 = icmp ugt i64 undef, 32768		; <i1> [#uses=1]
	%storemerge10953 = select i1 %cmp1014, i64 32768, i64 undef		; <i64> [#uses=1]
	store i64 %storemerge10953, ptr undef
	br i1 false, label %return, label %while.cond979

while.end1088:		; preds = %while.cond979
	br i1 undef, label %if.then1091, label %if.else1101

if.then1091:		; preds = %while.end1088
	store i8 undef, ptr undef
	%idx.ext1132.pre = zext i32 %startoffset.0.ph to i64		; <i64> [#uses=1]
	%add.ptr1133.pre = getelementptr i8, ptr %laststart.7.ph, i64 %idx.ext1132.pre		; <ptr> [#uses=1]
	%sub.ptr.lhs.cast1135.pre = ptrtoint ptr %add.ptr1133.pre to i64		; <i64> [#uses=1]
	br label %if.end1126

if.else1101:		; preds = %while.end1088
	%cond1109 = select i1 undef, i32 18, i32 14		; <i32> [#uses=1]
	%idx.ext1112 = zext i32 %startoffset.0.ph to i64		; <i64> [#uses=1]
	%add.ptr1113 = getelementptr i8, ptr %laststart.7.ph, i64 %idx.ext1112		; <ptr> [#uses=2]
	%sub.ptr.rhs.cast1121 = ptrtoint ptr %add.ptr1113 to i64		; <i64> [#uses=1]
	call fastcc void @insert_op1(i32 %cond1109, ptr %add.ptr1113, i32 undef, ptr %b.4.ph) ssp
	br label %if.end1126

if.end1126:		; preds = %if.else1101, %if.then1091
	%sub.ptr.lhs.cast1135.pre-phi = phi i64 [ %sub.ptr.rhs.cast1121, %if.else1101 ], [ %sub.ptr.lhs.cast1135.pre, %if.then1091 ]		; <i64> [#uses=1]
	%add.ptr1128 = getelementptr i8, ptr %b.4.ph, i64 3		; <ptr> [#uses=1]
	%sub.ptr.rhs.cast1136 = ptrtoint ptr %add.ptr1128 to i64		; <i64> [#uses=1]
	%sub.ptr.sub1137 = sub i64 %sub.ptr.lhs.cast1135.pre-phi, %sub.ptr.rhs.cast1136		; <i64> [#uses=1]
	%sub.ptr.sub11378527 = trunc i64 %sub.ptr.sub1137 to i32		; <i32> [#uses=1]
	%conv1139 = add i32 %sub.ptr.sub11378527, -3		; <i32> [#uses=1]
	store i8 undef, ptr undef
	%shr10.i8599 = lshr i32 %conv1139, 8		; <i32> [#uses=1]
	%conv6.i8600 = trunc i32 %shr10.i8599 to i8		; <i8> [#uses=1]
	store i8 %conv6.i8600, ptr undef
	br label %while.body

if.end1294:		; preds = %while.cond1267.preheader
	ret i32 12

do.body3527:		; preds = %if.end127
	br i1 undef, label %do.end3536, label %if.then3531

if.then3531:		; preds = %do.body3527
	unreachable

do.end3536:		; preds = %do.body3527
	ret i32 5

handle_interval:		; preds = %if.end127
	br i1 undef, label %do.body4547, label %cond.false4583

do.body4547:		; preds = %handle_interval
	br i1 undef, label %do.end4556, label %if.then4551

if.then4551:		; preds = %do.body4547
	unreachable

do.end4556:		; preds = %do.body4547
	ret i32 9

cond.false4583:		; preds = %handle_interval
	unreachable

return:		; preds = %if.end1006, %do.body994, %while.cond1267.preheader, %entry
	ret i32 undef
}

declare ptr @xrealloc(ptr, i64) ssp

declare fastcc ptr @skip_one_char(ptr) nounwind readonly ssp

declare fastcc void @insert_op1(i32, ptr, i32, ptr) nounwind ssp
