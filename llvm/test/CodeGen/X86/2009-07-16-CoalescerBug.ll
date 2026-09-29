; RUN: llc < %s -mtriple=x86_64-apple-darwin10
; rdar://7059496

	%struct.brinfo = type <{ ptr, ptr, ptr, i32, i32, i32, i8, i8, i8, i8 }>
	%struct.cadata = type <{ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i8, i8, i8, i8 }>
	%struct.cline = type <{ ptr, i32, i8, i8, i8, i8, ptr, i32, i8, i8, i8, i8, ptr, i32, i8, i8, i8, i8, ptr, i32, i32, ptr, ptr, i32, i32 }>
	%struct.cmatch = type <{ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i8, i8, i8, i8, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i16, i8, i8, i16, i8, i8 }>
	%struct.cmatcher = type <{ i32, i8, i8, i8, i8, ptr, i32, i8, i8, i8, i8, ptr, i32, i8, i8, i8, i8, ptr, i32, i8, i8, i8, i8, ptr, i32, i8, i8, i8, i8, ptr, i32, i8, i8, i8, i8 }>
	%struct.cpattern = type <{ ptr, i32, i8, i8, i8, i8, %union.anon }>
	%struct.patprog = type <{ i64, i64, i64, i64, i32, i32, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8 }>
	%union.anon = type <{ [8 x i8] }>

define i32 @addmatches(ptr %dat, ptr nocapture %argv) nounwind ssp {
entry:
	br i1 undef, label %if.else, label %if.then91

if.then91:		; preds = %entry
	br label %if.end96

if.else:		; preds = %entry
	br label %if.end96

if.end96:		; preds = %if.else, %if.then91
	br i1 undef, label %lor.lhs.false, label %if.then105

lor.lhs.false:		; preds = %if.end96
	br i1 undef, label %if.else139, label %if.then105

if.then105:		; preds = %lor.lhs.false, %if.end96
	unreachable

if.else139:		; preds = %lor.lhs.false
	br i1 undef, label %land.end, label %land.rhs

land.rhs:		; preds = %if.else139
	unreachable

land.end:		; preds = %if.else139
	br i1 undef, label %land.lhs.true285, label %if.then315

land.lhs.true285:		; preds = %land.end
	br i1 undef, label %if.end324, label %if.then322

if.then315:		; preds = %land.end
	unreachable

if.then322:		; preds = %land.lhs.true285
	unreachable

if.end324:		; preds = %land.lhs.true285
	br i1 undef, label %if.end384, label %if.then358

if.then358:		; preds = %if.end324
	unreachable

if.end384:		; preds = %if.end324
	br i1 undef, label %if.end394, label %land.lhs.true387

land.lhs.true387:		; preds = %if.end384
	unreachable

if.end394:		; preds = %if.end384
	br i1 undef, label %if.end498, label %land.lhs.true399

land.lhs.true399:		; preds = %if.end394
	br i1 undef, label %if.end498, label %if.then406

if.then406:		; preds = %land.lhs.true399
	unreachable

if.end498:		; preds = %land.lhs.true399, %if.end394
	br i1 undef, label %if.end514, label %if.then503

if.then503:		; preds = %if.end498
	unreachable

if.end514:		; preds = %if.end498
	br i1 undef, label %if.end585, label %if.then520

if.then520:		; preds = %if.end514
	br i1 undef, label %lor.lhs.false547, label %if.then560

lor.lhs.false547:		; preds = %if.then520
	unreachable

if.then560:		; preds = %if.then520
	br i1 undef, label %if.end585, label %land.lhs.true566

land.lhs.true566:		; preds = %if.then560
	br i1 undef, label %if.end585, label %if.then573

if.then573:		; preds = %land.lhs.true566
	unreachable

if.end585:		; preds = %land.lhs.true566, %if.then560, %if.end514
	br i1 undef, label %cond.true593, label %cond.false599

cond.true593:		; preds = %if.end585
	unreachable

cond.false599:		; preds = %if.end585
	br i1 undef, label %if.end647, label %if.then621

if.then621:		; preds = %cond.false599
	br i1 undef, label %cond.true624, label %cond.false630

cond.true624:		; preds = %if.then621
	br label %if.end647

cond.false630:		; preds = %if.then621
	unreachable

if.end647:		; preds = %cond.true624, %cond.false599
	br i1 undef, label %if.end723, label %if.then701

if.then701:		; preds = %if.end647
	br label %if.end723

if.end723:		; preds = %if.then701, %if.end647
	br i1 undef, label %if.else1090, label %if.then729

if.then729:		; preds = %if.end723
	br i1 undef, label %if.end887, label %if.then812

if.then812:		; preds = %if.then729
	unreachable

if.end887:		; preds = %if.then729
	br i1 undef, label %if.end972, label %if.then893

if.then893:		; preds = %if.end887
	br i1 undef, label %if.end919, label %if.then903

if.then903:		; preds = %if.then893
	unreachable

if.end919:		; preds = %if.then893
	br label %if.end972

if.end972:		; preds = %if.end919, %if.end887
	%sline.0 = phi ptr [ undef, %if.end919 ], [ null, %if.end887 ]		; <ptr> [#uses=5]
	%bcs.0 = phi i32 [ undef, %if.end919 ], [ 0, %if.end887 ]		; <i32> [#uses=5]
	br i1 undef, label %if.end1146, label %land.lhs.true975

land.lhs.true975:		; preds = %if.end972
	br i1 undef, label %if.end1146, label %if.then980

if.then980:		; preds = %land.lhs.true975
	br i1 undef, label %cond.false1025, label %cond.false1004

cond.false1004:		; preds = %if.then980
	unreachable

cond.false1025:		; preds = %if.then980
	br i1 undef, label %if.end1146, label %if.then1071

if.then1071:		; preds = %cond.false1025
	br i1 undef, label %if.then1074, label %if.end1081

if.then1074:		; preds = %if.then1071
	br label %if.end1081

if.end1081:		; preds = %if.then1074, %if.then1071
	%call1083 = call ptr @patcompile(ptr undef, i32 0, ptr null) nounwind ssp		; <ptr> [#uses=2]
	br i1 undef, label %if.end1146, label %if.then1086

if.then1086:		; preds = %if.end1081
	br label %if.end1146

if.else1090:		; preds = %if.end723
	br i1 undef, label %if.end1146, label %land.lhs.true1093

land.lhs.true1093:		; preds = %if.else1090
	br i1 undef, label %if.end1146, label %if.then1098

if.then1098:		; preds = %land.lhs.true1093
	unreachable

if.end1146:		; preds = %land.lhs.true1093, %if.else1090, %if.then1086, %if.end1081, %cond.false1025, %land.lhs.true975, %if.end972
	%cp.0 = phi ptr [ %call1083, %if.then1086 ], [ null, %if.end972 ], [ null, %land.lhs.true975 ], [ null, %cond.false1025 ], [ %call1083, %if.end1081 ], [ null, %if.else1090 ], [ null, %land.lhs.true1093 ]		; <ptr> [#uses=1]
	%sline.1 = phi ptr [ %sline.0, %if.then1086 ], [ %sline.0, %if.end972 ], [ %sline.0, %land.lhs.true975 ], [ %sline.0, %cond.false1025 ], [ %sline.0, %if.end1081 ], [ null, %if.else1090 ], [ null, %land.lhs.true1093 ]		; <ptr> [#uses=1]
	%bcs.1 = phi i32 [ %bcs.0, %if.then1086 ], [ %bcs.0, %if.end972 ], [ %bcs.0, %land.lhs.true975 ], [ %bcs.0, %cond.false1025 ], [ %bcs.0, %if.end1081 ], [ 0, %if.else1090 ], [ 0, %land.lhs.true1093 ]		; <i32> [#uses=1]
	br i1 undef, label %if.end1307, label %do.body1270

do.body1270:		; preds = %if.end1146
	unreachable

if.end1307:		; preds = %if.end1146
	br i1 undef, label %if.end1318, label %if.then1312

if.then1312:		; preds = %if.end1307
	unreachable

if.end1318:		; preds = %if.end1307
	br i1 undef, label %for.cond1330.preheader, label %if.then1323

if.then1323:		; preds = %if.end1318
	unreachable

for.cond1330.preheader:		; preds = %if.end1318
	%call1587 = call ptr @comp_match(ptr undef, ptr undef, ptr undef, ptr %cp.0, ptr undef, i32 0, ptr undef, i32 0, ptr undef, i32 %bcs.1, ptr undef) nounwind ssp		; <ptr> [#uses=0]
	%call1667 = call ptr @add_match_data(i32 0, ptr undef, ptr undef, ptr undef, ptr undef, ptr null, ptr undef, ptr undef, ptr undef, ptr undef, ptr null, ptr undef, ptr %sline.1, ptr undef, i32 undef, i32 undef) ssp		; <ptr> [#uses=0]
	unreachable
}

declare ptr @patcompile(ptr, i32, ptr) ssp

declare ptr @comp_match(ptr, ptr, ptr, ptr, ptr, i32, ptr, i32, ptr, i32, ptr) ssp

declare ptr @add_match_data(i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32) nounwind ssp
