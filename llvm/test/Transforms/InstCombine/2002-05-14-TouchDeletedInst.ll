; This testcase, obviously distilled from a large program (bzip2 from
; Specint2000) caused instcombine to fail because it got the same instruction
; on it's worklist more than once (which is ok), but then deleted the
; instruction.  Since the inst stayed on the worklist, as soon as it came back
; up to be processed, bad things happened, and opt asserted.
;
; RUN: llvm-as < %s | opt -instcombine
;
; END.

@.LC0 = internal global [21 x i8] c"hbMakeCodeLengths(1)\00"		; <[21 x i8]*> [#uses=1]
@.LC1 = internal global [21 x i8] c"hbMakeCodeLengths(2)\00"		; <[21 x i8]*> [#uses=1]

define void @hbMakeCodeLengths(i8* %len, i32* %freq, i32 %alphaSize, i32 %maxLen) {
bb0:
	%len.upgrd.1 = alloca i8*		; <i8**> [#uses=2]
	store i8* %len, i8** %len.upgrd.1
	%freq.upgrd.2 = alloca i32*		; <i32**> [#uses=2]
	store i32* %freq, i32** %freq.upgrd.2
	%alphaSize.upgrd.3 = alloca i32		; <i32*> [#uses=2]
	store i32 %alphaSize, i32* %alphaSize.upgrd.3
	%maxLen.upgrd.4 = alloca i32		; <i32*> [#uses=2]
	store i32 %maxLen, i32* %maxLen.upgrd.4
	%heap = alloca i32, i32 260		; <i32*> [#uses=27]
	%weight = alloca i32, i32 516		; <i32*> [#uses=18]
	%parent = alloca i32, i32 516		; <i32*> [#uses=7]
	br label %bb1

bb1:		; preds = %bb0
	%reg107 = load i8** %len.upgrd.1		; <i8*> [#uses=1]
	%reg108 = load i32** %freq.upgrd.2		; <i32*> [#uses=1]
	%reg109 = load i32* %alphaSize.upgrd.3		; <i32> [#uses=10]
	%reg110 = load i32* %maxLen.upgrd.4		; <i32> [#uses=1]
	%cond747 = icmp sge i32 0, %reg109		; <i1> [#uses=1]
	br i1 %cond747, label %bb6, label %bb2

bb2:		; preds = %bb5, %bb1
	%reg591 = phi i32 [ %reg594, %bb5 ], [ 0, %bb1 ]		; <i32> [#uses=3]
	%reg591-idxcast1 = bitcast i32 %reg591 to i32		; <i32> [#uses=1]
	%reg591-idxcast1-offset = add i32 %reg591-idxcast1, 1		; <i32> [#uses=1]
	%reg591-idxcast1-offset.upgrd.5 = zext i32 %reg591-idxcast1-offset to i64		; <i64> [#uses=1]
	%reg126 = getelementptr i32* %weight, i64 %reg591-idxcast1-offset.upgrd.5		; <i32*> [#uses=1]
	%reg591-idxcast = sext i32 %reg591 to i64		; <i64> [#uses=1]
	%reg132 = getelementptr i32* %reg108, i64 %reg591-idxcast		; <i32*> [#uses=1]
	%reg133 = load i32* %reg132		; <i32> [#uses=2]
	%cond748 = icmp eq i32 %reg133, 0		; <i1> [#uses=1]
	br i1 %cond748, label %bb4, label %bb3

bb3:		; preds = %bb2
	%reg127 = shl i32 %reg133, 8		; <i32> [#uses=1]
	br label %bb5

bb4:		; preds = %bb2
	br label %bb5

bb5:		; preds = %bb4, %bb3
	%reg593 = phi i32 [ 256, %bb4 ], [ %reg127, %bb3 ]		; <i32> [#uses=1]
	store i32 %reg593, i32* %reg126
	%reg594 = add i32 %reg591, 1		; <i32> [#uses=2]
	%cond749 = icmp slt i32 %reg594, %reg109		; <i1> [#uses=1]
	br i1 %cond749, label %bb2, label %bb6

bb6:		; preds = %bb43, %bb41, %bb5, %bb1
	store i32 0, i32* %heap
	store i32 0, i32* %weight
	store i32 -2, i32* %parent
	%cond750 = icmp sgt i32 1, %reg109		; <i1> [#uses=1]
	br i1 %cond750, label %bb11, label %bb7

bb7:		; preds = %bb10, %bb6
	%reg597 = phi i32 [ %reg598, %bb10 ], [ 0, %bb6 ]		; <i32> [#uses=5]
	%reg597-casted = bitcast i32 %reg597 to i32		; <i32> [#uses=1]
	%reg596 = add i32 %reg597-casted, 1		; <i32> [#uses=3]
	%reg597-offset = add i32 %reg597, 1		; <i32> [#uses=1]
	%reg597-offset.upgrd.6 = zext i32 %reg597-offset to i64		; <i64> [#uses=1]
	%reg149 = getelementptr i32* %parent, i64 %reg597-offset.upgrd.6		; <i32*> [#uses=1]
	store i32 -1, i32* %reg149
	%reg598 = add i32 %reg597, 1		; <i32> [#uses=3]
	%reg597-offset1 = add i32 %reg597, 1		; <i32> [#uses=1]
	%reg597-offset1.upgrd.7 = zext i32 %reg597-offset1 to i64		; <i64> [#uses=1]
	%reg157 = getelementptr i32* %heap, i64 %reg597-offset1.upgrd.7		; <i32*> [#uses=1]
	store i32 %reg596, i32* %reg157
	br label %bb9

bb8:		; preds = %bb9
	%reg599 = zext i32 %reg599.upgrd.8 to i64		; <i64> [#uses=1]
	%reg198 = getelementptr i32* %heap, i64 %reg599		; <i32*> [#uses=1]
	store i32 %reg182, i32* %reg198
	%cast938 = bitcast i32 %reg174 to i32		; <i32> [#uses=1]
	br label %bb9

bb9:		; preds = %bb8, %bb7
	%reg599.upgrd.8 = phi i32 [ %cast938, %bb8 ], [ %reg598, %bb7 ]		; <i32> [#uses=3]
	%cast807 = bitcast i32 %reg599.upgrd.8 to i32		; <i32> [#uses=1]
	%reg597-offset2 = add i32 %reg597, 1		; <i32> [#uses=1]
	%reg597-offset2.upgrd.9 = zext i32 %reg597-offset2 to i64		; <i64> [#uses=1]
	%reg173 = getelementptr i32* %weight, i64 %reg597-offset2.upgrd.9		; <i32*> [#uses=1]
	%reg174 = ashr i32 %cast807, 1		; <i32> [#uses=2]
	%reg174-idxcast = bitcast i32 %reg174 to i32		; <i32> [#uses=1]
	zext i32 %reg174-idxcast to i64		; <i64>:0 [#uses=1]
	%reg181 = getelementptr i32* %heap, i64 %0		; <i32*> [#uses=1]
	%reg182 = load i32* %reg181		; <i32> [#uses=2]
	%reg182-idxcast = bitcast i32 %reg182 to i32		; <i32> [#uses=1]
	zext i32 %reg182-idxcast to i64		; <i64>:1 [#uses=1]
	%reg189 = getelementptr i32* %weight, i64 %1		; <i32*> [#uses=1]
	%reg190 = load i32* %reg173		; <i32> [#uses=1]
	%reg191 = load i32* %reg189		; <i32> [#uses=1]
	%cond751 = icmp slt i32 %reg190, %reg191		; <i1> [#uses=1]
	br i1 %cond751, label %bb8, label %bb10

bb10:		; preds = %bb9
	zext i32 %reg599.upgrd.8 to i64		; <i64>:2 [#uses=1]
	%reg214 = getelementptr i32* %heap, i64 %2		; <i32*> [#uses=1]
	store i32 %reg596, i32* %reg214
	%reg601 = add i32 %reg596, 1		; <i32> [#uses=1]
	%cond752 = icmp sle i32 %reg601, %reg109		; <i1> [#uses=1]
	br i1 %cond752, label %bb7, label %bb11

bb11:		; preds = %bb10, %bb6
	%reg602 = phi i32 [ %reg598, %bb10 ], [ 0, %bb6 ]		; <i32> [#uses=3]
	%cast819 = bitcast i32 %reg602 to i32		; <i32> [#uses=1]
	%cast818 = bitcast i32 %reg602 to i32		; <i32> [#uses=1]
	%cond753 = icmp sle i32 %cast818, 259		; <i1> [#uses=1]
	br i1 %cond753, label %bb13, label %bb12

bb12:		; preds = %bb11
	zext i32 0 to i64		; <i64>:3 [#uses=1]
	zext i32 0 to i64		; <i64>:4 [#uses=1]
	%cast784 = getelementptr [21 x i8]* @.LC0, i64 %3, i64 %4		; <i8*> [#uses=1]
	call void @panic( i8* %cast784 )
	br label %bb13

bb13:		; preds = %bb12, %bb11
	%cond754 = icmp sle i32 %cast819, 1		; <i1> [#uses=1]
	%cast918 = bitcast i32 %reg109 to i32		; <i32> [#uses=1]
	%cast940 = bitcast i32 %reg602 to i32		; <i32> [#uses=1]
	%cast942 = bitcast i32 %reg109 to i32		; <i32> [#uses=1]
	br i1 %cond754, label %bb32, label %bb14

bb14:		; preds = %bb31, %bb13
	%cann-indvar1 = phi i32 [ 0, %bb13 ], [ %add1-indvar1, %bb31 ]		; <i32> [#uses=3]
	%cann-indvar1-casted = bitcast i32 %cann-indvar1 to i32		; <i32> [#uses=1]
	%reg603-scale = mul i32 %cann-indvar1-casted, -1		; <i32> [#uses=1]
	%reg603 = add i32 %reg603-scale, %cast940		; <i32> [#uses=4]
	%reg604 = add i32 %cann-indvar1, %cast942		; <i32> [#uses=4]
	%add1-indvar1 = add i32 %cann-indvar1, 1		; <i32> [#uses=1]
	zext i32 1 to i64		; <i64>:5 [#uses=1]
	%reg7551 = getelementptr i32* %heap, i64 %5		; <i32*> [#uses=1]
	%reg113 = load i32* %reg7551		; <i32> [#uses=2]
	%reg603-idxcast = bitcast i32 %reg603 to i32		; <i32> [#uses=1]
	zext i32 %reg603-idxcast to i64		; <i64>:6 [#uses=1]
	%reg222 = getelementptr i32* %heap, i64 %6		; <i32*> [#uses=1]
	%reg223 = load i32* %reg222		; <i32> [#uses=1]
	zext i32 1 to i64		; <i64>:7 [#uses=1]
	%reg7561 = getelementptr i32* %heap, i64 %7		; <i32*> [#uses=1]
	store i32 %reg223, i32* %reg7561
	%reg605 = add i32 %reg603, -1		; <i32> [#uses=4]
	zext i32 1 to i64		; <i64>:8 [#uses=1]
	%reg757 = getelementptr i32* %heap, i64 %8		; <i32*> [#uses=1]
	%reg226 = load i32* %reg757		; <i32> [#uses=2]
	%cond758 = icmp sgt i32 2, %reg605		; <i1> [#uses=1]
	br i1 %cond758, label %bb20, label %bb15

bb15:		; preds = %bb19, %bb14
	%reg606 = phi i32 [ %reg611, %bb19 ], [ 2, %bb14 ]		; <i32> [#uses=6]
	%reg607 = phi i32 [ %reg609, %bb19 ], [ 1, %bb14 ]		; <i32> [#uses=2]
	%cond759 = icmp sge i32 %reg606, %reg605		; <i1> [#uses=1]
	br i1 %cond759, label %bb18, label %bb16

bb16:		; preds = %bb15
	%reg606-idxcast = bitcast i32 %reg606 to i32		; <i32> [#uses=1]
	%reg606-idxcast-offset = add i32 %reg606-idxcast, 1		; <i32> [#uses=1]
	zext i32 %reg606-idxcast-offset to i64		; <i64>:9 [#uses=1]
	%reg241 = getelementptr i32* %heap, i64 %9		; <i32*> [#uses=1]
	%reg242 = load i32* %reg241		; <i32> [#uses=1]
	%reg242-idxcast = bitcast i32 %reg242 to i32		; <i32> [#uses=1]
	zext i32 %reg242-idxcast to i64		; <i64>:10 [#uses=1]
	%reg249 = getelementptr i32* %weight, i64 %10		; <i32*> [#uses=1]
	%reg606-idxcast1 = bitcast i32 %reg606 to i32		; <i32> [#uses=1]
	zext i32 %reg606-idxcast1 to i64		; <i64>:11 [#uses=1]
	%reg256 = getelementptr i32* %heap, i64 %11		; <i32*> [#uses=1]
	%reg257 = load i32* %reg256		; <i32> [#uses=1]
	%reg257-idxcast = bitcast i32 %reg257 to i32		; <i32> [#uses=1]
	zext i32 %reg257-idxcast to i64		; <i64>:12 [#uses=1]
	%reg264 = getelementptr i32* %weight, i64 %12		; <i32*> [#uses=1]
	%reg265 = load i32* %reg249		; <i32> [#uses=1]
	%reg266 = load i32* %reg264		; <i32> [#uses=1]
	%cond760 = icmp sge i32 %reg265, %reg266		; <i1> [#uses=1]
	br i1 %cond760, label %bb18, label %bb17

bb17:		; preds = %bb16
	%reg608 = add i32 %reg606, 1		; <i32> [#uses=1]
	br label %bb18

bb18:		; preds = %bb17, %bb16, %bb15
	%reg609 = phi i32 [ %reg608, %bb17 ], [ %reg606, %bb16 ], [ %reg606, %bb15 ]		; <i32> [#uses=4]
	%reg226-idxcast = bitcast i32 %reg226 to i32		; <i32> [#uses=1]
	zext i32 %reg226-idxcast to i64		; <i64>:13 [#uses=1]
	%reg273 = getelementptr i32* %weight, i64 %13		; <i32*> [#uses=1]
	%reg609-idxcast = bitcast i32 %reg609 to i32		; <i32> [#uses=1]
	zext i32 %reg609-idxcast to i64		; <i64>:14 [#uses=1]
	%reg280 = getelementptr i32* %heap, i64 %14		; <i32*> [#uses=1]
	%reg281 = load i32* %reg280		; <i32> [#uses=2]
	%reg281-idxcast = bitcast i32 %reg281 to i32		; <i32> [#uses=1]
	zext i32 %reg281-idxcast to i64		; <i64>:15 [#uses=1]
	%reg288 = getelementptr i32* %weight, i64 %15		; <i32*> [#uses=1]
	%reg289 = load i32* %reg273		; <i32> [#uses=1]
	%reg290 = load i32* %reg288		; <i32> [#uses=1]
	%cond761 = icmp slt i32 %reg289, %reg290		; <i1> [#uses=1]
	br i1 %cond761, label %bb20, label %bb19

bb19:		; preds = %bb18
	%reg607-idxcast = bitcast i32 %reg607 to i32		; <i32> [#uses=1]
	zext i32 %reg607-idxcast to i64		; <i64>:16 [#uses=1]
	%reg297 = getelementptr i32* %heap, i64 %16		; <i32*> [#uses=1]
	store i32 %reg281, i32* %reg297
	%reg611 = shl i32 %reg609, 1		; <i32> [#uses=2]
	%cond762 = icmp sle i32 %reg611, %reg605		; <i1> [#uses=1]
	br i1 %cond762, label %bb15, label %bb20

bb20:		; preds = %bb19, %bb18, %bb14
	%reg612 = phi i32 [ %reg609, %bb19 ], [ %reg607, %bb18 ], [ 1, %bb14 ]		; <i32> [#uses=1]
	%reg612-idxcast = bitcast i32 %reg612 to i32		; <i32> [#uses=1]
	zext i32 %reg612-idxcast to i64		; <i64>:17 [#uses=1]
	%reg312 = getelementptr i32* %heap, i64 %17		; <i32*> [#uses=1]
	store i32 %reg226, i32* %reg312
	zext i32 1 to i64		; <i64>:18 [#uses=1]
	%reg7631 = getelementptr i32* %heap, i64 %18		; <i32*> [#uses=1]
	%reg114 = load i32* %reg7631		; <i32> [#uses=2]
	%reg603-idxcast1 = bitcast i32 %reg603 to i32		; <i32> [#uses=1]
	%reg603-idxcast1-offset = add i32 %reg603-idxcast1, 1073741823		; <i32> [#uses=1]
	zext i32 %reg603-idxcast1-offset to i64		; <i64>:19 [#uses=1]
	%reg319 = getelementptr i32* %heap, i64 %19		; <i32*> [#uses=1]
	%reg320 = load i32* %reg319		; <i32> [#uses=1]
	zext i32 1 to i64		; <i64>:20 [#uses=1]
	%reg7641 = getelementptr i32* %heap, i64 %20		; <i32*> [#uses=1]
	store i32 %reg320, i32* %reg7641
	%reg613 = add i32 %reg605, -1		; <i32> [#uses=4]
	zext i32 1 to i64		; <i64>:21 [#uses=1]
	%reg765 = getelementptr i32* %heap, i64 %21		; <i32*> [#uses=1]
	%reg323 = load i32* %reg765		; <i32> [#uses=2]
	%cond766 = icmp sgt i32 2, %reg613		; <i1> [#uses=1]
	br i1 %cond766, label %bb26, label %bb21

bb21:		; preds = %bb25, %bb20
	%reg614 = phi i32 [ %reg619, %bb25 ], [ 2, %bb20 ]		; <i32> [#uses=6]
	%reg615 = phi i32 [ %reg617, %bb25 ], [ 1, %bb20 ]		; <i32> [#uses=2]
	%cond767 = icmp sge i32 %reg614, %reg613		; <i1> [#uses=1]
	br i1 %cond767, label %bb24, label %bb22

bb22:		; preds = %bb21
	%reg614-idxcast = bitcast i32 %reg614 to i32		; <i32> [#uses=1]
	%reg614-idxcast-offset = add i32 %reg614-idxcast, 1		; <i32> [#uses=1]
	zext i32 %reg614-idxcast-offset to i64		; <i64>:22 [#uses=1]
	%reg338 = getelementptr i32* %heap, i64 %22		; <i32*> [#uses=1]
	%reg339 = load i32* %reg338		; <i32> [#uses=1]
	%reg339-idxcast = bitcast i32 %reg339 to i32		; <i32> [#uses=1]
	zext i32 %reg339-idxcast to i64		; <i64>:23 [#uses=1]
	%reg346 = getelementptr i32* %weight, i64 %23		; <i32*> [#uses=1]
	%reg614-idxcast1 = bitcast i32 %reg614 to i32		; <i32> [#uses=1]
	zext i32 %reg614-idxcast1 to i64		; <i64>:24 [#uses=1]
	%reg353 = getelementptr i32* %heap, i64 %24		; <i32*> [#uses=1]
	%reg354 = load i32* %reg353		; <i32> [#uses=1]
	%reg354-idxcast = bitcast i32 %reg354 to i32		; <i32> [#uses=1]
	zext i32 %reg354-idxcast to i64		; <i64>:25 [#uses=1]
	%reg361 = getelementptr i32* %weight, i64 %25		; <i32*> [#uses=1]
	%reg362 = load i32* %reg346		; <i32> [#uses=1]
	%reg363 = load i32* %reg361		; <i32> [#uses=1]
	%cond768 = icmp sge i32 %reg362, %reg363		; <i1> [#uses=1]
	br i1 %cond768, label %bb24, label %bb23

bb23:		; preds = %bb22
	%reg616 = add i32 %reg614, 1		; <i32> [#uses=1]
	br label %bb24

bb24:		; preds = %bb23, %bb22, %bb21
	%reg617 = phi i32 [ %reg616, %bb23 ], [ %reg614, %bb22 ], [ %reg614, %bb21 ]		; <i32> [#uses=4]
	%reg323-idxcast = bitcast i32 %reg323 to i32		; <i32> [#uses=1]
	zext i32 %reg323-idxcast to i64		; <i64>:26 [#uses=1]
	%reg370 = getelementptr i32* %weight, i64 %26		; <i32*> [#uses=1]
	%reg617-idxcast = bitcast i32 %reg617 to i32		; <i32> [#uses=1]
	zext i32 %reg617-idxcast to i64		; <i64>:27 [#uses=1]
	%reg377 = getelementptr i32* %heap, i64 %27		; <i32*> [#uses=1]
	%reg378 = load i32* %reg377		; <i32> [#uses=2]
	%reg378-idxcast = bitcast i32 %reg378 to i32		; <i32> [#uses=1]
	zext i32 %reg378-idxcast to i64		; <i64>:28 [#uses=1]
	%reg385 = getelementptr i32* %weight, i64 %28		; <i32*> [#uses=1]
	%reg386 = load i32* %reg370		; <i32> [#uses=1]
	%reg387 = load i32* %reg385		; <i32> [#uses=1]
	%cond769 = icmp slt i32 %reg386, %reg387		; <i1> [#uses=1]
	br i1 %cond769, label %bb26, label %bb25

bb25:		; preds = %bb24
	%reg615-idxcast = bitcast i32 %reg615 to i32		; <i32> [#uses=1]
	zext i32 %reg615-idxcast to i64		; <i64>:29 [#uses=1]
	%reg394 = getelementptr i32* %heap, i64 %29		; <i32*> [#uses=1]
	store i32 %reg378, i32* %reg394
	%reg619 = shl i32 %reg617, 1		; <i32> [#uses=2]
	%cond770 = icmp sle i32 %reg619, %reg613		; <i1> [#uses=1]
	br i1 %cond770, label %bb21, label %bb26

bb26:		; preds = %bb25, %bb24, %bb20
	%reg620 = phi i32 [ %reg617, %bb25 ], [ %reg615, %bb24 ], [ 1, %bb20 ]		; <i32> [#uses=1]
	%reg620-idxcast = bitcast i32 %reg620 to i32		; <i32> [#uses=1]
	zext i32 %reg620-idxcast to i64		; <i64>:30 [#uses=1]
	%reg409 = getelementptr i32* %heap, i64 %30		; <i32*> [#uses=1]
	store i32 %reg323, i32* %reg409
	%reg621 = add i32 %reg604, 1		; <i32> [#uses=5]
	%reg113-idxcast = bitcast i32 %reg113 to i32		; <i32> [#uses=1]
	zext i32 %reg113-idxcast to i64		; <i64>:31 [#uses=1]
	%reg416 = getelementptr i32* %parent, i64 %31		; <i32*> [#uses=1]
	%reg114-idxcast = bitcast i32 %reg114 to i32		; <i32> [#uses=1]
	zext i32 %reg114-idxcast to i64		; <i64>:32 [#uses=1]
	%reg423 = getelementptr i32* %parent, i64 %32		; <i32*> [#uses=1]
	%cast889 = bitcast i32 %reg621 to i32		; <i32> [#uses=1]
	store i32 %cast889, i32* %reg423
	%cast890 = bitcast i32 %reg621 to i32		; <i32> [#uses=1]
	store i32 %cast890, i32* %reg416
	%reg604-offset = add i32 %reg604, 1		; <i32> [#uses=1]
	zext i32 %reg604-offset to i64		; <i64>:33 [#uses=1]
	%reg431 = getelementptr i32* %weight, i64 %33		; <i32*> [#uses=1]
	%reg113-idxcast2 = bitcast i32 %reg113 to i32		; <i32> [#uses=1]
	zext i32 %reg113-idxcast2 to i64		; <i64>:34 [#uses=1]
	%reg4381 = getelementptr i32* %weight, i64 %34		; <i32*> [#uses=1]
	%reg439 = load i32* %reg4381		; <i32> [#uses=2]
	%reg440 = and i32 %reg439, -256		; <i32> [#uses=1]
	%reg114-idxcast2 = bitcast i32 %reg114 to i32		; <i32> [#uses=1]
	zext i32 %reg114-idxcast2 to i64		; <i64>:35 [#uses=1]
	%reg4471 = getelementptr i32* %weight, i64 %35		; <i32*> [#uses=1]
	%reg448 = load i32* %reg4471		; <i32> [#uses=2]
	%reg449 = and i32 %reg448, -256		; <i32> [#uses=1]
	%reg450 = add i32 %reg440, %reg449		; <i32> [#uses=1]
	%reg460 = and i32 %reg439, 255		; <i32> [#uses=2]
	%reg451 = and i32 %reg448, 255		; <i32> [#uses=2]
	%cond771 = icmp sge i32 %reg451, %reg460		; <i1> [#uses=1]
	br i1 %cond771, label %bb28, label %bb27

bb27:		; preds = %bb26
	br label %bb28

bb28:		; preds = %bb27, %bb26
	%reg623 = phi i32 [ %reg460, %bb27 ], [ %reg451, %bb26 ]		; <i32> [#uses=1]
	%reg469 = add i32 %reg623, 1		; <i32> [#uses=1]
	%reg470 = or i32 %reg450, %reg469		; <i32> [#uses=1]
	store i32 %reg470, i32* %reg431
	%reg604-offset1 = add i32 %reg604, 1		; <i32> [#uses=1]
	zext i32 %reg604-offset1 to i64		; <i64>:36 [#uses=1]
	%reg4771 = getelementptr i32* %parent, i64 %36		; <i32*> [#uses=1]
	store i32 -1, i32* %reg4771
	%reg624 = add i32 %reg613, 1		; <i32> [#uses=2]
	%reg603-idxcast2 = bitcast i32 %reg603 to i32		; <i32> [#uses=1]
	%reg603-idxcast2-offset = add i32 %reg603-idxcast2, 1073741823		; <i32> [#uses=1]
	zext i32 %reg603-idxcast2-offset to i64		; <i64>:37 [#uses=1]
	%reg485 = getelementptr i32* %heap, i64 %37		; <i32*> [#uses=1]
	%cast902 = bitcast i32 %reg621 to i32		; <i32> [#uses=1]
	store i32 %cast902, i32* %reg485
	br label %bb30

bb29:		; preds = %bb30
	%reg625-idxcast = bitcast i32 %reg625 to i32		; <i32> [#uses=1]
	zext i32 %reg625-idxcast to i64		; <i64>:38 [#uses=1]
	%reg526 = getelementptr i32* %heap, i64 %38		; <i32*> [#uses=1]
	store i32 %reg510, i32* %reg526
	br label %bb30

bb30:		; preds = %bb29, %bb28
	%reg625 = phi i32 [ %reg502, %bb29 ], [ %reg624, %bb28 ]		; <i32> [#uses=3]
	%reg604-offset2 = add i32 %reg604, 1		; <i32> [#uses=1]
	zext i32 %reg604-offset2 to i64		; <i64>:39 [#uses=1]
	%reg501 = getelementptr i32* %weight, i64 %39		; <i32*> [#uses=1]
	%reg502 = ashr i32 %reg625, 1		; <i32> [#uses=2]
	%reg502-idxcast = bitcast i32 %reg502 to i32		; <i32> [#uses=1]
	zext i32 %reg502-idxcast to i64		; <i64>:40 [#uses=1]
	%reg509 = getelementptr i32* %heap, i64 %40		; <i32*> [#uses=1]
	%reg510 = load i32* %reg509		; <i32> [#uses=2]
	%reg510-idxcast = bitcast i32 %reg510 to i32		; <i32> [#uses=1]
	zext i32 %reg510-idxcast to i64		; <i64>:41 [#uses=1]
	%reg517 = getelementptr i32* %weight, i64 %41		; <i32*> [#uses=1]
	%reg518 = load i32* %reg501		; <i32> [#uses=1]
	%reg519 = load i32* %reg517		; <i32> [#uses=1]
	%cond772 = icmp slt i32 %reg518, %reg519		; <i1> [#uses=1]
	br i1 %cond772, label %bb29, label %bb31

bb31:		; preds = %bb30
	%reg625-idxcast1 = bitcast i32 %reg625 to i32		; <i32> [#uses=1]
	zext i32 %reg625-idxcast1 to i64		; <i64>:42 [#uses=1]
	%reg542 = getelementptr i32* %heap, i64 %42		; <i32*> [#uses=1]
	%cast916 = bitcast i32 %reg621 to i32		; <i32> [#uses=1]
	store i32 %cast916, i32* %reg542
	%cond773 = icmp sgt i32 %reg624, 1		; <i1> [#uses=1]
	br i1 %cond773, label %bb14, label %bb32

bb32:		; preds = %bb31, %bb13
	%reg627 = phi i32 [ %reg621, %bb31 ], [ %cast918, %bb13 ]		; <i32> [#uses=1]
	%cast919 = bitcast i32 %reg627 to i32		; <i32> [#uses=1]
	%cond774 = icmp sle i32 %cast919, 515		; <i1> [#uses=1]
	br i1 %cond774, label %bb34, label %bb33

bb33:		; preds = %bb32
	zext i32 0 to i64		; <i64>:43 [#uses=1]
	zext i32 0 to i64		; <i64>:44 [#uses=1]
	%cast785 = getelementptr [21 x i8]* @.LC1, i64 %43, i64 %44		; <i8*> [#uses=1]
	call void @panic( i8* %cast785 )
	br label %bb34

bb34:		; preds = %bb33, %bb32
	%cond775 = icmp sgt i32 1, %reg109		; <i1> [#uses=1]
	br i1 %cond775, label %bb40, label %bb35

bb35:		; preds = %bb39, %bb34
	%reg629 = phi i8 [ %reg639, %bb39 ], [ 0, %bb34 ]		; <i8> [#uses=1]
	%cann-indvar = phi i32 [ 0, %bb34 ], [ %add1-indvar, %bb39 ]		; <i32> [#uses=4]
	%cann-indvar-casted = bitcast i32 %cann-indvar to i32		; <i32> [#uses=1]
	%reg630 = add i32 %cann-indvar-casted, 1		; <i32> [#uses=2]
	%add1-indvar = add i32 %cann-indvar, 1		; <i32> [#uses=1]
	%cann-indvar-offset1 = add i32 %cann-indvar, 1		; <i32> [#uses=1]
	zext i32 %cann-indvar-offset1 to i64		; <i64>:45 [#uses=1]
	%reg589 = getelementptr i32* %parent, i64 %45		; <i32*> [#uses=1]
	%reg590 = load i32* %reg589		; <i32> [#uses=1]
	%cond776 = icmp slt i32 %reg590, 0		; <i1> [#uses=1]
	%parent-idxcast = ptrtoint i32* %parent to i32		; <i32> [#uses=1]
	%cast948 = bitcast i32 %reg630 to i32		; <i32> [#uses=1]
	br i1 %cond776, label %bb37, label %bb36

bb36:		; preds = %bb36, %bb35
	%reg632 = phi i32 [ %reg634, %bb36 ], [ %cast948, %bb35 ]		; <i32> [#uses=1]
	%reg633 = phi i32 [ %reg635, %bb36 ], [ 0, %bb35 ]		; <i32> [#uses=3]
	%reg633-casted = inttoptr i32 %reg633 to i8*		; <i8*> [#uses=0]
	%reg631-scale = mul i32 %reg633, 0		; <i32> [#uses=1]
	%reg631-scale.upgrd.10 = inttoptr i32 %reg631-scale to i8*		; <i8*> [#uses=1]
	zext i32 %parent-idxcast to i64		; <i64>:46 [#uses=1]
	%reg6311 = getelementptr i8* %reg631-scale.upgrd.10, i64 %46		; <i8*> [#uses=2]
	%reg632-scale = mul i32 %reg632, 4		; <i32> [#uses=1]
	zext i32 %reg632-scale to i64		; <i64>:47 [#uses=1]
	%reg5581 = getelementptr i8* %reg6311, i64 %47		; <i8*> [#uses=1]
	%cast924 = bitcast i8* %reg5581 to i32*		; <i32*> [#uses=1]
	%reg634 = load i32* %cast924		; <i32> [#uses=2]
	%reg635 = add i32 %reg633, 1		; <i32> [#uses=2]
	%reg634-scale = mul i32 %reg634, 4		; <i32> [#uses=1]
	zext i32 %reg634-scale to i64		; <i64>:48 [#uses=1]
	%reg5501 = getelementptr i8* %reg6311, i64 %48		; <i8*> [#uses=1]
	%cast925 = bitcast i8* %reg5501 to i32*		; <i32*> [#uses=1]
	%reg551 = load i32* %cast925		; <i32> [#uses=1]
	%cond777 = icmp sge i32 %reg551, 0		; <i1> [#uses=1]
	br i1 %cond777, label %bb36, label %bb37

bb37:		; preds = %bb36, %bb35
	%reg637 = phi i32 [ %reg635, %bb36 ], [ 0, %bb35 ]		; <i32> [#uses=2]
	%cast928 = bitcast i32 %reg637 to i32		; <i32> [#uses=1]
	%cann-indvar-offset = add i32 %cann-indvar, 1		; <i32> [#uses=1]
	zext i32 %cann-indvar-offset to i64		; <i64>:49 [#uses=1]
	%reg561 = getelementptr i8* %reg107, i64 %49		; <i8*> [#uses=1]
	zext i32 -1 to i64		; <i64>:50 [#uses=1]
	%reg778 = getelementptr i8* %reg561, i64 %50		; <i8*> [#uses=1]
	%cast788 = trunc i32 %reg637 to i8		; <i8> [#uses=1]
	store i8 %cast788, i8* %reg778
	%cond779 = icmp sle i32 %cast928, %reg110		; <i1> [#uses=1]
	br i1 %cond779, label %bb39, label %bb38

bb38:		; preds = %bb37
	br label %bb39

bb39:		; preds = %bb38, %bb37
	%reg639 = phi i8 [ 1, %bb38 ], [ %reg629, %bb37 ]		; <i8> [#uses=2]
	%reg640 = add i32 %reg630, 1		; <i32> [#uses=1]
	%cond780 = icmp sle i32 %reg640, %reg109		; <i1> [#uses=1]
	br i1 %cond780, label %bb35, label %bb40

bb40:		; preds = %bb39, %bb34
	%reg641 = phi i8 [ %reg639, %bb39 ], [ 0, %bb34 ]		; <i8> [#uses=1]
	%cond781 = icmp eq i8 %reg641, 0		; <i1> [#uses=1]
	br i1 %cond781, label %bb44, label %bb41

bb41:		; preds = %bb40
	%cond782 = icmp sge i32 1, %reg109		; <i1> [#uses=1]
	br i1 %cond782, label %bb6, label %bb42

bb42:		; preds = %bb42, %bb41
	%cann-indvar2 = phi i32 [ 0, %bb41 ], [ %add1-indvar2, %bb42 ]		; <i32> [#uses=3]
	%reg643 = add i32 %cann-indvar2, 1		; <i32> [#uses=1]
	%add1-indvar2 = add i32 %cann-indvar2, 1		; <i32> [#uses=1]
	%cann-indvar2-idxcast = bitcast i32 %cann-indvar2 to i32		; <i32> [#uses=1]
	%cann-indvar2-idxcast-offset = add i32 %cann-indvar2-idxcast, 1		; <i32> [#uses=1]
	zext i32 %cann-indvar2-idxcast-offset to i64		; <i64>:51 [#uses=1]
	%reg569 = getelementptr i32* %weight, i64 %51		; <i32*> [#uses=2]
	%reg570 = load i32* %reg569		; <i32> [#uses=2]
	%reg644 = ashr i32 %reg570, 8		; <i32> [#uses=1]
	%reg572 = ashr i32 %reg570, 31		; <i32> [#uses=1]
	%cast933 = bitcast i32 %reg572 to i32		; <i32> [#uses=1]
	%reg573 = lshr i32 %cast933, 31		; <i32> [#uses=1]
	%cast934 = bitcast i32 %reg573 to i32		; <i32> [#uses=1]
	%reg574 = add i32 %reg644, %cast934		; <i32> [#uses=1]
	%reg571 = ashr i32 %reg574, 1		; <i32> [#uses=1]
	%reg645 = add i32 %reg571, 1		; <i32> [#uses=1]
	%reg582 = shl i32 %reg645, 8		; <i32> [#uses=1]
	store i32 %reg582, i32* %reg569
	%reg646 = add i32 %reg643, 1		; <i32> [#uses=1]
	%cond783 = icmp slt i32 %reg646, %reg109		; <i1> [#uses=1]
	br i1 %cond783, label %bb42, label %bb43

bb43:		; preds = %bb42
	br label %bb6

bb44:		; preds = %bb40
	ret void
}

declare void @panic(i8*)
