; RUN: llc < %s -mtriple=thumbv6-apple-darwin10 | grep rsbs | grep "#0"

	%struct.FILE = type { ptr, i32, i32, i16, i16, %struct.__sbuf, i32, ptr, ptr, ptr, ptr, ptr, %struct.__sbuf, ptr, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { ptr, i32 }
	%struct.adpcm_state = type { i16, i8 }
@stepsizeTable = internal constant [89 x i32] [i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 16, i32 17, i32 19, i32 21, i32 23, i32 25, i32 28, i32 31, i32 34, i32 37, i32 41, i32 45, i32 50, i32 55, i32 60, i32 66, i32 73, i32 80, i32 88, i32 97, i32 107, i32 118, i32 130, i32 143, i32 157, i32 173, i32 190, i32 209, i32 230, i32 253, i32 279, i32 307, i32 337, i32 371, i32 408, i32 449, i32 494, i32 544, i32 598, i32 658, i32 724, i32 796, i32 876, i32 963, i32 1060, i32 1166, i32 1282, i32 1411, i32 1552, i32 1707, i32 1878, i32 2066, i32 2272, i32 2499, i32 2749, i32 3024, i32 3327, i32 3660, i32 4026, i32 4428, i32 4871, i32 5358, i32 5894, i32 6484, i32 7132, i32 7845, i32 8630, i32 9493, i32 10442, i32 11487, i32 12635, i32 13899, i32 15289, i32 16818, i32 18500, i32 20350, i32 22385, i32 24623, i32 27086, i32 29794, i32 32767]		; <ptr> [#uses=4]
@indexTable = internal constant [16 x i32] [i32 -1, i32 -1, i32 -1, i32 -1, i32 2, i32 4, i32 6, i32 8, i32 -1, i32 -1, i32 -1, i32 -1, i32 2, i32 4, i32 6, i32 8]		; <ptr> [#uses=2]
@abuf = common global [500 x i8] zeroinitializer		; <ptr> [#uses=1]
@.str = private constant [11 x i8] c"input file\00", section "__TEXT,__cstring,cstring_literals", align 1		; <ptr> [#uses=1]
@sbuf = common global [1000 x i16] zeroinitializer		; <ptr> [#uses=1]
@state = common global %struct.adpcm_state zeroinitializer		; <ptr> [#uses=3]
@__stderrp = external global ptr		; <ptr> [#uses=1]
@.str1 = private constant [28 x i8] c"Final valprev=%d, index=%d\0A\00", section "__TEXT,__cstring,cstring_literals", align 1		; <ptr> [#uses=1]

define void @adpcm_coder(ptr nocapture %indata, ptr nocapture %outdata, i32 %len, ptr nocapture %state) nounwind {
entry:
	%0 = getelementptr %struct.adpcm_state, ptr %state, i32 0, i32 0		; <ptr> [#uses=2]
	%1 = load i16, ptr %0, align 2		; <i16> [#uses=1]
	%2 = sext i16 %1 to i32		; <i32> [#uses=2]
	%3 = getelementptr %struct.adpcm_state, ptr %state, i32 0, i32 1		; <ptr> [#uses=2]
	%4 = load i8, ptr %3, align 2		; <i8> [#uses=1]
	%5 = sext i8 %4 to i32		; <i32> [#uses=3]
	%6 = getelementptr [89 x i32], ptr @stepsizeTable, i32 0, i32 %5		; <ptr> [#uses=1]
	%7 = load i32, ptr %6, align 4		; <i32> [#uses=1]
	%8 = icmp sgt i32 %len, 0		; <i1> [#uses=1]
	br i1 %8, label %bb, label %bb27

bb:		; preds = %bb25, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb25 ]		; <i32> [#uses=2]
	%outp.136 = phi ptr [ %outdata, %entry ], [ %outp.0, %bb25 ]		; <ptr> [#uses=3]
	%bufferstep.035 = phi i32 [ 1, %entry ], [ %tmp, %bb25 ]		; <i32> [#uses=3]
	%outputbuffer.134 = phi i32 [ undef, %entry ], [ %outputbuffer.0, %bb25 ]		; <i32> [#uses=2]
	%index.033 = phi i32 [ %5, %entry ], [ %index.2, %bb25 ]		; <i32> [#uses=1]
	%valpred.132 = phi i32 [ %2, %entry ], [ %valpred.2, %bb25 ]		; <i32> [#uses=2]
	%step.031 = phi i32 [ %7, %entry ], [ %36, %bb25 ]		; <i32> [#uses=5]
	%inp.038 = getelementptr i16, ptr %indata, i32 %indvar		; <ptr> [#uses=1]
	%9 = load i16, ptr %inp.038, align 2		; <i16> [#uses=1]
	%10 = sext i16 %9 to i32		; <i32> [#uses=1]
	%11 = sub i32 %10, %valpred.132		; <i32> [#uses=3]
	%12 = icmp slt i32 %11, 0		; <i1> [#uses=1]
	%iftmp.1.0 = select i1 %12, i32 8, i32 0		; <i32> [#uses=2]
	%13 = sub i32 0, %11		; <i32> [#uses=1]
	%14 = icmp eq i32 %iftmp.1.0, 0		; <i1> [#uses=2]
	%. = select i1 %14, i32 %11, i32 %13		; <i32> [#uses=2]
	%15 = ashr i32 %step.031, 3		; <i32> [#uses=1]
	%16 = icmp slt i32 %., %step.031		; <i1> [#uses=2]
	%delta.0 = select i1 %16, i32 0, i32 4		; <i32> [#uses=2]
	%17 = select i1 %16, i32 0, i32 %step.031		; <i32> [#uses=2]
	%diff.1 = sub i32 %., %17		; <i32> [#uses=2]
	%18 = ashr i32 %step.031, 1		; <i32> [#uses=2]
	%19 = icmp slt i32 %diff.1, %18		; <i1> [#uses=2]
	%20 = or i32 %delta.0, 2		; <i32> [#uses=1]
	%21 = select i1 %19, i32 %delta.0, i32 %20		; <i32> [#uses=1]
	%22 = select i1 %19, i32 0, i32 %18		; <i32> [#uses=2]
	%diff.2 = sub i32 %diff.1, %22		; <i32> [#uses=1]
	%23 = ashr i32 %step.031, 2		; <i32> [#uses=2]
	%24 = icmp slt i32 %diff.2, %23		; <i1> [#uses=2]
	%25 = zext i1 %24 to i32		; <i32> [#uses=1]
	%26 = select i1 %24, i32 0, i32 %23		; <i32> [#uses=1]
	%vpdiff.0 = add i32 %17, %15		; <i32> [#uses=1]
	%vpdiff.1 = add i32 %vpdiff.0, %22		; <i32> [#uses=1]
	%vpdiff.2 = add i32 %vpdiff.1, %26		; <i32> [#uses=2]
	%tmp30 = sub i32 0, %vpdiff.2		; <i32> [#uses=1]
	%valpred.0.p = select i1 %14, i32 %vpdiff.2, i32 %tmp30		; <i32> [#uses=1]
	%valpred.0 = add i32 %valpred.0.p, %valpred.132		; <i32> [#uses=3]
	%27 = icmp sgt i32 %valpred.0, 32767		; <i1> [#uses=1]
	br i1 %27, label %bb18, label %bb16

bb16:		; preds = %bb
	%28 = icmp slt i32 %valpred.0, -32768		; <i1> [#uses=1]
	br i1 %28, label %bb17, label %bb18

bb17:		; preds = %bb16
	br label %bb18

bb18:		; preds = %bb17, %bb16, %bb
	%valpred.2 = phi i32 [ -32768, %bb17 ], [ 32767, %bb ], [ %valpred.0, %bb16 ]		; <i32> [#uses=2]
	%delta.1 = or i32 %21, %iftmp.1.0		; <i32> [#uses=1]
	%delta.2 = or i32 %delta.1, %25		; <i32> [#uses=1]
	%29 = xor i32 %delta.2, 1		; <i32> [#uses=3]
	%30 = getelementptr [16 x i32], ptr @indexTable, i32 0, i32 %29		; <ptr> [#uses=1]
	%31 = load i32, ptr %30, align 4		; <i32> [#uses=1]
	%32 = add i32 %31, %index.033		; <i32> [#uses=2]
	%33 = icmp slt i32 %32, 0		; <i1> [#uses=1]
	%index.1 = select i1 %33, i32 0, i32 %32		; <i32> [#uses=2]
	%34 = icmp sgt i32 %index.1, 88		; <i1> [#uses=1]
	%index.2 = select i1 %34, i32 88, i32 %index.1		; <i32> [#uses=3]
	%35 = getelementptr [89 x i32], ptr @stepsizeTable, i32 0, i32 %index.2		; <ptr> [#uses=1]
	%36 = load i32, ptr %35, align 4		; <i32> [#uses=1]
	%37 = icmp eq i32 %bufferstep.035, 0		; <i1> [#uses=1]
	br i1 %37, label %bb24, label %bb23

bb23:		; preds = %bb18
	%38 = shl i32 %29, 4		; <i32> [#uses=1]
	%39 = and i32 %38, 240		; <i32> [#uses=1]
	br label %bb25

bb24:		; preds = %bb18
	%40 = trunc i32 %29 to i8		; <i8> [#uses=1]
	%41 = and i8 %40, 15		; <i8> [#uses=1]
	%42 = trunc i32 %outputbuffer.134 to i8		; <i8> [#uses=1]
	%43 = or i8 %41, %42		; <i8> [#uses=1]
	store i8 %43, ptr %outp.136, align 1
	%44 = getelementptr i8, ptr %outp.136, i32 1		; <ptr> [#uses=1]
	br label %bb25

bb25:		; preds = %bb24, %bb23
	%outputbuffer.0 = phi i32 [ %39, %bb23 ], [ %outputbuffer.134, %bb24 ]		; <i32> [#uses=2]
	%outp.0 = phi ptr [ %outp.136, %bb23 ], [ %44, %bb24 ]		; <ptr> [#uses=2]
	%tmp = xor i32 %bufferstep.035, 1		; <i32> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %len		; <i1> [#uses=1]
	br i1 %exitcond, label %bb26.bb27_crit_edge, label %bb

bb26.bb27_crit_edge:		; preds = %bb25
	%phitmp44 = icmp eq i32 %bufferstep.035, 1		; <i1> [#uses=1]
	br label %bb27

bb27:		; preds = %bb26.bb27_crit_edge, %entry
	%outp.1.lcssa = phi ptr [ %outp.0, %bb26.bb27_crit_edge ], [ %outdata, %entry ]		; <ptr> [#uses=1]
	%bufferstep.0.lcssa = phi i1 [ %phitmp44, %bb26.bb27_crit_edge ], [ false, %entry ]		; <i1> [#uses=1]
	%outputbuffer.1.lcssa = phi i32 [ %outputbuffer.0, %bb26.bb27_crit_edge ], [ undef, %entry ]		; <i32> [#uses=1]
	%index.0.lcssa = phi i32 [ %index.2, %bb26.bb27_crit_edge ], [ %5, %entry ]		; <i32> [#uses=1]
	%valpred.1.lcssa = phi i32 [ %valpred.2, %bb26.bb27_crit_edge ], [ %2, %entry ]		; <i32> [#uses=1]
	br i1 %bufferstep.0.lcssa, label %bb28, label %bb29

bb28:		; preds = %bb27
	%45 = trunc i32 %outputbuffer.1.lcssa to i8		; <i8> [#uses=1]
	store i8 %45, ptr %outp.1.lcssa, align 1
	br label %bb29

bb29:		; preds = %bb28, %bb27
	%46 = trunc i32 %valpred.1.lcssa to i16		; <i16> [#uses=1]
	store i16 %46, ptr %0, align 2
	%47 = trunc i32 %index.0.lcssa to i8		; <i8> [#uses=1]
	store i8 %47, ptr %3, align 2
	ret void
}

define void @adpcm_decoder(ptr nocapture %indata, ptr nocapture %outdata, i32 %len, ptr nocapture %state) nounwind {
entry:
	%0 = getelementptr %struct.adpcm_state, ptr %state, i32 0, i32 0		; <ptr> [#uses=2]
	%1 = load i16, ptr %0, align 2		; <i16> [#uses=1]
	%2 = sext i16 %1 to i32		; <i32> [#uses=2]
	%3 = getelementptr %struct.adpcm_state, ptr %state, i32 0, i32 1		; <ptr> [#uses=2]
	%4 = load i8, ptr %3, align 2		; <i8> [#uses=1]
	%5 = sext i8 %4 to i32		; <i32> [#uses=3]
	%6 = getelementptr [89 x i32], ptr @stepsizeTable, i32 0, i32 %5		; <ptr> [#uses=1]
	%7 = load i32, ptr %6, align 4		; <i32> [#uses=1]
	%8 = icmp sgt i32 %len, 0		; <i1> [#uses=1]
	br i1 %8, label %bb, label %bb22

bb:		; preds = %bb20, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb20 ]		; <i32> [#uses=2]
	%inp.131 = phi ptr [ %indata, %entry ], [ %inp.0, %bb20 ]		; <ptr> [#uses=3]
	%bufferstep.028 = phi i32 [ 0, %entry ], [ %tmp, %bb20 ]		; <i32> [#uses=2]
	%inputbuffer.127 = phi i32 [ undef, %entry ], [ %inputbuffer.0, %bb20 ]		; <i32> [#uses=2]
	%index.026 = phi i32 [ %5, %entry ], [ %index.2, %bb20 ]		; <i32> [#uses=1]
	%valpred.125 = phi i32 [ %2, %entry ], [ %valpred.2, %bb20 ]		; <i32> [#uses=1]
	%step.024 = phi i32 [ %7, %entry ], [ %35, %bb20 ]		; <i32> [#uses=4]
	%outp.030 = getelementptr i16, ptr %outdata, i32 %indvar		; <ptr> [#uses=1]
	%9 = icmp eq i32 %bufferstep.028, 0		; <i1> [#uses=1]
	br i1 %9, label %bb2, label %bb3

bb2:		; preds = %bb
	%10 = load i8, ptr %inp.131, align 1		; <i8> [#uses=1]
	%11 = sext i8 %10 to i32		; <i32> [#uses=2]
	%12 = getelementptr i8, ptr %inp.131, i32 1		; <ptr> [#uses=1]
	%13 = ashr i32 %11, 4		; <i32> [#uses=1]
	br label %bb3

bb3:		; preds = %bb2, %bb
	%inputbuffer.0 = phi i32 [ %11, %bb2 ], [ %inputbuffer.127, %bb ]		; <i32> [#uses=1]
	%delta.0.in = phi i32 [ %13, %bb2 ], [ %inputbuffer.127, %bb ]		; <i32> [#uses=5]
	%inp.0 = phi ptr [ %12, %bb2 ], [ %inp.131, %bb ]		; <ptr> [#uses=1]
	%delta.0 = and i32 %delta.0.in, 15		; <i32> [#uses=1]
	%tmp = xor i32 %bufferstep.028, 1		; <i32> [#uses=1]
	%14 = getelementptr [16 x i32], ptr @indexTable, i32 0, i32 %delta.0		; <ptr> [#uses=1]
	%15 = load i32, ptr %14, align 4		; <i32> [#uses=1]
	%16 = add i32 %15, %index.026		; <i32> [#uses=2]
	%17 = icmp slt i32 %16, 0		; <i1> [#uses=1]
	%index.1 = select i1 %17, i32 0, i32 %16		; <i32> [#uses=2]
	%18 = icmp sgt i32 %index.1, 88		; <i1> [#uses=1]
	%index.2 = select i1 %18, i32 88, i32 %index.1		; <i32> [#uses=3]
	%19 = and i32 %delta.0.in, 8		; <i32> [#uses=1]
	%20 = ashr i32 %step.024, 3		; <i32> [#uses=1]
	%21 = and i32 %delta.0.in, 4		; <i32> [#uses=1]
	%22 = icmp eq i32 %21, 0		; <i1> [#uses=1]
	%23 = select i1 %22, i32 0, i32 %step.024		; <i32> [#uses=1]
	%vpdiff.0 = add i32 %23, %20		; <i32> [#uses=2]
	%24 = and i32 %delta.0.in, 2		; <i32> [#uses=1]
	%25 = icmp eq i32 %24, 0		; <i1> [#uses=1]
	br i1 %25, label %bb11, label %bb10

bb10:		; preds = %bb3
	%26 = ashr i32 %step.024, 1		; <i32> [#uses=1]
	%27 = add i32 %vpdiff.0, %26		; <i32> [#uses=1]
	br label %bb11

bb11:		; preds = %bb10, %bb3
	%vpdiff.1 = phi i32 [ %27, %bb10 ], [ %vpdiff.0, %bb3 ]		; <i32> [#uses=2]
	%28 = and i32 %delta.0.in, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %28, 0		; <i1> [#uses=1]
	br i1 %toBool, label %bb13, label %bb12

bb12:		; preds = %bb11
	%29 = ashr i32 %step.024, 2		; <i32> [#uses=1]
	%30 = add i32 %vpdiff.1, %29		; <i32> [#uses=1]
	br label %bb13

bb13:		; preds = %bb12, %bb11
	%vpdiff.2 = phi i32 [ %30, %bb12 ], [ %vpdiff.1, %bb11 ]		; <i32> [#uses=2]
	%31 = icmp eq i32 %19, 0		; <i1> [#uses=1]
	%tmp23 = sub i32 0, %vpdiff.2		; <i32> [#uses=1]
	%valpred.0.p = select i1 %31, i32 %vpdiff.2, i32 %tmp23		; <i32> [#uses=1]
	%valpred.0 = add i32 %valpred.0.p, %valpred.125		; <i32> [#uses=3]
	%32 = icmp sgt i32 %valpred.0, 32767		; <i1> [#uses=1]
	br i1 %32, label %bb20, label %bb18

bb18:		; preds = %bb13
	%33 = icmp slt i32 %valpred.0, -32768		; <i1> [#uses=1]
	br i1 %33, label %bb19, label %bb20

bb19:		; preds = %bb18
	br label %bb20

bb20:		; preds = %bb19, %bb18, %bb13
	%valpred.2 = phi i32 [ -32768, %bb19 ], [ 32767, %bb13 ], [ %valpred.0, %bb18 ]		; <i32> [#uses=3]
	%34 = getelementptr [89 x i32], ptr @stepsizeTable, i32 0, i32 %index.2		; <ptr> [#uses=1]
	%35 = load i32, ptr %34, align 4		; <i32> [#uses=1]
	%36 = trunc i32 %valpred.2 to i16		; <i16> [#uses=1]
	store i16 %36, ptr %outp.030, align 2
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %len		; <i1> [#uses=1]
	br i1 %exitcond, label %bb22, label %bb

bb22:		; preds = %bb20, %entry
	%index.0.lcssa = phi i32 [ %5, %entry ], [ %index.2, %bb20 ]		; <i32> [#uses=1]
	%valpred.1.lcssa = phi i32 [ %2, %entry ], [ %valpred.2, %bb20 ]		; <i32> [#uses=1]
	%37 = trunc i32 %valpred.1.lcssa to i16		; <i16> [#uses=1]
	store i16 %37, ptr %0, align 2
	%38 = trunc i32 %index.0.lcssa to i8		; <i8> [#uses=1]
	store i8 %38, ptr %3, align 2
	ret void
}

define i32 @main() nounwind {
entry:
	br label %bb

bb:		; preds = %bb3, %entry
	%0 = tail call  i32 (...) @read(i32 0, ptr @abuf, i32 500) nounwind		; <i32> [#uses=4]
	%1 = icmp slt i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb2

bb1:		; preds = %bb
	tail call  void @perror(ptr @.str) nounwind
	ret i32 1

bb2:		; preds = %bb
	%2 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %2, label %bb4, label %bb3

bb3:		; preds = %bb2
	%3 = shl i32 %0, 1		; <i32> [#uses=1]
	tail call  void @adpcm_decoder(ptr @abuf, ptr @sbuf, i32 %3, ptr @state) nounwind
	%4 = shl i32 %0, 2		; <i32> [#uses=1]
	%5 = tail call  i32 (...) @write(i32 1, ptr @sbuf, i32 %4) nounwind		; <i32> [#uses=0]
	br label %bb

bb4:		; preds = %bb2
	%6 = load ptr, ptr @__stderrp, align 4		; <ptr> [#uses=1]
	%7 = load i16, ptr @state, align 4		; <i16> [#uses=1]
	%8 = sext i16 %7 to i32		; <i32> [#uses=1]
	%9 = load i8, ptr getelementptr (%struct.adpcm_state, ptr @state, i32 0, i32 1), align 2		; <i8> [#uses=1]
	%10 = sext i8 %9 to i32		; <i32> [#uses=1]
	%11 = tail call  i32 (ptr, ptr, ...) @fprintf(ptr %6, ptr @.str1, i32 %8, i32 %10) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @read(...)

declare void @perror(ptr nocapture) nounwind

declare i32 @write(...)

declare i32 @fprintf(ptr nocapture, ptr nocapture, ...) nounwind
