; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split

target triple = "i386-apple-darwin9.5"
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
@"\01LC1" = external constant [48 x i8]		; <[48 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
	br label %bb5.us

bb5.us:		; preds = %bb8.split, %bb5.us, %entry
	%i.0.reg2mem.0.ph = phi i32 [ 0, %entry ], [ %indvar.next53, %bb8.split ], [ %i.0.reg2mem.0.ph, %bb5.us ]		; <i32> [#uses=2]
	%j.0.reg2mem.0.us = phi i32 [ %indvar.next47, %bb5.us ], [ 0, %bb8.split ], [ 0, %entry ]		; <i32> [#uses=1]
	%indvar.next47 = add i32 %j.0.reg2mem.0.us, 1		; <i32> [#uses=2]
	%exitcond48 = icmp eq i32 %indvar.next47, 256		; <i1> [#uses=1]
	br i1 %exitcond48, label %bb8.split, label %bb5.us

bb8.split:		; preds = %bb5.us
	%indvar.next53 = add i32 %i.0.reg2mem.0.ph, 1		; <i32> [#uses=2]
	%exitcond54 = icmp eq i32 %indvar.next53, 256		; <i1> [#uses=1]
	br i1 %exitcond54, label %bb11, label %bb5.us

bb11:		; preds = %bb11, %bb8.split
	%i.1.reg2mem.0 = phi i32 [ %indvar.next44, %bb11 ], [ 0, %bb8.split ]		; <i32> [#uses=1]
	%indvar.next44 = add i32 %i.1.reg2mem.0, 1		; <i32> [#uses=2]
	%exitcond45 = icmp eq i32 %indvar.next44, 63		; <i1> [#uses=1]
	br i1 %exitcond45, label %bb14, label %bb11

bb14:		; preds = %bb14, %bb11
	%indvar = phi i32 [ %indvar.next40, %bb14 ], [ 0, %bb11 ]		; <i32> [#uses=1]
	%indvar.next40 = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond41 = icmp eq i32 %indvar.next40, 32768		; <i1> [#uses=1]
	br i1 %exitcond41, label %bb28, label %bb14

bb28:		; preds = %bb14
	%0 = fdiv double 2.550000e+02, 0.000000e+00		; <double> [#uses=1]
	br label %bb30

bb30:		; preds = %bb36, %bb28
	%m.1.reg2mem.0 = phi i32 [ %m.0, %bb36 ], [ 0, %bb28 ]		; <i32> [#uses=1]
	%1 = mul double 0.000000e+00, %0		; <double> [#uses=1]
	%2 = fptosi double %1 to i32		; <i32> [#uses=1]
	br i1 false, label %bb36, label %bb35

bb35:		; preds = %bb30
	%3 = tail call i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* null, i8* getelementptr ([48 x i8]* @"\01LC1", i32 0, i32 0), i32 0, i32 0, i32 0, i32 %2) nounwind		; <i32> [#uses=0]
	br label %bb36

bb36:		; preds = %bb35, %bb30
	%m.0 = phi i32 [ 0, %bb35 ], [ %m.1.reg2mem.0, %bb30 ]		; <i32> [#uses=1]
	br label %bb30
}

declare i32 @fprintf(%struct.FILE*, i8*, ...) nounwind
