; RUN: llvm-as < %s | opt -loop-index-split -disable-output
; Handle Exit block phis that do not have any use inside the loop.

	%struct.ATOM = type { double, double, double, double, double, double, i32, double, double, double, double, i8*, i8, [9 x i8], double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, double, [200 x i8*], [32 x i8*], [32 x i8], i32 }
	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }

define i32 @math([80 x i8]* %tokens, double* %fvalue, i32* %ivalue, %struct.FILE* %ip, %struct.FILE* %op, i32 %echo) nounwind  {
entry:
	br i1 false, label %bb.i, label %bb35.i
bb.i:		; preds = %entry
	br i1 false, label %bb6.i, label %bb9.i
bb9.i:		; preds = %bb.i
	ret i32 0
bb35.i:		; preds = %entry
	ret i32 0
bb6.i:		; preds = %bb.i
	br i1 false, label %a_l2_f.exit, label %bb16.i
bb16.i:		; preds = %bb6.i
	ret i32 0
a_l2_f.exit:		; preds = %bb6.i
	br i1 false, label %bb7.i97, label %bb6.i71
bb6.i71:		; preds = %a_l2_f.exit
	ret i32 0
bb7.i97:		; preds = %a_l2_f.exit
	br i1 false, label %bb, label %bb18.i102
bb18.i102:		; preds = %bb7.i97
	ret i32 0
bb:		; preds = %bb7.i97
	br i1 false, label %bb38, label %AFOUND
bb38:		; preds = %bb
	br i1 false, label %bb111, label %bb7.i120
AFOUND:		; preds = %bb
	ret i32 0
bb7.i120:		; preds = %bb38
	ret i32 0
bb111:		; preds = %bb38
	switch i32 0, label %bb574 [
		 i32 1, label %bb158
		 i32 0, label %bb166
	]
bb158:		; preds = %bb111
	ret i32 0
bb166:		; preds = %bb111
	ret i32 0
bb574:		; preds = %bb111
	br i1 false, label %bb11.i249, label %bb600
bb11.i249:		; preds = %bb574
	br i1 false, label %bb11.i265, label %bb596
bb11.i265:		; preds = %bb590, %bb11.i249
	%i.1.reg2mem.0 = phi i32 [ %tmp589.reg2mem.0, %bb590 ], [ 0, %bb11.i249 ]		; <i32> [#uses=2]
	%tmp13.i264 = icmp slt i32 %i.1.reg2mem.0, 1		; <i1> [#uses=1]
	br i1 %tmp13.i264, label %bb16.i267, label %bb30.i279
bb16.i267:		; preds = %bb11.i265
	br label %bb590
bb30.i279:		; preds = %bb11.i265
	br label %bb590
bb590:		; preds = %bb30.i279, %bb16.i267
	%tmp5876282.reg2mem.0 = phi %struct.ATOM* [ null, %bb30.i279 ], [ null, %bb16.i267 ]		; <%struct.ATOM*> [#uses=1]
	%tmp589.reg2mem.0 = add i32 %i.1.reg2mem.0, 1		; <i32> [#uses=2]
	%tmp593 = icmp slt i32 %tmp589.reg2mem.0, 0		; <i1> [#uses=1]
	br i1 %tmp593, label %bb11.i265, label %bb596
bb596:		; preds = %bb590, %bb11.i249
	%ap.0.reg2mem.0 = phi %struct.ATOM* [ null, %bb11.i249 ], [ %tmp5876282.reg2mem.0, %bb590 ]		; <%struct.ATOM*> [#uses=0]
	ret i32 0
bb600:		; preds = %bb574
	ret i32 0
}
