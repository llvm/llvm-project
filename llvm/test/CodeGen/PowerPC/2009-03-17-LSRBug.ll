; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu
; rdar://6692215

define fastcc void @_qsort(i8* %a, i32 %n, i32 %es, i32 (i8*, i8*)* %cmp, i32 %depth_limit) nounwind optsize ssp {
entry:
	br i1 false, label %bb21, label %bb20.loopexit

bb20.loopexit:		; preds = %entry
	ret void

bb21:		; preds = %entry
	%0 = getelementptr i8, i8* %a, i32 0		; <i8*> [#uses=2]
	br label %bb35

bb29:		; preds = %bb35
	br i1 false, label %bb7.i252, label %bb34

bb7.i252:		; preds = %bb7.i252, %bb29
	%pj.0.rec.i247 = phi i32 [ %indvar.next488, %bb7.i252 ], [ 0, %bb29 ]		; <i32> [#uses=2]
	%pi.0.i248 = getelementptr i8, i8* %pa.1, i32 %pj.0.rec.i247		; <i8*> [#uses=0]
	%indvar.next488 = add i32 %pj.0.rec.i247, 1		; <i32> [#uses=1]
	br i1 false, label %bb34, label %bb7.i252

bb34:		; preds = %bb7.i252, %bb29
	%indvar.next505 = add i32 %indvar504, 1		; <i32> [#uses=1]
	br label %bb35

bb35:		; preds = %bb34, %bb21
	%indvar504 = phi i32 [ %indvar.next505, %bb34 ], [ 0, %bb21 ]		; <i32> [#uses=2]
	%pa.1 = phi i8* [ null, %bb34 ], [ %0, %bb21 ]		; <i8*> [#uses=2]
	%pb.0.rec = mul i32 %indvar504, %es		; <i32> [#uses=1]
	br i1 false, label %bb43, label %bb29

bb43:		; preds = %bb43, %bb35
	br i1 false, label %bb50, label %bb43

bb50:		; preds = %bb43
	%1 = ptrtoint i8* %pa.1 to i32		; <i32> [#uses=1]
	%2 = sub i32 %1, 0		; <i32> [#uses=2]
	%3 = icmp sle i32 0, %2		; <i1> [#uses=1]
	%min = select i1 %3, i32 0, i32 %2		; <i32> [#uses=1]
	br label %bb7.i161

bb7.i161:		; preds = %bb7.i161, %bb50
	%pj.0.rec.i156 = phi i32 [ %indvar.next394, %bb7.i161 ], [ 0, %bb50 ]		; <i32> [#uses=2]
	%.sum279 = sub i32 %pj.0.rec.i156, %min		; <i32> [#uses=1]
	%pb.0.sum542 = add i32 %pb.0.rec, %.sum279		; <i32> [#uses=1]
	%pj.0.i158 = getelementptr i8, i8* %0, i32 %pb.0.sum542		; <i8*> [#uses=0]
	%indvar.next394 = add i32 %pj.0.rec.i156, 1		; <i32> [#uses=1]
	br label %bb7.i161
}
