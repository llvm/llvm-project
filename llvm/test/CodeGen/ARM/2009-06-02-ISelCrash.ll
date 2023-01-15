; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=pic -mattr=+v6,+vfp2

@"\01LC" = external constant [15 x i8]		; <ptr> [#uses=1]

declare i32 @printf(ptr nocapture, ...) nounwind

define i32 @main() nounwind {
entry:
	br label %bb.i1.i

bb.i1.i:		; preds = %Cos.exit.i.i, %entry
	br label %bb.i.i.i

bb.i.i.i:		; preds = %bb.i.i.i, %bb.i1.i
	br i1 undef, label %Cos.exit.i.i, label %bb.i.i.i

Cos.exit.i.i:		; preds = %bb.i.i.i
	br i1 undef, label %bb2.i.i, label %bb.i1.i

bb2.i.i:		; preds = %Cos.exit.i.i
	br label %bb3.i.i

bb3.i.i:		; preds = %bb5.i.i, %bb2.i.i
	br label %bb4.i.i

bb4.i.i:		; preds = %bb4.i.i, %bb3.i.i
	br i1 undef, label %bb5.i.i, label %bb4.i.i

bb5.i.i:		; preds = %bb4.i.i
	br i1 undef, label %bb.i, label %bb3.i.i

bb.i:		; preds = %bb.i, %bb5.i.i
	br i1 undef, label %bb1.outer2.i.i.outer, label %bb.i

bb1.outer2.i.i.outer:		; preds = %Fft.exit.i, %bb5.i12.i, %bb.i
	br label %bb1.outer2.i.i

bb1.outer2.i.i:		; preds = %bb2.i9.i, %bb1.outer2.i.i.outer
	br label %bb1.i.i

bb1.i.i:		; preds = %bb1.i.i, %bb1.outer2.i.i
	br i1 undef, label %bb2.i9.i, label %bb1.i.i

bb2.i9.i:		; preds = %bb1.i.i
	br i1 undef, label %bb4.i11.i, label %bb1.outer2.i.i

bb4.i11.i:		; preds = %bb4.i11.i, %bb2.i9.i
	br i1 undef, label %bb5.i12.i, label %bb4.i11.i

bb5.i12.i:		; preds = %bb4.i11.i
	br i1 undef, label %bb7.i.i, label %bb1.outer2.i.i.outer

bb7.i.i:		; preds = %bb7.i.i, %bb5.i12.i
	br i1 undef, label %Fft.exit.i, label %bb7.i.i

Fft.exit.i:		; preds = %bb7.i.i
	br i1 undef, label %bb5.i, label %bb1.outer2.i.i.outer

bb5.i:		; preds = %Fft.exit.i
	%0 = tail call i32 (ptr, ...) @printf(ptr @"\01LC", double undef, double undef) nounwind		; <i32> [#uses=0]
	unreachable
}
