; RUN: llc < %s -march=arm -mtriple=armv6-apple-darwin9

@XX = external global i32*		; <i32**> [#uses=1]

define void @SIM(i8* %A, i8* %B, i32 %M, i32 %N, i32 %K, [256 x i32]* %V, i32 %Q, i32 %R, i32 %nseq) nounwind {
entry:
	br i1 undef, label %bb5, label %bb

bb:		; preds = %bb, %entry
	br label %bb

bb5:		; preds = %entry
	br i1 undef, label %bb6, label %bb8

bb6:		; preds = %bb6, %bb5
	br i1 undef, label %bb8, label %bb6

bb8:		; preds = %bb6, %bb5
	br label %bb15

bb9:		; preds = %bb15
	br i1 undef, label %bb10, label %bb11

bb10:		; preds = %bb9
	unreachable

bb11:		; preds = %bb9
	br i1 undef, label %bb15, label %bb12

bb12:		; preds = %bb11
	%0 = load i32*, i32** @XX, align 4		; <i32*> [#uses=0]
	br label %bb228.i

bb74.i:		; preds = %bb228.i
	br i1 undef, label %bb138.i, label %bb145.i

bb138.i:		; preds = %bb74.i
	br label %bb145.i

bb145.i:		; preds = %bb228.i, %bb138.i, %bb74.i
	br i1 undef, label %bb146.i, label %bb151.i

bb146.i:		; preds = %bb145.i
	br i1 undef, label %bb228.i, label %bb151.i

bb151.i:		; preds = %bb146.i, %bb145.i
	br i1 undef, label %bb153.i, label %bb228.i

bb153.i:		; preds = %bb151.i
	br i1 undef, label %bb220.i, label %bb.nph.i98

bb.nph.i98:		; preds = %bb153.i
	br label %bb158.i

bb158.i:		; preds = %bb218.i, %bb.nph.i98
	%1 = sub i32 undef, undef		; <i32> [#uses=4]
	%2 = sub i32 undef, undef		; <i32> [#uses=1]
	br i1 undef, label %bb168.i, label %bb160.i

bb160.i:		; preds = %bb158.i
	br i1 undef, label %bb161.i, label %bb168.i

bb161.i:		; preds = %bb160.i
	br i1 undef, label %bb168.i, label %bb163.i

bb163.i:		; preds = %bb161.i
	br i1 undef, label %bb167.i, label %bb168.i

bb167.i:		; preds = %bb163.i
	br label %bb168.i

bb168.i:		; preds = %bb167.i, %bb163.i, %bb161.i, %bb160.i, %bb158.i
	%f.5.i = phi i32 [ %1, %bb167.i ], [ %2, %bb158.i ], [ %1, %bb160.i ], [ %1, %bb161.i ], [ %1, %bb163.i ]		; <i32> [#uses=1]
	%c.14.i = select i1 undef, i32 %f.5.i, i32 undef		; <i32> [#uses=1]
	store i32 %c.14.i, i32* undef, align 4
	store i32 undef, i32* null, align 4
	br i1 undef, label %bb211.i, label %bb218.i

bb211.i:		; preds = %bb168.i
	br label %bb218.i

bb218.i:		; preds = %bb211.i, %bb168.i
	br i1 undef, label %bb220.i, label %bb158.i

bb220.i:		; preds = %bb218.i, %bb153.i
	br i1 undef, label %bb221.i, label %bb228.i

bb221.i:		; preds = %bb220.i
	br label %bb228.i

bb228.i:		; preds = %bb221.i, %bb220.i, %bb151.i, %bb146.i, %bb12
	br i1 undef, label %bb74.i, label %bb145.i

bb15:		; preds = %bb11, %bb8
	br i1 undef, label %return, label %bb9

return:		; preds = %bb15
	ret void
}
