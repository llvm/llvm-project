; RUN: llc < %s -mtriple=armv6-apple-darwin9

@no_mat = external global i32		; <ptr> [#uses=1]
@no_mis = external global i32		; <ptr> [#uses=2]
@"\01LC11" = external constant [33 x i8], align 1		; <ptr> [#uses=1]
@"\01LC15" = external constant [33 x i8], align 1		; <ptr> [#uses=1]
@"\01LC17" = external constant [47 x i8], align 1		; <ptr> [#uses=1]

declare i32 @printf(ptr nocapture, ...) nounwind

declare void @diff(ptr, ptr, i32, i32, i32, i32) nounwind

define void @SIM(ptr %A, ptr %B, i32 %M, i32 %N, i32 %K, ptr %V, i32 %Q, i32 %R, i32 %nseq) nounwind {
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
	%0 = load i32, ptr undef, align 4		; <i32> [#uses=3]
	%1 = add i32 %0, 1		; <i32> [#uses=2]
	store i32 %1, ptr undef, align 4
	%2 = load i32, ptr undef, align 4		; <i32> [#uses=2]
	%3 = sub i32 %2, %0		; <i32> [#uses=1]
	store i32 0, ptr @no_mat, align 4
	store i32 0, ptr @no_mis, align 4
	%4 = getelementptr i8, ptr %B, i32 %0		; <ptr> [#uses=1]
	tail call  void @diff(ptr undef, ptr %4, i32 undef, i32 %3, i32 undef, i32 undef) nounwind
	%5 = tail call  i32 (ptr, ...) @printf(ptr @"\01LC11", i32 %tmp13) nounwind		; <i32> [#uses=0]
	%6 = load i32, ptr @no_mis, align 4		; <i32> [#uses=1]
	%7 = tail call  i32 (ptr, ...) @printf(ptr @"\01LC15", i32 %6) nounwind		; <i32> [#uses=0]
	%8 = tail call  i32 (ptr, ...) @printf(ptr @"\01LC17", i32 undef, i32 %1, i32 undef, i32 %2) nounwind		; <i32> [#uses=0]
	br i1 undef, label %bb15, label %bb12

bb12:		; preds = %bb11
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
	%indvar11 = phi i32 [ 0, %bb8 ], [ %tmp13, %bb11 ]		; <i32> [#uses=2]
	%tmp13 = add i32 %indvar11, 1		; <i32> [#uses=2]
	%count.0 = sub i32 undef, %indvar11		; <i32> [#uses=0]
	br i1 undef, label %return, label %bb9

return:		; preds = %bb15
	ret void
}
