; RUN: llc < %s -mtriple=armv6-apple-darwin9

@r = external global i32		; <ptr> [#uses=1]
@qr = external global i32		; <ptr> [#uses=1]
@II = external global ptr		; <ptr> [#uses=1]
@no_mis = external global i32		; <ptr> [#uses=1]
@name1 = external global ptr		; <ptr> [#uses=1]

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
	%0 = load ptr, ptr @name1, align 4		; <ptr> [#uses=0]
	br label %bb15

bb9:		; preds = %bb15
	br i1 undef, label %bb10, label %bb11

bb10:		; preds = %bb9
	unreachable

bb11:		; preds = %bb9
	store i32 0, ptr @no_mis, align 4
	%1 = getelementptr i8, ptr %A, i32 0		; <ptr> [#uses=1]
	%2 = getelementptr i8, ptr %B, i32 0		; <ptr> [#uses=1]
	tail call  void @diff(ptr %1, ptr %2, i32 undef, i32 undef, i32 undef, i32 undef) nounwind
	br i1 undef, label %bb15, label %bb12

bb12:		; preds = %bb11
	%3 = load ptr, ptr @II, align 4		; <ptr> [#uses=1]
	%4 = load i32, ptr @r, align 4		; <i32> [#uses=1]
	%5 = load i32, ptr @qr, align 4		; <i32> [#uses=1]
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
	%6 = add i32 undef, -1		; <i32> [#uses=3]
	br i1 undef, label %bb220.i, label %bb.nph.i98

bb.nph.i98:		; preds = %bb153.i
	br label %bb158.i

bb158.i:		; preds = %bb218.i, %bb.nph.i98
	%c.1020.i = phi i32 [ 0, %bb.nph.i98 ], [ %c.14.i, %bb218.i ]		; <i32> [#uses=1]
	%f.419.i = phi i32 [ undef, %bb.nph.i98 ], [ %f.5.i, %bb218.i ]		; <i32> [#uses=1]
	%pi.316.i = phi i32 [ undef, %bb.nph.i98 ], [ %10, %bb218.i ]		; <i32> [#uses=1]
	%fj.515.i = phi i32 [ %6, %bb.nph.i98 ], [ %fj.4.i, %bb218.i ]		; <i32> [#uses=2]
	%fi.614.i = phi i32 [ undef, %bb.nph.i98 ], [ %fi.5.i, %bb218.i ]		; <i32> [#uses=3]
	%cj.811.i = phi i32 [ %6, %bb.nph.i98 ], [ %cj.11.i100, %bb218.i ]		; <i32> [#uses=3]
	%ci.910.i = phi i32 [ undef, %bb.nph.i98 ], [ %ci.12.i, %bb218.i ]		; <i32> [#uses=2]
	%7 = sub i32 %f.419.i, %4		; <i32> [#uses=5]
	%8 = sub i32 %c.1020.i, %5		; <i32> [#uses=2]
	%9 = icmp slt i32 %7, %8		; <i1> [#uses=1]
	br i1 %9, label %bb168.i, label %bb160.i

bb160.i:		; preds = %bb158.i
	br i1 undef, label %bb161.i, label %bb168.i

bb161.i:		; preds = %bb160.i
	br i1 undef, label %bb168.i, label %bb163.i

bb163.i:		; preds = %bb161.i
	br i1 undef, label %bb167.i, label %bb168.i

bb167.i:		; preds = %bb163.i
	br label %bb168.i

bb168.i:		; preds = %bb167.i, %bb163.i, %bb161.i, %bb160.i, %bb158.i
	%fi.5.i = phi i32 [ %fi.614.i, %bb167.i ], [ %ci.910.i, %bb158.i ], [ %fi.614.i, %bb160.i ], [ %ci.910.i, %bb161.i ], [ %fi.614.i, %bb163.i ]		; <i32> [#uses=2]
	%fj.4.i = phi i32 [ %cj.811.i, %bb167.i ], [ %cj.811.i, %bb158.i ], [ %fj.515.i, %bb160.i ], [ %cj.811.i, %bb161.i ], [ %fj.515.i, %bb163.i ]		; <i32> [#uses=2]
	%f.5.i = phi i32 [ %7, %bb167.i ], [ %8, %bb158.i ], [ %7, %bb160.i ], [ %7, %bb161.i ], [ %7, %bb163.i ]		; <i32> [#uses=2]
	%scevgep88.i = getelementptr i32, ptr %3, i32 undef		; <ptr> [#uses=1]
	%ci.10.i = select i1 undef, i32 %pi.316.i, i32 undef		; <i32> [#uses=0]
	%ci.12.i = select i1 undef, i32 %fi.5.i, i32 undef		; <i32> [#uses=1]
	%cj.11.i100 = select i1 undef, i32 %fj.4.i, i32 undef		; <i32> [#uses=1]
	%c.14.i = select i1 undef, i32 %f.5.i, i32 undef		; <i32> [#uses=1]
	%10 = load i32, ptr %scevgep88.i, align 4		; <i32> [#uses=1]
	br i1 undef, label %bb211.i, label %bb218.i

bb211.i:		; preds = %bb168.i
	br label %bb218.i

bb218.i:		; preds = %bb211.i, %bb168.i
	br i1 undef, label %bb220.i, label %bb158.i

bb220.i:		; preds = %bb218.i, %bb153.i
	%11 = getelementptr i32, ptr null, i32 %6		; <ptr> [#uses=1]
	store i32 undef, ptr %11, align 4
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
