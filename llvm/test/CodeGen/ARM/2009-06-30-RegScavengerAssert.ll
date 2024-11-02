; RUN: llc < %s -mtriple=armv6-apple-darwin9

@nn = external global i32		; <ptr> [#uses=1]
@al_len = external global i32		; <ptr> [#uses=2]
@no_mat = external global i32		; <ptr> [#uses=2]
@no_mis = external global i32		; <ptr> [#uses=2]
@"\01LC12" = external constant [29 x i8], align 1		; <ptr> [#uses=1]
@"\01LC16" = external constant [33 x i8], align 1		; <ptr> [#uses=1]
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
	%0 = load i32, ptr undef, align 4		; <i32> [#uses=2]
	%1 = add i32 %0, 1		; <i32> [#uses=2]
	store i32 %1, ptr undef, align 4
	%2 = load i32, ptr undef, align 4		; <i32> [#uses=1]
	store i32 %2, ptr @nn, align 4
	store i32 0, ptr @al_len, align 4
	store i32 0, ptr @no_mat, align 4
	store i32 0, ptr @no_mis, align 4
	%3 = getelementptr i8, ptr %B, i32 %0		; <ptr> [#uses=1]
	tail call  void @diff(ptr undef, ptr %3, i32 undef, i32 undef, i32 undef, i32 undef) nounwind
	%4 = sitofp i32 undef to double		; <double> [#uses=1]
	%5 = fdiv double %4, 1.000000e+01		; <double> [#uses=1]
	%6 = tail call  i32 (ptr, ...) @printf(ptr @"\01LC12", double %5) nounwind		; <i32> [#uses=0]
	%7 = load i32, ptr @al_len, align 4		; <i32> [#uses=1]
	%8 = load i32, ptr @no_mat, align 4		; <i32> [#uses=1]
	%9 = load i32, ptr @no_mis, align 4		; <i32> [#uses=1]
	%10 = sub i32 %7, %8		; <i32> [#uses=1]
	%11 = sub i32 %10, %9		; <i32> [#uses=1]
	%12 = tail call  i32 (ptr, ...) @printf(ptr @"\01LC16", i32 %11) nounwind		; <i32> [#uses=0]
	%13 = tail call  i32 (ptr, ...) @printf(ptr @"\01LC17", i32 undef, i32 %1, i32 undef, i32 undef) nounwind		; <i32> [#uses=0]
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
	br i1 undef, label %return, label %bb9

return:		; preds = %bb15
	ret void
}
