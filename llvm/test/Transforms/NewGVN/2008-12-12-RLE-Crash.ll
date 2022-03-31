; RUN: opt < %s -passes=newgvn | llvm-dis
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	br label %bb84

bb41:		; preds = %bb82
	%tmp = load i8, i8* %opt.0, align 1		; <i8> [#uses=0]
	%tmp1 = getelementptr i8, i8* %opt.0, i32 1		; <i8*> [#uses=2]
	switch i32 0, label %bb81 [
		i32 102, label %bb82
		i32 110, label %bb79
		i32 118, label %bb80
	]

bb79:		; preds = %bb41
	br label %bb82

bb80:		; preds = %bb41
	ret i32 0

bb81:		; preds = %bb41
	ret i32 1

bb82:		; preds = %bb84, %bb79, %bb41
	%opt.0 = phi i8* [ %tmp3, %bb84 ], [ %tmp1, %bb79 ], [ %tmp1, %bb41 ]		; <i8*> [#uses=3]
	%tmp2 = load i8, i8* %opt.0, align 1		; <i8> [#uses=0]
	br i1 false, label %bb84, label %bb41

bb84:		; preds = %bb82, %entry
	%tmp3 = getelementptr i8, i8* null, i32 1		; <i8*> [#uses=1]
	br label %bb82
}
