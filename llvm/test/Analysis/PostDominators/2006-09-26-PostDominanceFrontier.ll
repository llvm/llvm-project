; RUN: llvm-as < %s | opt -analyze -postdomfrontier \
; RUN:   -disable-verify
; ModuleID = '2006-09-26-PostDominanceFrontier.bc'
target datalayout = "e-p:64:64"
target triple = "alphaev67-unknown-linux-gnu"
	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i32, [44 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
@TOP = external global i64*		; <i64**> [#uses=1]
@BOT = external global i64*		; <i64**> [#uses=1]
@str = external global [2 x i8]		; <[2 x i8]*> [#uses=0]

declare void @fopen()

define void @main(i8** %argv) {
entry:
	%netSelect.i507 = alloca i64, align 8		; <i64*> [#uses=0]
	%topStart.i = alloca i64, align 8		; <i64*> [#uses=0]
	%topEnd.i = alloca i64, align 8		; <i64*> [#uses=0]
	%botStart.i = alloca i64, align 8		; <i64*> [#uses=0]
	%botEnd.i = alloca i64, align 8		; <i64*> [#uses=0]
	%c1.i154 = alloca i32, align 4		; <i32*> [#uses=0]
	%b1.i155 = alloca i32, align 4		; <i32*> [#uses=0]
	%t1.i156 = alloca i32, align 4		; <i32*> [#uses=0]
	%c1.i = alloca i32, align 4		; <i32*> [#uses=0]
	%b1.i = alloca i32, align 4		; <i32*> [#uses=0]
	%t1.i = alloca i32, align 4		; <i32*> [#uses=0]
	%netSelect.i5 = alloca i64, align 8		; <i64*> [#uses=0]
	%netSelect.i = alloca i64, align 8		; <i64*> [#uses=0]
	%tmp2.i = getelementptr i8** %argv, i32 1		; <i8**> [#uses=1]
	%tmp3.i4 = load i8** %tmp2.i		; <i8*> [#uses=0]
	call void @fopen( )
	br i1 false, label %DimensionChannel.exit, label %bb.backedge.i

bb.backedge.i:		; preds = %entry
	ret void

DimensionChannel.exit:		; preds = %entry
	%tmp13.i137 = malloc i64, i32 0		; <i64*> [#uses=1]
	%tmp610.i = malloc i64, i32 0		; <i64*> [#uses=1]
	br label %cond_true.i143

cond_true.i143:		; preds = %cond_true.i143, %DimensionChannel.exit
	%tmp9.i140 = getelementptr i64* %tmp13.i137, i64 0		; <i64*> [#uses=0]
	%tmp12.i = getelementptr i64* %tmp610.i, i64 0		; <i64*> [#uses=0]
	br i1 false, label %bb18.i144, label %cond_true.i143

bb18.i144:		; preds = %cond_true.i143
	call void @fopen( )
	%tmp76.i105 = malloc i64, i32 0		; <i64*> [#uses=3]
	%tmp674.i = malloc i64, i32 0		; <i64*> [#uses=2]
	%tmp1072.i = malloc i64, i32 0		; <i64*> [#uses=2]
	%tmp1470.i = malloc i64, i32 0		; <i64*> [#uses=1]
	br label %cond_true.i114

cond_true.i114:		; preds = %cond_true.i114, %bb18.i144
	%tmp17.i108 = getelementptr i64* %tmp76.i105, i64 0		; <i64*> [#uses=0]
	%tmp20.i = getelementptr i64* %tmp674.i, i64 0		; <i64*> [#uses=0]
	%tmp23.i111 = getelementptr i64* %tmp1470.i, i64 0		; <i64*> [#uses=0]
	br i1 false, label %cond_true40.i, label %cond_true.i114

cond_true40.i:		; preds = %cond_true40.i, %cond_true.i114
	%tmp33.i115 = getelementptr i64* %tmp1072.i, i64 0		; <i64*> [#uses=0]
	br i1 false, label %bb142.i, label %cond_true40.i

cond_next54.i:		; preds = %cond_true76.i
	%tmp57.i = getelementptr i64* %tmp55.i, i64 0		; <i64*> [#uses=0]
	br i1 false, label %bb64.i, label %bb69.i

bb64.i:		; preds = %cond_true76.i, %cond_next54.i
	%tmp67.i117 = getelementptr i64* %tmp76.i105, i64 0		; <i64*> [#uses=0]
	br i1 false, label %bb114.i, label %cond_true111.i

bb69.i:		; preds = %cond_next54.i
	br i1 false, label %bb79.i, label %cond_true76.i

cond_true76.i:		; preds = %bb142.i, %bb69.i
	%tmp48.i = getelementptr i64* %tmp46.i, i64 0		; <i64*> [#uses=0]
	br i1 false, label %bb64.i, label %cond_next54.i

bb79.i:		; preds = %bb69.i
	br i1 false, label %bb114.i, label %cond_true111.i

cond_true111.i:		; preds = %bb79.i, %bb64.i
	%tmp84.i127 = getelementptr i64* %tmp46.i, i64 0		; <i64*> [#uses=0]
	ret void

bb114.i:		; preds = %bb142.i, %bb79.i, %bb64.i
	%tmp117.i = getelementptr i64* %tmp76.i105, i64 0		; <i64*> [#uses=0]
	%tmp132.i131 = getelementptr i64* %tmp674.i, i64 0		; <i64*> [#uses=0]
	%tmp122.i = getelementptr i64* %tmp1072.i, i64 0		; <i64*> [#uses=0]
	ret void

bb142.i:		; preds = %cond_true40.i
	%tmp46.i = load i64** @BOT		; <i64*> [#uses=2]
	%tmp55.i = load i64** @TOP		; <i64*> [#uses=1]
	br i1 false, label %bb114.i, label %cond_true76.i
}
