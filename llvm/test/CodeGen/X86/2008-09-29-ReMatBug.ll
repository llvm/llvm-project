; RUN: llc < %s -mtriple=i386-apple-darwin -relocation-model=pic -frame-pointer=all

	%struct..0objc_selector = type opaque
	%struct.NSString = type opaque
	%struct.XCStringList = type { i32, ptr }
	%struct._XCStringListNode = type { [3 x i8], [0 x i8], i8 }
	%struct.__builtin_CFString = type { ptr, i32, ptr, i32 }
@0 = internal constant %struct.__builtin_CFString { ptr @__CFConstantStringClassReference, i32 1992, ptr @"\01LC", i32 2 }		; <ptr>:0 [#uses=1]
@__CFConstantStringClassReference = external global [0 x i32]		; <ptr> [#uses=1]
@"\01LC" = internal constant [3 x i8] c"NO\00"		; <ptr> [#uses=1]
@"\01LC1" = internal constant [1 x i8] zeroinitializer		; <ptr> [#uses=1]
@llvm.used1 = appending global [1 x ptr] [ ptr @"-[XCStringList stringRepresentation]" ], section "llvm.metadata"		; <ptr> [#uses=0]

define ptr @"-[XCStringList stringRepresentation]"(ptr %self, ptr %_cmd) nounwind {
entry:
	%0 = load i32, ptr null, align 4		; <i32> [#uses=1]
	%1 = and i32 %0, 16777215		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %bb44, label %bb4

bb4:		; preds = %entry
	%3 = load ptr, ptr null, align 4		; <ptr> [#uses=2]
	%4 = icmp eq ptr %3, null		; <i1> [#uses=1]
	br label %bb37.outer

bb6:		; preds = %bb37
	br label %bb19

bb19:		; preds = %bb37, %bb6
	%.rle = phi i32 [ 0, %bb6 ], [ %9, %bb37 ]		; <i32> [#uses=1]
	%bufptr.0.lcssa = phi ptr [ null, %bb6 ], [ null, %bb37 ]		; <ptr> [#uses=2]
	%5 = and i32 %.rle, 16777215		; <i32> [#uses=1]
	%6 = icmp eq i32 %5, 0		; <i1> [#uses=1]
	br i1 %6, label %bb25.split, label %bb37

bb25.split:		; preds = %bb19
	call void @foo(ptr @"\01LC1") nounwind nounwind
	br label %bb35.outer

bb34:		; preds = %bb35, %bb35, %bb35, %bb35
	%7 = getelementptr i8, ptr %bufptr.0.lcssa, i32 %totalLength.0.ph		; <ptr> [#uses=1]
	store i8 92, ptr %7, align 1
	br label %bb35.outer

bb35.outer:		; preds = %bb34, %bb25.split
	%totalLength.0.ph = add i32 0, %totalLength.1.ph		; <i32> [#uses=2]
	br label %bb35

bb35:		; preds = %bb35, %bb35.outer
	%8 = load i8, ptr null, align 1		; <i8> [#uses=1]
	switch i8 %8, label %bb35 [
		i8 0, label %bb37.outer
		i8 32, label %bb34
		i8 92, label %bb34
		i8 34, label %bb34
		i8 39, label %bb34
	]

bb37.outer:		; preds = %bb35, %bb4
	%totalLength.1.ph = phi i32 [ 0, %bb4 ], [ %totalLength.0.ph, %bb35 ]		; <i32> [#uses=1]
	%bufptr.1.ph = phi ptr [ null, %bb4 ], [ %bufptr.0.lcssa, %bb35 ]		; <ptr> [#uses=2]
	br i1 %4, label %bb39.split, label %bb37

bb37:		; preds = %bb37.outer, %bb19
	%9 = load i32, ptr %3, align 4		; <i32> [#uses=1]
	br i1 false, label %bb6, label %bb19

bb39.split:		; preds = %bb37.outer
	%10 = icmp eq ptr null, %bufptr.1.ph		; <i1> [#uses=1]
	br i1 %10, label %bb44, label %bb42

bb42:		; preds = %bb39.split
	call void @quux(ptr %bufptr.1.ph) nounwind nounwind
	ret ptr null

bb44:		; preds = %bb39.split, %entry
	%.0 = phi ptr [ @0, %entry ], [ null, %bb39.split ]		; <ptr> [#uses=1]
	ret ptr %.0
}

declare void @foo(ptr)

declare void @quux(ptr)
