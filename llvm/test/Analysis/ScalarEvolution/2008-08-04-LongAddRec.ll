; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0 2>&1 | FileCheck %s
; PR2621

define i32 @a() nounwind  {
entry:
	br label %bb1

bb:		; preds = %bb1
	add i16 %x17.0, 1		; <i16>:0 [#uses=2]
	add i16 %0, %x16.0		; <i16>:1 [#uses=2]
	add i16 %1, %x15.0		; <i16>:2 [#uses=2]
	add i16 %2, %x14.0		; <i16>:3 [#uses=2]
	add i16 %3, %x13.0		; <i16>:4 [#uses=2]
	add i16 %4, %x12.0		; <i16>:5 [#uses=2]
	add i16 %5, %x11.0		; <i16>:6 [#uses=2]
	add i16 %6, %x10.0		; <i16>:7 [#uses=2]
	add i16 %7, %x9.0		; <i16>:8 [#uses=2]
	add i16 %8, %x8.0		; <i16>:9 [#uses=2]
	add i16 %9, %x7.0		; <i16>:10 [#uses=2]
	add i16 %10, %x6.0		; <i16>:11 [#uses=2]
	add i16 %11, %x5.0		; <i16>:12 [#uses=2]
	add i16 %12, %x4.0		; <i16>:13 [#uses=2]
	add i16 %13, %x3.0		; <i16>:14 [#uses=2]
	add i16 %14, %x2.0		; <i16>:15 [#uses=2]
	add i16 %15, %x1.0		; <i16>:16 [#uses=1]
	add i32 %i.0, 1		; <i32>:17 [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%x2.0 = phi i16 [ 0, %entry ], [ %15, %bb ]		; <i16> [#uses=1]
	%x3.0 = phi i16 [ 0, %entry ], [ %14, %bb ]		; <i16> [#uses=1]
	%x4.0 = phi i16 [ 0, %entry ], [ %13, %bb ]		; <i16> [#uses=1]
	%x5.0 = phi i16 [ 0, %entry ], [ %12, %bb ]		; <i16> [#uses=1]
	%x6.0 = phi i16 [ 0, %entry ], [ %11, %bb ]		; <i16> [#uses=1]
	%x7.0 = phi i16 [ 0, %entry ], [ %10, %bb ]		; <i16> [#uses=1]
	%x8.0 = phi i16 [ 0, %entry ], [ %9, %bb ]		; <i16> [#uses=1]
	%x9.0 = phi i16 [ 0, %entry ], [ %8, %bb ]		; <i16> [#uses=1]
	%x10.0 = phi i16 [ 0, %entry ], [ %7, %bb ]		; <i16> [#uses=1]
	%x11.0 = phi i16 [ 0, %entry ], [ %6, %bb ]		; <i16> [#uses=1]
	%x12.0 = phi i16 [ 0, %entry ], [ %5, %bb ]		; <i16> [#uses=1]
	%x13.0 = phi i16 [ 0, %entry ], [ %4, %bb ]		; <i16> [#uses=1]
	%x14.0 = phi i16 [ 0, %entry ], [ %3, %bb ]		; <i16> [#uses=1]
	%x15.0 = phi i16 [ 0, %entry ], [ %2, %bb ]		; <i16> [#uses=1]
	%x16.0 = phi i16 [ 0, %entry ], [ %1, %bb ]		; <i16> [#uses=1]
	%x17.0 = phi i16 [ 0, %entry ], [ %0, %bb ]		; <i16> [#uses=1]
	%i.0 = phi i32 [ 0, %entry ], [ %17, %bb ]		; <i32> [#uses=2]
	%x1.0 = phi i16 [ 0, %entry ], [ %16, %bb ]		; <i16> [#uses=2]
	icmp ult i32 %i.0, 8888		; <i1>:18 [#uses=1]
	br i1 %18, label %bb, label %bb2

bb2:		; preds = %bb1
	zext i16 %x1.0 to i32		; <i32>:19 [#uses=1]
	ret i32 %19
}

; CHECK: Exits: -19168

