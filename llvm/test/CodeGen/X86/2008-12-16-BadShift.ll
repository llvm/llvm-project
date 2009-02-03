; RUN: llvm-as < %s | llc | not grep shrl
; Note: this test is really trying to make sure that the shift
; returns the right result; shrl is most likely wrong,
; but if CodeGen starts legitimately using an shrl here,
; please adjust the test appropriately.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@.str = internal constant [6 x i8] c"%lld\0A\00"		; <[6 x i8]*> [#uses=1]

define i64 @mebbe_shift(i32 %xx, i32 %test) nounwind {
entry:
	%conv = zext i32 %xx to i64		; <i64> [#uses=1]
	%tobool = icmp ne i32 %test, 0		; <i1> [#uses=1]
	%shl = select i1 %tobool, i64 3, i64 0		; <i64> [#uses=1]
	%x.0 = shl i64 %conv, %shl		; <i64> [#uses=1]
	ret i64 %x.0
}

