; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

; CHECK: @t
; CHECK: blr
define i32 @t(i64 %byteStart, i32 %activeIndex) nounwind  {
entry:
	%tmp50 = load i32, i32* null, align 4		; <i32> [#uses=1]
	%tmp5051 = zext i32 %tmp50 to i64		; <i64> [#uses=3]
	%tmp53 = udiv i64 %byteStart, %tmp5051		; <i64> [#uses=1]
	%tmp5354 = trunc i64 %tmp53 to i32		; <i32> [#uses=1]
	%tmp62 = urem i64 %byteStart, %tmp5051		; <i64> [#uses=1]
	%tmp94 = add i32 0, 1		; <i32> [#uses=1]
	%tmp100 = urem i32 %tmp94, 0		; <i32> [#uses=2]
	%tmp108 = add i32 0, %activeIndex		; <i32> [#uses=1]
	%tmp110 = sub i32 %tmp108, 0		; <i32> [#uses=1]
	%tmp112 = urem i32 %tmp110, 0		; <i32> [#uses=2]
	%tmp122 = icmp ult i32 %tmp112, %tmp100		; <i1> [#uses=1]
	%iftmp.175.0 = select i1 %tmp122, i32 %tmp112, i32 %tmp100		; <i32> [#uses=1]
	%tmp119 = add i32 %tmp5354, 0		; <i32> [#uses=1]
	%tmp131 = add i32 %tmp119, %iftmp.175.0		; <i32> [#uses=1]
	%tmp131132 = zext i32 %tmp131 to i64		; <i64> [#uses=1]
	%tmp147 = mul i64 %tmp131132, %tmp5051		; <i64> [#uses=1]
	br i1 false, label %bb164, label %bb190
bb164:		; preds = %entry
	%tmp171172 = and i64 %tmp62, 4294967295		; <i64> [#uses=1]
	%tmp173 = add i64 %tmp171172, %tmp147		; <i64> [#uses=0]
	ret i32 0
bb190:		; preds = %entry
	ret i32 0
}
