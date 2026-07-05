; RUN: llc < %s -mtriple=arm-linux-gnueabi
; PR4528

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv6-elf"

define i32 @file_read_actor(ptr nocapture %desc, ptr %page, i32 %offset, i32 %size) nounwind optsize {
entry:
	br i1 undef, label %fault_in_pages_writeable.exit, label %bb5.i

bb5.i:		; preds = %entry
	%asmtmp.i = tail call i32 asm sideeffect "1:\09strbt\09$1,[$2]\0A2:\0A\09.section .fixup,\22ax\22\0A\09.align\092\0A3:\09mov\09$0, $3\0A\09b\092b\0A\09.previous\0A\09.section __ex_table,\22a\22\0A\09.align\093\0A\09.long\091b, 3b\0A\09.previous", "=r,r,r,i,0,~{cc}"(i8 0, i32 undef, i32 -14, i32 0) nounwind		; <i32> [#uses=1]
	%0 = icmp eq i32 %asmtmp.i, 0		; <i1> [#uses=1]
	br i1 %0, label %bb6.i, label %fault_in_pages_writeable.exit

bb6.i:		; preds = %bb5.i
	br i1 undef, label %fault_in_pages_writeable.exit, label %bb7.i

bb7.i:		; preds = %bb6.i
	unreachable

fault_in_pages_writeable.exit:		; preds = %bb6.i, %bb5.i, %entry
	br i1 undef, label %bb2, label %bb3

bb2:		; preds = %fault_in_pages_writeable.exit
	unreachable

bb3:		; preds = %fault_in_pages_writeable.exit
	%1 = tail call  i32 @__copy_to_user(ptr undef, ptr undef, i32 undef) nounwind		; <i32> [#uses=0]
	unreachable
}

declare i32 @__copy_to_user(ptr, ptr, i32)
