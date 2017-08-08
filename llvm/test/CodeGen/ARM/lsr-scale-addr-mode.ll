; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s
; Should use scaled addressing mode.

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a53 %s -o - | FileCheck %s -check-prefix CHECK-NONEGOFF
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a57 %s -o - | FileCheck %s -check-prefix CHECK-NONEGOFF
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-r52 %s -o - | FileCheck %s -check-prefix CHECK-NONEGOFF
; Should not generate negated register offset

define void @sintzero(i32* %a) nounwind {
entry:
	store i32 0, i32* %a
	br label %cond_next

cond_next:		; preds = %cond_next, %entry
	%indvar = phi i32 [ 0, %entry ], [ %tmp25, %cond_next ]		; <i32> [#uses=1]
	%tmp25 = add i32 %indvar, 1		; <i32> [#uses=3]
	%tmp36 = getelementptr i32, i32* %a, i32 %tmp25		; <i32*> [#uses=1]
	store i32 0, i32* %tmp36
	icmp eq i32 %tmp25, -1		; <i1>:0 [#uses=1]
	br i1 %0, label %return, label %cond_next

return:		; preds = %cond_next
	ret void
}

; CHECK: lsl{{.*}}#2]
; CHECK-NONEGOFF: [{{r[0-9]+}}, {{r[0-9]+}}, lsl{{.*}}#2]

