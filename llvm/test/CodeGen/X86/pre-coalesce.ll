; RUN: llc -regalloc=greedy -mtriple=x86_64-unknown-linux-gnu  < %s -o - | FileCheck %s
;
; The test is to check no redundent mov as follows will be generated in %while.body loop.
;  .LBB0_2:
;    movsbl	%cl, %ecx
;    movl	%edx, %eax   ==> This movl can be promoted outside of loop.
;    shll	$5, %eax
;    ...
;    movl	%eax, %edx
;    jne     .LBB0_2
;
; CHECK-LABEL: foo:
; CHECK: [[L0:.LBB0_[0-9]+]]: # %while.body
; CHECK: movl %[[REGA:.*]], %[[REGB:.*]]
; CHECK-NOT: movl %[[REGB]], %[[REGA]]
; CHECK: jne [[L0]]
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@b = common local_unnamed_addr global i8* null, align 8
@a = common local_unnamed_addr global i32 0, align 4

define i32 @foo() local_unnamed_addr {
entry:
  %t0 = load i8*, i8** @b, align 8
  %t1 = load i8, i8* %t0, align 1
  %cmp4 = icmp eq i8 %t1, 0
  %t2 = load i32, i32* @a, align 4
  br i1 %cmp4, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %t3 = phi i32 [ %add3, %while.body ], [ %t2, %while.body.preheader ]
  %t4 = phi i8 [ %t5, %while.body ], [ %t1, %while.body.preheader ]
  %conv = sext i8 %t4 to i32
  %add = mul i32 %t3, 33
  %add3 = add nsw i32 %add, %conv
  store i32 %add3, i32* @a, align 4
  %t5 = load i8, i8* %t0, align 1
  %cmp = icmp eq i8 %t5, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %.lcssa = phi i32 [ %t2, %entry ], [ %add3, %while.end.loopexit ]
  ret i32 %.lcssa
}
