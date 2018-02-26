; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-none-eabi"

; These are regression tests for
;  https://bugs.llvm.org/show_bug.cgi?id=34106
;    "ARMTargetLowering::isLegalAddressingMode can accept incorrect
;    addressing modes for Thumb1 target"
;  https://reviews.llvm.org/D34583
;    "[LSR] Narrow search space by filtering non-optimal formulae with the
;    same ScaledReg and Scale."
;
; Due to a bug in ARMTargetLowering::isLegalAddressingMode LSR got 
; 4*reg({0,+,-1}) and -4*reg({0,+,-1}) had the same cost for the Thumb1 target.
; Another issue was that LSR got that -1*reg was free for the Thumb1 target.

; Test case 01: -1*reg is not free for the Thumb1 target.
; 
; CHECK-LABEL: @negativeOneCase
; CHECK-NOT: mul
; CHECK: ret i8
define i8* @negativeOneCase(i8* returned %a, i8* nocapture readonly %b, i32 %n) nounwind {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %a, i32 -1
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %p.0 = phi i8* [ %add.ptr, %entry ], [ %incdec.ptr, %while.cond ]
  %incdec.ptr = getelementptr inbounds i8, i8* %p.0, i32 1
  %0 = load i8, i8* %incdec.ptr, align 1
  %cmp = icmp eq i8 %0, 0
  br i1 %cmp, label %while.cond2.preheader, label %while.cond

while.cond2.preheader:                            ; preds = %while.cond
  br label %while.cond2

while.cond2:                                      ; preds = %while.cond2.preheader, %while.body5
  %b.addr.0 = phi i8* [ %incdec.ptr6, %while.body5 ], [ %b, %while.cond2.preheader ]
  %n.addr.0 = phi i32 [ %dec, %while.body5 ], [ %n, %while.cond2.preheader ]
  %p.1 = phi i8* [ %incdec.ptr7, %while.body5 ], [ %incdec.ptr, %while.cond2.preheader ]
  %cmp3 = icmp eq i32 %n.addr.0, 0
  br i1 %cmp3, label %while.end8, label %while.body5

while.body5:                                      ; preds = %while.cond2
  %dec = add i32 %n.addr.0, -1
  %incdec.ptr6 = getelementptr inbounds i8, i8* %b.addr.0, i32 1
  %1 = load i8, i8* %b.addr.0, align 1
  %incdec.ptr7 = getelementptr inbounds i8, i8* %p.1, i32 1
  store i8 %1, i8* %p.1, align 1
  br label %while.cond2

while.end8:                                       ; preds = %while.cond2
  %scevgep = getelementptr i8, i8* %incdec.ptr, i32 %n
  store i8 0, i8* %scevgep, align 1
  ret i8* %a
}

; Test case 02: 4*reg({0,+,-1}) and -4*reg({0,+,-1}) are not supported for
;               the Thumb1 target.
; 
; CHECK-LABEL: @negativeFourCase
; CHECK-NOT: mul
; CHECK: ret void
define void @negativeFourCase(i8* %ptr1, i32* %ptr2) nounwind {
entry:
  br label %for.cond6.preheader.us.i.i

for.cond6.preheader.us.i.i:                       ; preds = %if.end48.us.i.i, %entry
  %addr.0108.us.i.i = phi i8* [ %scevgep.i.i, %if.end48.us.i.i ], [ %ptr1, %entry ]
  %inc49.us.i.i = phi i32 [ %inc50.us.i.i, %if.end48.us.i.i ], [ 0, %entry ]
  %c1.0104.us.i.i = phi i32* [ %c0.0103.us.i.i, %if.end48.us.i.i ], [ %ptr2, %entry ]
  %c0.0103.us.i.i = phi i32* [ %c1.0104.us.i.i, %if.end48.us.i.i ], [ %ptr2, %entry ]
  br label %for.body8.us.i.i

if.end48.us.i.i:                                  ; preds = %for.inc.us.i.i
  %scevgep.i.i = getelementptr i8, i8* %addr.0108.us.i.i, i32 256
  %inc50.us.i.i = add nuw nsw i32 %inc49.us.i.i, 1
  %exitcond110.i.i = icmp eq i32 %inc50.us.i.i, 256
  br i1 %exitcond110.i.i, label %exit.i, label %for.cond6.preheader.us.i.i

for.body8.us.i.i:                                 ; preds = %for.inc.us.i.i, %for.cond6.preheader.us.i.i
  %addr.198.us.i.i = phi i8* [ %addr.0108.us.i.i, %for.cond6.preheader.us.i.i ], [ %incdec.ptr.us.i.i, %for.inc.us.i.i ]
  %inc.196.us.i.i = phi i32 [ 0, %for.cond6.preheader.us.i.i ], [ %inc.2.us.i.i, %for.inc.us.i.i ]
  %c.093.us.i.i = phi i32 [ 0, %for.cond6.preheader.us.i.i ], [ %inc43.us.i.i, %for.inc.us.i.i ]
  %incdec.ptr.us.i.i = getelementptr inbounds i8, i8* %addr.198.us.i.i, i32 1
  %0 = load i8, i8* %addr.198.us.i.i, align 1
  %cmp9.us.i.i = icmp eq i8 %0, -1
  br i1 %cmp9.us.i.i, label %if.end37.us.i.i, label %if.else.us.i.i

if.else.us.i.i:                                   ; preds = %for.body8.us.i.i
  %add12.us.i.i = add nuw nsw i32 %c.093.us.i.i, 1
  %arrayidx13.us.i.i = getelementptr inbounds i32, i32* %c1.0104.us.i.i, i32 %add12.us.i.i
  %1 = load i32, i32* %arrayidx13.us.i.i, align 4
  %arrayidx16.us.i.i = getelementptr inbounds i32, i32* %c1.0104.us.i.i, i32 %c.093.us.i.i
  %2 = load i32, i32* %arrayidx16.us.i.i, align 4
  %sub19.us.i.i = add nsw i32 %c.093.us.i.i, -1
  %arrayidx20.us.i.i = getelementptr inbounds i32, i32* %c1.0104.us.i.i, i32 %sub19.us.i.i
  %3 = load i32, i32* %arrayidx20.us.i.i, align 4
  br label %if.end37.us.i.i

if.end37.us.i.i:                                  ; preds = %if.else.us.i.i, %for.body8.us.i.i
  %4 = phi i32 [ %3, %if.else.us.i.i ], [ 0, %for.body8.us.i.i ]
  %arrayidx36.us.i.i = getelementptr inbounds i32, i32* %c0.0103.us.i.i, i32 %c.093.us.i.i
  store i32 %4, i32* %arrayidx36.us.i.i, align 4
  %inc.us.i.i = add nsw i32 %inc.196.us.i.i, 1
  %cmp38.us.i.i = icmp sgt i32 %inc.196.us.i.i, 6
  br i1 %cmp38.us.i.i, label %if.then40.us.i.i, label %for.inc.us.i.i

if.then40.us.i.i:                                 ; preds = %if.end37.us.i.i
  br label %for.inc.us.i.i

for.inc.us.i.i:                                   ; preds = %if.then40.us.i.i, %if.end37.us.i.i
  %inc.2.us.i.i = phi i32 [ 0, %if.then40.us.i.i ], [ %inc.us.i.i, %if.end37.us.i.i ]
  %inc43.us.i.i = add nuw nsw i32 %c.093.us.i.i, 1
  %exitcond.i.i = icmp eq i32 %inc43.us.i.i, 256
  br i1 %exitcond.i.i, label %if.end48.us.i.i, label %for.body8.us.i.i

exit.i:                               ; preds = %if.end48.us.i.i
  ret void
}

