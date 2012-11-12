; RUN: llc < %s -mtriple=armv7-apple-ios -mcpu=cortex-a9 -stress-ivchain | FileCheck %s
; REQUIRES: asserts

; @sharedidx is an unrolled variant of this loop:
;  for (unsigned long i = 0; i < len; i += s) {
;    c[i] = a[i] + b[i];
;  }
; where 's' cannot be folded into the addressing mode.
;
; This is not quite profitable to chain. But with -stress-ivchain, we
; can form three address chains in place of the shared induction
; variable.

; rdar://10674430
define void @sharedidx(i8* nocapture %a, i8* nocapture %b, i8* nocapture %c, i32 %s, i32 %len) nounwind ssp {
entry:
; CHECK: sharedidx:
  %cmp8 = icmp eq i32 %len, 0
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body.3
; CHECK: %for.body
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
  %i.09 = phi i32 [ %add5.3, %for.body.3 ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8* %a, i32 %i.09
  %0 = load i8* %arrayidx, align 1
  %conv6 = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i8* %b, i32 %i.09
  %1 = load i8* %arrayidx1, align 1
  %conv27 = zext i8 %1 to i32
  %add = add nsw i32 %conv27, %conv6
  %conv3 = trunc i32 %add to i8
  %arrayidx4 = getelementptr inbounds i8* %c, i32 %i.09
  store i8 %conv3, i8* %arrayidx4, align 1
  %add5 = add i32 %i.09, %s
  %cmp = icmp ult i32 %add5, %len
  br i1 %cmp, label %for.body.1, label %for.end

for.end:                                          ; preds = %for.body, %for.body.1, %for.body.2, %for.body.3, %entry
  ret void

for.body.1:                                       ; preds = %for.body
; CHECK: %for.body.1
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
  %arrayidx.1 = getelementptr inbounds i8* %a, i32 %add5
  %2 = load i8* %arrayidx.1, align 1
  %conv6.1 = zext i8 %2 to i32
  %arrayidx1.1 = getelementptr inbounds i8* %b, i32 %add5
  %3 = load i8* %arrayidx1.1, align 1
  %conv27.1 = zext i8 %3 to i32
  %add.1 = add nsw i32 %conv27.1, %conv6.1
  %conv3.1 = trunc i32 %add.1 to i8
  %arrayidx4.1 = getelementptr inbounds i8* %c, i32 %add5
  store i8 %conv3.1, i8* %arrayidx4.1, align 1
  %add5.1 = add i32 %add5, %s
  %cmp.1 = icmp ult i32 %add5.1, %len
  br i1 %cmp.1, label %for.body.2, label %for.end

for.body.2:                                       ; preds = %for.body.1
; CHECK: %for.body.2
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
  %arrayidx.2 = getelementptr inbounds i8* %a, i32 %add5.1
  %4 = load i8* %arrayidx.2, align 1
  %conv6.2 = zext i8 %4 to i32
  %arrayidx1.2 = getelementptr inbounds i8* %b, i32 %add5.1
  %5 = load i8* %arrayidx1.2, align 1
  %conv27.2 = zext i8 %5 to i32
  %add.2 = add nsw i32 %conv27.2, %conv6.2
  %conv3.2 = trunc i32 %add.2 to i8
  %arrayidx4.2 = getelementptr inbounds i8* %c, i32 %add5.1
  store i8 %conv3.2, i8* %arrayidx4.2, align 1
  %add5.2 = add i32 %add5.1, %s
  %cmp.2 = icmp ult i32 %add5.2, %len
  br i1 %cmp.2, label %for.body.3, label %for.end

for.body.3:                                       ; preds = %for.body.2
; CHECK: %for.body.3
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
; CHECK: ldrb {{r[0-9]|lr}}, [{{r[0-9]|lr}}, {{r[0-9]|lr}}]!
  %arrayidx.3 = getelementptr inbounds i8* %a, i32 %add5.2
  %6 = load i8* %arrayidx.3, align 1
  %conv6.3 = zext i8 %6 to i32
  %arrayidx1.3 = getelementptr inbounds i8* %b, i32 %add5.2
  %7 = load i8* %arrayidx1.3, align 1
  %conv27.3 = zext i8 %7 to i32
  %add.3 = add nsw i32 %conv27.3, %conv6.3
  %conv3.3 = trunc i32 %add.3 to i8
  %arrayidx4.3 = getelementptr inbounds i8* %c, i32 %add5.2
  store i8 %conv3.3, i8* %arrayidx4.3, align 1
  %add5.3 = add i32 %add5.2, %s
  %cmp.3 = icmp ult i32 %add5.3, %len
  br i1 %cmp.3, label %for.body, label %for.end
}
