; Test shrink wrapping placement is correct with respect to calls to llvm.{stacksave,stackrestore}

; void f(int n, int x[]) {
;   if (n < 0)
;     return;
;
;  int a[n];
;
;  for (int i = 0; i < n; i++)
;    a[i] = x[n - i - 1];
;
;  for (int i = 0; i < n; i++)
;    x[i] = a[i] + 1;
; }
;
; RUN: llc -mtriple aarch64-linux %s -o - | FileCheck %s

define dso_local void @f(i32 %n, i32* nocapture %x) {
entry:
  %cmp = icmp slt i32 %n, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %0 = zext i32 %n to i64
  %1 = tail call i8* @llvm.stacksave()
  %vla = alloca i32, i64 %0, align 16
  %cmp132 = icmp eq i32 %n, 0
  br i1 %cmp132, label %for.cond.cleanup8, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.end
  %sub = add i32 %n, -1
  br label %for.body

for.cond6.preheader:                              ; preds = %for.body
  %cmp730 = icmp sgt i32 %n, 0
  br i1 %cmp730, label %for.body9, label %for.cond.cleanup8

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv34 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next35, %for.body ]
  %2 = trunc i64 %indvars.iv34 to i32
  %sub2 = sub i32 %sub, %2
  %idxprom = sext i32 %sub2 to i64
  %arrayidx = getelementptr inbounds i32, i32* %x, i64 %idxprom
  %3 = load i32, i32* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %vla, i64 %indvars.iv34
  store i32 %3, i32* %arrayidx4, align 4
  %indvars.iv.next35 = add nuw nsw i64 %indvars.iv34, 1
  %exitcond37 = icmp eq i64 %indvars.iv.next35, %0
  br i1 %exitcond37, label %for.cond6.preheader, label %for.body

for.cond.cleanup8:                                ; preds = %for.body9, %if.end, %for.cond6.preheader
  tail call void @llvm.stackrestore(i8* %1)
  br label %return

for.body9:                                        ; preds = %for.cond6.preheader, %for.body9
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body9 ], [ 0, %for.cond6.preheader ]
  %arrayidx11 = getelementptr inbounds i32, i32* %vla, i64 %indvars.iv
  %4 = load i32, i32* %arrayidx11, align 4
  %add = add nsw i32 %4, 1
  %arrayidx13 = getelementptr inbounds i32, i32* %x, i64 %indvars.iv
  store i32 %add, i32* %arrayidx13, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %0
  br i1 %exitcond, label %for.cond.cleanup8, label %for.body9

return:                                           ; preds = %entry, %for.cond.cleanup8
  ret void
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave()

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*)

; Check that llvm.stackrestore() happens before CSRs are popped off the stack

; CHECK-LABEL: f

; CHECK:      stp x29, x30, [sp, #-16]!
; CHECK-NEXT: mov x29, sp

; VLA allocation
; CHECK: mov [[X2:x[0-9]+]], sp
; CHECK: mov [[SAVE:x[0-9]+]], sp
; CHECK: add [[X1:x[0-9]+]], [[X1]], #15
; CHECK: and [[X1]], [[X1]], #0x7fffffff0
; Saving the SP via llvm.stacksave()
; CHECK: sub [[X2]], [[X2]], [[X1]]

; The next instruction comes from llvm.stackrestore()
; CHECK:      mov sp, [[SAVE]]
; Epilogue
; CHECK-NEXT: mov sp, x29
; CHECK-NEXT: ldp x29, x30, [sp], #16
