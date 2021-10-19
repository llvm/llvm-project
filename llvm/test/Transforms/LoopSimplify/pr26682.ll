; RUN: opt < %s -lcssa -loop-simplify -indvars -S | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

@a = external global i32, align 4

; Check that loop-simplify merges two loop exits, but preserves LCSSA form.
; CHECK-LABEL: @foo
; CHECK: for:
; CHECK: %or.cond = select i1 %cmp1, i1 %cmp2, i1 false
; CHECK-NOT: for.cond:
; CHECK: for.end:
; CHECK: %a.lcssa = phi i32 [ %a, %for ]
define i32 @foo(i32 %x) {
entry:
  br label %for

for:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.cond ]
  %cmp1 = icmp eq i32 %x, 0
  %iv.next = add nuw nsw i32 %iv, 1
  %a = load i32, i32* @a
  br i1 %cmp1, label %for.cond, label %for.end

for.cond:
  %cmp2 = icmp slt i32 %iv.next, 4
  br i1 %cmp2, label %for, label %for.end

for.end:
  %a.lcssa = phi i32 [ %a, %for ], [ %a, %for.cond ]
  ret i32 %a.lcssa
}
