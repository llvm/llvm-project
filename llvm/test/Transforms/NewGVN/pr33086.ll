; RUN: opt -newgvn -S %s | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: define void @tinkywinky() {
; CHECK: entry:
; CHECK-NEXT:   br i1 undef, label %for.cond18, label %for.cond.preheader
; CHECK: for.cond.preheader:
; CHECK-NEXT:   br label %for.cond2thread-pre-split
; CHECK: for.cond2thread-pre-split:
; CHECK-NEXT:   %conv24 = phi i32 [ 0, %for.cond.preheader ], [ %conv, %for.inc.split ]
; CHECK-NEXT:   br label %for.inc.split
; CHECK: for.inc.split:
; CHECK-NEXT:   %add = shl nsw i32 %conv24, 16
; CHECK-NEXT:   %sext23 = add i32 %add, 65536
; CHECK-NEXT:   %conv = ashr exact i32 %sext23, 16
; CHECK-NEXT:   %cmp = icmp slt i32 %sext23, 3604480
; CHECK-NEXT:   br i1 %cmp, label %for.cond2thread-pre-split, label %l1.loopexit
; CHECK: l1.loopexit:
; CHECK-NEXT:   br label %l1
; CHECK: l1:
; CHECK-NEXT:   %0 = load i16, i16* null, align 2
; CHECK-NEXT:   %g.0.g.0..pr = load i16, i16* null, align 2
; CHECK-NEXT:   ret void
; CHECK: for.cond18:
; CHECK-NEXT:   br label %l1
; CHECK-NEXT: }

define void @tinkywinky() {
entry:
  br i1 undef, label %for.cond18, label %for.cond.preheader

for.cond.preheader:
  br label %for.cond2thread-pre-split

for.cond2thread-pre-split:
  %conv24 = phi i32 [ 0, %for.cond.preheader ], [ %conv, %for.inc.split ]
  br label %for.inc.split

for.inc.split:
  %add = shl nsw i32 %conv24, 16
  %sext23 = add i32 %add, 65536
  %conv = ashr exact i32 %sext23, 16
  %cmp = icmp slt i32 %sext23, 3604480
  br i1 %cmp, label %for.cond2thread-pre-split, label %l1.loopexit

l1.loopexit:
  br label %l1

l1:
  %h.0 = phi i16* [ undef, %for.cond18 ], [ null, %l1.loopexit ]
  %0 = load i16, i16* %h.0, align 2
  store i16 %0, i16* null, align 2
  %g.0.g.0..pr = load i16, i16* null, align 2
  %tobool15 = icmp eq i16 %g.0.g.0..pr, 0
  ret void

for.cond18:
  br label %l1
}
