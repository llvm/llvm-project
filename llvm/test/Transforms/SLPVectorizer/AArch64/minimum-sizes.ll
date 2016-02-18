; RUN: opt -S -slp-vectorizer < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; This test ensures that we do not regress due to PR26364. The vectorizer
; should not compute a smaller size for %k.13 since it is in a use-def cycle
; and cannot be demoted.
;
; CHECK-LABEL: @PR26364
; CHECK: %k.13 = phi i32
;
define fastcc void @PR26364() {
entry:
  br i1 undef, label %for.end11, label %for.cond4

for.cond4:
  %k.13 = phi i32 [ undef, %entry ], [ %k.3, %for.cond4 ]
  %e.02 = phi i32 [ 1, %entry ], [ 0, %for.cond4 ]
  %e.1 = select i1 undef, i32 %e.02, i32 0
  %k.3 = select i1 undef, i32 %k.13, i32 undef
  br label %for.cond4

for.end11:
  ret void
}

; This test ensures that we do not regress due to PR26629. We must look at
; every root in the vectorizable tree when computing minimum sizes since one
; root may require fewer bits than another.
;
; CHECK-LABEL: @PR26629
; CHECK-NOT: {{.*}} and <2 x i72>
;
define void @PR26629(i32* %c) {
entry:
  br i1 undef, label %for.ph, label %for.end

for.ph:
  %0 = load i32, i32* %c, align 4
  br label %for.body

for.body:
  %d = phi i72 [ 576507472957710340, %for.ph ], [ %bf.set17, %for.body ]
  %sub = sub i32 %0, undef
  %bf.clear13 = and i72 %d, -576460748008464384
  %1 = zext i32 %sub to i72
  %bf.value15 = and i72 %1, 8191
  %bf.clear16 = or i72 %bf.value15, %bf.clear13
  %bf.set17 = or i72 %bf.clear16, undef
  br label %for.body

for.end:
  ret void
}
