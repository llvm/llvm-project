; The goal of this test is checking if LastCallToStaticBonus is applied
; correctly while deciding inline deferral. For the test code below, when
; inliner evaluates the callsite of bar->baz, it checks if inlining of bar->baz
; prevents ininling of foo->bar, even when foo->bar inlining is more beneficial
; than bar->baz inlining. As LastCallToStaticBonus has a massive value, and
; both baz and bar has only one caller, the cost of foo->bar inlining and
; bar->baz inlining should be non-trivial for inliner to compute that bar->baz
; inlining can actaully prevent foo->bar inlining. To make the cost of these
; callsites big enough, loop unrolling pass with very high threshold is used to
; preprocess the test.

; RUN: opt < %s -loop-unroll -inline -unroll-threshold=15000 -inline-threshold=250 -S | FileCheck %s
; RUN: opt < %s -passes='function(require<opt-remark-emit>,unroll),require<profile-summary>,cgscc(inline)' -unroll-threshold=15000 -inline-threshold=250 -S | FileCheck %s
; CHECK-LABEL: define internal i32 @bar()

define internal i32 @baz() {
entry:
  br label %bb1

bb1:
  %ind = phi i32 [ 0, %entry ], [ %inc, %bb1 ]
  call void @extern()
  %inc = add nsw i32 %ind, 1
  %cmp = icmp sgt i32 %inc, 510
  br i1 %cmp, label %ret, label %bb1

ret:
  ret i32 0
}

define internal i32 @bar() {
entry:
  br label %bb1

bb1:
  %ind = phi i32 [ 0, %entry ], [ %inc, %bb1 ]
  call void @extern()
  %inc = add nsw i32 %ind, 1
  %cmp = icmp sgt i32 %inc, 510
  br i1 %cmp, label %ret, label %bb1

ret:
  call i32 @baz()
  ret i32 0
}

define i32 @foo() {
entry:
  call i32 @bar()
  ret i32 0
}

declare void @extern()
