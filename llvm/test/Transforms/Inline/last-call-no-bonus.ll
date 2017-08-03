; This code is virtually identical to last-call-bonus.ll, but the callsites
; to the internal functions are cold, thereby preventing the last call to
; static bonus from being applied.

; RUN: opt < %s -passes='function(require<opt-remark-emit>,unroll),require<profile-summary>,cgscc(inline)' -unroll-threshold=15000 -inline-threshold=250 -S | FileCheck %s

; CHECK-LABEL: define internal i32 @baz
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

; CHECK-LABEL: define internal i32 @bar
define internal i32 @bar(i1 %b) {
entry:
  br label %bb1

bb1:
  %ind = phi i32 [ 0, %entry ], [ %inc, %bb1 ]
  call void @extern()
  %inc = add nsw i32 %ind, 1
  %cmp = icmp sgt i32 %inc, 510
  br i1 %cmp, label %for.exit, label %bb1

for.exit:
  br i1 %b, label %bb2, label %ret, !prof !0
bb2:
; CHECK: call i32 @baz
  call i32 @baz()
  br label %ret
ret:
  ret i32 0
}
; CHECK-LABEL: define i32 @foo
define i32 @foo(i1 %b) {
entry:
  br i1 %b, label %bb1, label %ret, !prof !0
bb1:
; CHECK: call i32 @bar
  call i32 @bar(i1 %b)
  br label %ret
ret:
  ret i32 0
}

declare void @extern()

!0 = !{!"branch_weights", i32 1, i32 2500}
