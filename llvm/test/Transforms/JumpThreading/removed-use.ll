; RUN: opt -S < %s -jump-threading | FileCheck %s
; CHECK-LABEL: @foo
; CHECK: bb6:
; CHECK-NEXT: ret void
; CHECK: bb3:
; CHECK: br label %bb3
define void @foo() {
entry:
  br i1 true, label %bb6, label %bb3

bb3:
  %x0 = phi i32 [ undef, %entry ], [ %x1, %bb5 ]
  %y  = and i64 undef, 1
  %p  = icmp ne i64 %y, 0
  br i1 %p, label %bb4, label %bb5

bb4:
  br label %bb5

bb5:
  %x1 = phi i32 [ %x0, %bb3 ], [ %x0, %bb4 ]
  %z  = phi i32 [ 0, %bb3 ], [ 1, %bb4 ]
  %q  = icmp eq i32 %z, 0
  br i1 %q, label %bb3, label %bb6

bb6:
  ret void
}

; CHECK-LABEL: @bar
; Just check that we don't crash on this test.
define void @bar(i1 %p) {
entry:
  br i1 false, label %bb2, label %exit

bb2:
  %x0 = phi i32 [ undef, %entry ], [ %x1, %bb5 ]
  br i1 %p, label %bb3, label %bb4

bb3:
  br label %bb5

bb4:
  br label %bb5

bb5:
  %x1 = phi i32 [ %x0, %bb3 ], [ 0, %bb4 ]
  switch i32 %x1, label %exit [
    i32 10, label %bb2
  ]

exit:
  ret void
}
