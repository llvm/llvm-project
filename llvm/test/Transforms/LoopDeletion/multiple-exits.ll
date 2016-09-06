; RUN: opt < %s -loop-deletion -S | FileCheck %s

; Checks whether dead loops with multiple exits can be eliminated

define void @foo(i64 %n, i64 %m) nounwind {
; CHECK-LABEL: @foo(
; CHECK:      entry:
; CHECK-NEXT:   br label %return

; CHECK:      return:
; CHECK-NEXT:   ret void
entry:
  br label %bb

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb2 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
bb2:
  %t2 = icmp slt i64 %x.0, %m
  br i1 %t1, label %bb, label %return

return:
  ret void
}

define i64 @bar(i64 %n, i64 %m) nounwind {
; CHECK-LABEL:  @bar(
; CHECK: entry:
; CHECK-NEXT:  br label %return

; CHECK: return:
; CHECK-NEXT:  ret i64 10

entry:
  br label %bb

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb3 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
bb2:
  %t2 = icmp slt i64 %x.0, %m
  br i1 %t2, label %bb3, label %return
bb3:
  %t3 = icmp slt i64 %x.0, %m
  br i1 %t3, label %bb, label %return

return:
  %x.lcssa = phi i64 [ 10, %bb ], [ 10, %bb2 ], [ 10, %bb3 ]
  ret i64 %x.lcssa
}

define i64 @baz(i64 %n, i64 %m) nounwind {
; CHECK-LABEL:  @baz(
; CHECK: return:
; CHECK-NEXT:  %x.lcssa = phi i64 [ 12, %bb ], [ 10, %bb2 ]
; CHECK-NEXT:  ret i64 %x.lcssa

entry:
  br label %bb

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb3 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
bb2:
  %t2 = icmp slt i64 %x.0, %m
  br i1 %t2, label %bb3, label %return
bb3:
  %t3 = icmp slt i64 %x.0, %m
  br i1 %t3, label %bb, label %return

return:
  %x.lcssa = phi i64 [ 12, %bb ], [ 10, %bb2 ], [ 10, %bb3 ]
  ret i64 %x.lcssa
}
