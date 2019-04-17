; RUN: opt -jump-threading -simplifycfg -S < %s | FileCheck %s
; CHECK-NOT: bb6:
; CHECK-NOT: bb7:
; CHECK-NOT: bb8:
; CHECK-NOT: bb11:
; CHECK-NOT: bb12:
; CHECK: bb:
; CHECK: bb2:
; CHECK: bb4:
; CHECK: bb10:
; CHECK: bb13:
declare void @ham()

define void @hoge() {
bb:
  %tmp = and i32 undef, 1073741823
  %tmp1 = icmp eq i32 %tmp, 2
  br i1 %tmp1, label %bb12, label %bb2

bb2:
  %tmp3 = icmp eq i32 %tmp, 3
  br i1 %tmp3, label %bb13, label %bb4

bb4:
  %tmp5 = icmp eq i32 %tmp, 5
  br i1 %tmp5, label %bb6, label %bb7

bb6:
  tail call void @ham()
  br label %bb7

bb7:
  br i1 %tmp3, label %bb13, label %bb8

bb8:
  %tmp9 = icmp eq i32 %tmp, 4
  br i1 %tmp9, label %bb13, label %bb10

bb10:
  br i1 %tmp9, label %bb11, label %bb13

bb11:
  br label %bb13

bb12:
  br label %bb2

bb13:
  ret void
}
