; RUN: opt -jump-threading -S < %s | FileCheck %s
; RUN: opt -passes=jump-threading -S < %s | FileCheck %s

; Check that all zero branch weights do not cause a crash.
define void @zero_branch_weights(i32 %tmp, i32 %tmp3) {
bb:
  %tmp1 = icmp eq i32 %tmp, 1
  br i1 %tmp1, label %bb5, label %bb2
; CHECK-NOT: br i1 %tmp1,{{.*}} !prof

bb2:
  %tmp4 = icmp ne i32 %tmp3, 1
  br label %bb5
; CHECK: br i1 %tmp4, {{.*}} !prof ![[PROF:[0-9]+]]

bb5:
  %tmp6 = phi i1 [ false, %bb ], [ %tmp4, %bb2 ]
  br i1 %tmp6, label %bb8, label %bb7, !prof !{!"branch_weights", i32 0, i32 0}

bb7:
  br label %bb9

bb8:
  br label %bb9

bb9:
  ret void
}

;CHECK: ![[PROF]] = !{!"branch_weights", i32 -2147483648, i32 0}
