; RUN: opt < %s -disable-output -passes="print<block-freq>" 2>&1 | FileCheck %s
; Sanity check precision for small-ish min/max spread.

@g = global i32 0

; CHECK-LABEL: block-frequency-info: func0
; CHECK: - entry: float = 1.0, {{.*}}, count = 1000
; CHECK: - cmp0_true: float = 0.4, {{.*}}, count = 388
; CHECK: - cmp0_false: float = 0.6, {{.*}}, count = 600
; CHECK: - cmp1_true: float = 0.1, {{.*}}, count = 88
; CHECK: - cmp1_false: float = 0.3, {{.*}}, count = 288
; CHECK: - join: float = 1.0, {{.*}}, count = 1000

define void @func0(i32 %a0, i32 %a1) !prof !0 {
entry:
  %cmp0 = icmp ne i32 %a0, 0
  br i1 %cmp0, label %cmp0_true, label %cmp0_false, !prof !1

cmp0_true:
  store volatile i32 1, ptr @g
  %cmp1 = icmp ne i32 %a1, 0
  br i1 %cmp1, label %cmp1_true, label %cmp1_false, !prof !2

cmp0_false:
  store volatile i32 2, ptr @g
  br label %join

cmp1_true:
  store volatile i32 3, ptr @g
  br label %join

cmp1_false:
  store volatile i32 4, ptr @g
  br label %join

join:
  store volatile i32 5, ptr @g
  ret void
}

!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 400, i32 600}
!2 = !{!"branch_weights", i32 1, i32 3}
