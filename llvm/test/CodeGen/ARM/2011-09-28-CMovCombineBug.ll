; RUN: llc -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 < %s

; rdar://10196296
; ARM target specific dag combine created a cycle in DAG.

define void @t() nounwind ssp {
  %1 = load i64* undef, align 4
  %2 = shl i32 5, 0
  %3 = zext i32 %2 to i64
  %4 = and i64 %1, %3
  %5 = lshr i64 %4, undef
  switch i64 %5, label %8 [
    i64 0, label %9
    i64 1, label %6
    i64 4, label %9
    i64 5, label %7
  ]

; <label>:6                                       ; preds = %0
  unreachable

; <label>:7                                       ; preds = %0
  unreachable

; <label>:8                                       ; preds = %0
  unreachable

; <label>:9                                       ; preds = %0, %0
  ret void
}
