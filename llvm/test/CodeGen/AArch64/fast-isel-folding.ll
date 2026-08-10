; RUN: llc -mtriple=aarch64-apple-darwin -O0 -fast-isel-abort=1 -verify-machineinstrs < %s

; Test that we don't fold the shift.
define i64 @fold_shift_test(i64 %a, i1 %c) {
  %1 = sub i64 %a, 8
  %2 = ashr i64 %1, 3
  br i1 %c, label %bb1, label %bb2
bb1:
  %3 = icmp ult i64 0, %2
  br i1 %3, label %bb2, label %bb3
bb2:
  ret i64 1
bb3:
  ret i64 2
}

; Test that we don't fold the sign-extend.
define i64 @fold_sext_test1(i32 %a, i1 %c) {
  %1 = sub i32 %a, 8
  %2 = sext i32 %1 to i64
  br i1 %c, label %bb1, label %bb2
bb1:
  %3 = icmp ult i64 0, %2
  br i1 %3, label %bb2, label %bb3
bb2:
  ret i64 1
bb3:
  ret i64 2
}

; Test that we don't fold the sign-extend.
define i64 @fold_sext_test2(i32 %a, i1 %c) {
  %1 = sub i32 %a, 8
  %2 = sext i32 %1 to i64
  br i1 %c, label %bb1, label %bb2
bb1:
  %3 = shl i64 %2, 4
  ret i64 %3
bb2:
  ret i64 %2
}

; Test that we clear the kill flag.
define i32 @fold_kill_test(i32 %a) {
  %1 = sub i32 %a, 8
  %2 = shl i32 %1, 3
  %3 = icmp ult i32 0, %2
  br i1 %3, label %bb1, label %bb2
bb1:
  ret i32 %2
bb2:
  %4 = add i32 %2, 4
  ret i32 %4
}
