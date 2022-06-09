; Test memcmp with 0 size.

; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; REQUIRES: asserts

declare i32 @memcmp(i8* nocapture, i8* nocapture, i64)

define hidden void @fun() {
; CHECK-LABEL: fun
entry:
  %len = extractvalue [2 x i64] zeroinitializer, 1
  br i1 undef, label %end, label %call

call:
  %res = tail call signext i32 @memcmp(i8* noundef undef, i8* noundef undef, i64 noundef %len)
  unreachable

end:
  unreachable
}
