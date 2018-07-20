; Test memcmp using CLC.  In this test case the CLC loop will do all the work
; and the DoneMBB becomes empty. It will not pass the mischeduling verifiers
; if DoneMBB does not have CC in its live-in list.

; RUN: llc < %s -mtriple=s390x-linux-gnu -misched=shuffle | FileCheck %s
; REQUIRES: asserts

declare i32 @memcmp(i8* nocapture, i8* nocapture, i64)

define i32 @fun() {
; CHECK-LABEL: fun
  %call = call signext i32 @memcmp(i8* nonnull undef, i8* nonnull undef, i64 2048)
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %labT, label %labF

labT:
  ret i32 0

labF:
  ret i32 1
}
