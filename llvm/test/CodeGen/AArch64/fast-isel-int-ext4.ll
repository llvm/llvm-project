; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define i32 @kill_flag(i16 signext %a) {
; CHECK-LABEL: kill_flag
entry:
  %0 = sext i16 %a to i32
  br label %bb1

bb1:
  %1 = icmp slt i32 undef, %0
  br i1 %1, label %loop, label %exit

loop:
  %2 = sext i16 %a to i32
  %3 = icmp slt i32 undef, %2
  br i1 %3, label %bb1, label %exit

exit:
  ret i32 0
}
