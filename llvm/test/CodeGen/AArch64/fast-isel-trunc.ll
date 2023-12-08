; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s

; Test that %1 doesn't get the kill flag set before its last use.
define i32 @test_trunc(i32 %a) {
  %1 = add i32 %a, 1
  %2 = trunc i32 %1 to i16
  %3 = icmp ult i16 1, %2
  %4 = add i32 %1, 1
  %5 = sext i1 %3 to i32
  %6 = and i32 %4, %5
  ret i32 %6
}
