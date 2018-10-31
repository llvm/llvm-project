; Test that a rosbg conversion involving a sign extend operation rotates with
; the right number of steps.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -O0 | FileCheck %s

@g_136 = external global i16, align 2
@g_999 = external global i32, align 4

; Function Attrs: nounwind
define void @main() {
  %1 = load i32, i32* undef, align 4
  store i16 -28141, i16* @g_136, align 2
  %2 = load i32, i32* undef, align 4
  %3 = xor i32 -28141, %2
  %4 = xor i32 %1, %3
  %5 = sext i32 %4 to i64
  %6 = icmp sgt i64 0, %5
  %7 = zext i1 %6 to i32
  %8 = load i32, i32* @g_999, align 4
  %9 = or i32 %8, %7
; CHECK: rosbg   {{%r[0-9]+}}, {{%r[0-9]+}}, 63, 63, 33
  store i32 %9, i32* @g_999, align 4
  ret void
}
