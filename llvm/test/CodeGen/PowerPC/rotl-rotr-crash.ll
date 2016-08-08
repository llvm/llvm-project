; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr8

; Ensure this does not crash

; Function Attrs: norecurse nounwind
define <4 x i32> @func1 (<4 x i32> %a) {
entry:
  %0 = lshr <4 x i32> %a, <i32 16, i32 16, i32 16, i32 16>
  %1 = shl <4 x i32> %a, <i32 16, i32 16, i32 16, i32 16>
  %2 = or <4 x i32> %1, %0
  ret <4 x i32> %2
}
