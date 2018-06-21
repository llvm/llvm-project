; RUN: llc < %s -mtriple=x86_64-linux-pc -mcpu=atom | FileCheck %s

define <16 x i8> @foo(<16 x i8> %in) {
  %r = shufflevector <16 x i8> %in, <16 x i8> undef, <16 x i32> < i32 7, i32 3, i32 2, i32 11, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  ret <16 x i8> %r
; CHECK: foo
; CHECK: pshufb
; CHECK-NEXT: ret
}
