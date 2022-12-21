; RUN: llc < %s -mtriple=armv7-apple-darwin | FileCheck %s

;CHECK: _foo
;CHECK-NOT: vld1.32
;CHECK-NOT: vst1.32
;CHECK: bx
define void @foo(ptr %J) {
  %A = load <16 x i8>, ptr %J
  %T1 = shufflevector <16 x i8> %A, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %T2 = shufflevector <8 x i8>  %T1, <8 x i8> undef, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  store <16 x i8> %T2, ptr %J
  ret void
}
