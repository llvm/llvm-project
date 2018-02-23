; RUN: llc -O0 -mtriple=aarch64-apple-ios -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

; Check that we don't invalidate the vreg map.
; This test is brittle: the invalidation only triggers when we grow the map.

; CHECK-LABEL: name: test_bitcast_invalid_vreg
define i32 @test_bitcast_invalid_vreg() {
  %tmp0 = add i32 1, 2
  %tmp1 = add i32 3, 4
  %tmp2 = add i32 5, 6
  %tmp3 = add i32 7, 8
  %tmp4 = add i32 9, 10
  %tmp5 = add i32 11, 12
  %tmp6 = add i32 13, 14
  %tmp7 = add i32 15, 16
  %tmp8 = add i32 17, 18
  %tmp9 = add i32 19, 20
  %tmp10 = add i32 21, 22
  %tmp11 = add i32 23, 24
  %tmp12 = add i32 25, 26
  %tmp13 = add i32 27, 28
  %tmp14 = add i32 29, 30
  %tmp15 = add i32 30, 30

; At this point we mapped 46 values. The 'i32 100' constant will grow the map.
; CHECK:  %46(s32) = G_CONSTANT i32 100
; CHECK:  %w0 = COPY %46(s32)
  %res = bitcast i32 100 to i32
  ret i32 %res
}
