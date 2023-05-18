; RUN: llc -O0 -mtriple=aarch64-apple-ios -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

; Test that extends correctly translate to G_[ZS]EXT. The translator will never
; emit a G_SEXT_INREG.

define i32 @test_zext(i32 %a) {
  ; CHECK-LABEL: name: test_zext
  ; CHECK: %0:_(s32) = COPY $w0
  ; CHECK: %1:_(s8) = G_TRUNC %0(s32)
  ; CHECK: %2:_(s16) = G_ZEXT %1(s8)
  ; CHECK: %3:_(s32) = G_ZEXT %2(s16)
  ; CHECK: $w0 = COPY %3(s32)
  %tmp0 = trunc i32 %a to i8
  %tmp1 = zext i8 %tmp0 to i16
  %tmp2 = zext i16 %tmp1 to i32
  ret i32 %tmp2
}

define i32 @test_sext(i32 %a) {
  ; CHECK-LABEL: name: test_sext
  ; CHECK: %0:_(s32) = COPY $w0
  ; CHECK: %1:_(s8) = G_TRUNC %0(s32)
  ; CHECK: %2:_(s16) = G_SEXT %1(s8)
  ; CHECK: %3:_(s32) = G_SEXT %2(s16)
  ; CHECK: $w0 = COPY %3(s32)
  %tmp0 = trunc i32 %a to i8
  %tmp1 = sext i8 %tmp0 to i16
  %tmp2 = sext i16 %tmp1 to i32
  ret i32 %tmp2
}
