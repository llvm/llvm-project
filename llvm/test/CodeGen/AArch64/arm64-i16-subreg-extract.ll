; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define i32 @foo(ptr %__a) nounwind {
; CHECK-LABEL: foo:
; CHECK: umov.h w{{[0-9]+}}, v{{[0-9]+}}[0]
  %tmp18 = load <4 x i16>, ptr %__a, align 8
  %vget_lane = extractelement <4 x i16> %tmp18, i32 0
  %conv = zext i16 %vget_lane to i32
  %mul = mul nsw i32 3, %conv
  ret i32 %mul
}

