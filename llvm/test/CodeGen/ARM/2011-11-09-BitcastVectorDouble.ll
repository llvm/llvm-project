; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s
; PR11319

@src1_v2i16 = global <2 x i16> <i16 0, i16 1>
@res_v2i16  = global <2 x i16> <i16 0, i16 0>

declare <2 x i16> @foo_v2i16(<2 x i16>) nounwind

define void @test_neon_call_return_v2i16() {
; CHECK-LABEL: test_neon_call_return_v2i16:
  %1 = load <2 x i16>, ptr @src1_v2i16
  %2 = call <2 x i16> @foo_v2i16(<2 x i16> %1) nounwind
  store <2 x i16> %2, ptr @res_v2i16
  ret void
}
