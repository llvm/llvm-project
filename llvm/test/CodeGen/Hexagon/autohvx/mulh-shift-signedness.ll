; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b,-packets < %s | FileCheck %s

; The shift of the most significant word must be logical for an unsigned
; product and arithmetic only when an operand is signed.

; CHECK-LABEL: mulhu_shr:
; CHECK: vlsr
; CHECK-NOT: vasr
define <32 x i32> @mulhu_shr(<32 x i32> %a0) #0 {
  %v0 = zext <32 x i32> %a0 to <32 x i64>
  %v1 = mul nuw <32 x i64> %v0, splat (i64 2863311531)
  %v2 = lshr <32 x i64> %v1, splat (i64 33)
  %v3 = trunc nuw nsw <32 x i64> %v2 to <32 x i32>
  ret <32 x i32> %v3
}

; CHECK-LABEL: mulhs_shr:
; CHECK: vasr
define <32 x i32> @mulhs_shr(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %v0 = sext <32 x i32> %a0 to <32 x i64>
  %v1 = sext <32 x i32> %a1 to <32 x i64>
  %v2 = mul nsw <32 x i64> %v0, %v1
  %v3 = ashr <32 x i64> %v2, splat (i64 33)
  %v4 = trunc <32 x i64> %v3 to <32 x i32>
  ret <32 x i32> %v4
}

; CHECK-LABEL: mulhsu_shr:
; CHECK: vasr
define <32 x i32> @mulhsu_shr(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %v0 = sext <32 x i32> %a0 to <32 x i64>
  %v1 = zext <32 x i32> %a1 to <32 x i64>
  %v2 = mul <32 x i64> %v0, %v1
  %v3 = ashr <32 x i64> %v2, splat (i64 33)
  %v4 = trunc <32 x i64> %v3 to <32 x i32>
  ret <32 x i32> %v4
}

attributes #0 = { nounwind "target-features"="+hvxv68,+hvx-length128b" }
