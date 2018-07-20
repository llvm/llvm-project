; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; CHECK-NOT: sxth

target triple = "hexagon"

@g0 = common global i32 0, align 4

define i32 @f0(i32 %a0, i32 %a1) {
b0:
  %v0 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %a0, i32 %a1)
  %v1 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %a1, i32 %a0)
  %v2 = shl i32 %v0, 16
  %v3 = ashr exact i32 %v2, 16
  %v4 = shl i32 %v1, 16
  %v5 = ashr exact i32 %v4, 16
  %v6 = tail call i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32 %v0, i32 %v1)
  %v7 = shl i32 %v6, 16
  %v8 = ashr exact i32 %v7, 16
  %v9 = load i32, i32* @g0, align 4
  %v10 = icmp ne i32 %v9, %v6
  %v11 = zext i1 %v10 to i32
  ret i32 %v11
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.A2.addh.l16.sat.ll(i32, i32) #0

attributes #0 = { nounwind readnone }
