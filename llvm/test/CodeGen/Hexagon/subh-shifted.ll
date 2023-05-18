; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = sub(r{{[0-9]+}}.{{L|l}},r{{[0-9]+}}.{{L|l}}):<<16

; Function Attrs: nounwind readnone
define i64 @f0(i64 %a0, i16 zeroext %a1, i16 zeroext %a2) #0 {
b0:
  %v0 = zext i16 %a1 to i32
  %v1 = zext i16 %a2 to i32
  %v2 = sub nsw i32 %v0, %v1
  %v3 = shl i32 %v2, 16
  %v4 = icmp slt i32 %v3, 65536
  %v5 = ashr exact i32 %v3, 16
  %v6 = select i1 %v4, i32 1, i32 %v5
  %v7 = icmp sgt i32 %v6, 4
  %v8 = add i32 %v6, 65535
  %v9 = shl i64 %a0, 2
  %v10 = and i32 %v8, 65535
  %v11 = zext i32 %v10 to i64
  %v12 = select i1 %v7, i64 3, i64 %v11
  %v13 = or i64 %v12, %v9
  ret i64 %v13
}

attributes #0 = { nounwind readnone }
