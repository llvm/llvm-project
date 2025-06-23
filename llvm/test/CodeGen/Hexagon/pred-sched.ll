; RUN: llc -O3 -mtriple=hexagon < %s | FileCheck %s

; We want to see a .new instruction in this sequence.
; CHECK: p[[PRED:[0-3]]] = tstbit
; CHECK: if (p[[PRED]].new)

target triple = "hexagon"

; Function Attrs: nounwind readnone
define zeroext i16 @f0(i8 zeroext %a0, i16 zeroext %a1) #0 {
b0:
  %v0 = zext i8 %a0 to i32
  %v1 = zext i16 %a1 to i32
  %v2 = xor i32 %v0, %v1
  %v3 = and i32 %v2, 1
  %v4 = lshr i8 %a0, 1
  %v5 = icmp eq i32 %v3, 0
  %v6 = lshr i16 %a1, 1
  %v7 = xor i16 %v6, -24575
  %v8 = select i1 %v5, i16 %v6, i16 %v7
  %v9 = zext i8 %v4 to i32
  %v10 = zext i16 %v8 to i32
  %v11 = xor i32 %v9, %v10
  %v12 = and i32 %v11, 1
  %v13 = lshr i8 %a0, 2
  %v14 = icmp eq i32 %v12, 0
  %v15 = lshr i16 %v8, 1
  %v16 = xor i16 %v15, -24575
  %v17 = select i1 %v14, i16 %v15, i16 %v16
  %v18 = zext i8 %v13 to i32
  %v19 = zext i16 %v17 to i32
  %v20 = xor i32 %v18, %v19
  %v21 = and i32 %v20, 1
  %v22 = icmp eq i32 %v21, 0
  %v23 = lshr i16 %v17, 1
  %v24 = xor i16 %v23, -24575
  %v25 = select i1 %v22, i16 %v23, i16 %v24
  ret i16 %v25
}

attributes #0 = { nounwind readnone }
