; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: danny:
; CHECK: r{{[0-9]+}} = mpy(r0,r1)
define i32 @danny(i32 %a0, i32 %a1) {
b2:
  %v3 = sext i32 %a0 to i64
  %v4 = sext i32 %a1 to i64
  %v5 = mul nsw i64 %v3, %v4
  %v6 = ashr i64 %v5, 32
  %v7 = trunc i64 %v6 to i32
  ret i32 %v7
}

; CHECK-LABEL: sammy:
; CHECK: r{{[0-9]+}} = mpy(r0,r1)
define i32 @sammy(i32 %a0, i32 %a1) {
b2:
  %v3 = sext i32 %a0 to i64
  %v4 = sext i32 %a1 to i64
  %v5 = mul nsw i64 %v3, %v4
  %v6 = lshr i64 %v5, 32
  %v7 = trunc i64 %v6 to i32
  ret i32 %v7
}
