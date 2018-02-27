; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

; CHECK-LABEL: mul_1
; CHECK: r1:0 = mpy(r2,r0)
define i64 @mul_1(i64 %a0, i64 %a1) #0 {
b2:
  %v3 = shl i64 %a0, 32
  %v4 = ashr exact i64 %v3, 32
  %v5 = shl i64 %a1, 32
  %v6 = ashr exact i64 %v5, 32
  %v7 = mul nsw i64 %v6, %v4
  ret i64 %v7
}

; CHECK-LABEL: mul_2
; CHECK: r0 = memb(r0+#0)
; CHECK: r1:0 = mpy(r2,r0)
; CHECK: jumpr r31
define i64 @mul_2(i8* %a0, i64 %a1) #0 {
b2:
  %v3 = load i8, i8* %a0
  %v4 = sext i8 %v3 to i64
  %v5 = shl i64 %a1, 32
  %v6 = ashr exact i64 %v5, 32
  %v7 = mul nsw i64 %v6, %v4
  ret i64 %v7
}

; CHECK-LABEL: mul_3
; CHECK: r[[REG30:[0-9]+]] = sxth(r2)
; CHECK: r1:0 = mpy(r[[REG30]],r0)
; CHECK: jumpr r31
define i64 @mul_3(i64 %a0, i64 %a1) #0 {
b2:
  %v3 = shl i64 %a0, 32
  %v4 = ashr exact i64 %v3, 32
  %v5 = shl i64 %a1, 48
  %v6 = ashr exact i64 %v5, 48
  %v7 = mul nsw i64 %v6, %v4
  ret i64 %v7
}

; CHECK-LABEL: mul_4
; CHECK: r[[REG40:[0-9]+]] = asrh(r2)
; CHECK: r1:0 = mpy(r1,r[[REG40]])
; CHECK: jumpr r31
define i64 @mul_4(i64 %a0, i64 %a1) #0 {
b2:
  %v3 = ashr i64 %a0, 32
  %v4 = trunc i64 %a1 to i32
  %v5 = ashr i32 %v4, 16
  %v6 = sext i32 %v5 to i64
  %v7 = mul nsw i64 %v3, %v6
  ret i64 %v7
}

; CHECK-LABEL: mul_acc_1
; CHECK: r5:4 += mpy(r2,r0)
; CHECK: r1:0 = combine(r5,r4)
; CHECK: jumpr r31
define i64 @mul_acc_1(i64 %a0, i64 %a1, i64 %a2) #0 {
b3:
  %v4 = shl i64 %a0, 32
  %v5 = ashr exact i64 %v4, 32
  %v6 = shl i64 %a1, 32
  %v7 = ashr exact i64 %v6, 32
  %v8 = mul nsw i64 %v7, %v5
  %v9 = add i64 %a2, %v8
  ret i64 %v9
}

; CHECK-LABEL: mul_acc_2
; CHECK: r2 = memw(r2+#0)
; CHECK: r5:4 += mpy(r2,r0)
; CHECK: r1:0 = combine(r5,r4)
; CHECK: jumpr r31
define i64 @mul_acc_2(i64 %a0, i32* %a1, i64 %a2) #0 {
b3:
  %v4 = shl i64 %a0, 32
  %v5 = ashr exact i64 %v4, 32
  %v6 = load i32, i32* %a1
  %v7 = sext i32 %v6 to i64
  %v8 = mul nsw i64 %v7, %v5
  %v9 = add i64 %a2, %v8
  ret i64 %v9
}

; CHECK-LABEL: mul_nac_1
; CHECK: r5:4 -= mpy(r2,r0)
; CHECK: r1:0 = combine(r5,r4)
; CHECK: jumpr r31
define i64 @mul_nac_1(i64 %a0, i64 %a1, i64 %a2) #0 {
b3:
  %v4 = shl i64 %a0, 32
  %v5 = ashr exact i64 %v4, 32
  %v6 = shl i64 %a1, 32
  %v7 = ashr exact i64 %v6, 32
  %v8 = mul nsw i64 %v7, %v5
  %v9 = sub i64 %a2, %v8
  ret i64 %v9
}

; CHECK-LABEL: mul_nac_2
; CHECK: r1:0 = combine(r5,r4)
; CHECK: r6 = memw(r0+#0)
; CHECK: r1:0 -= mpy(r2,r6)
; CHECK: jumpr r31
define i64 @mul_nac_2(i32* %a0, i64 %a1, i64 %a2) #0 {
b3:
  %v4 = load i32, i32* %a0
  %v5 = sext i32 %v4 to i64
  %v6 = shl i64 %a1, 32
  %v7 = ashr exact i64 %v6, 32
  %v8 = mul nsw i64 %v7, %v5
  %v9 = sub i64 %a2, %v8
  ret i64 %v9
}

attributes #0 = { nounwind }
