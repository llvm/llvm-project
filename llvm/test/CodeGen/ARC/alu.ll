; RUN: llc -march=arc < %s | FileCheck %s

; CHECK-LABEL: add_r
; CHECK: add %r0, %r{{[01]}}, %r{{[01]}}
define i32 @add_r(i32 %a, i32 %b) nounwind {
entry:
  %v = add i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: add_u6
; CHECK: add %r0, %r0, 15
define i32 @add_u6(i32 %a) nounwind {
  %v = add i32 %a, 15
  ret i32 %v
}

; CHECK-LABEL: add_limm
; CHECK: add %r0, %r0, 12345
define i32 @add_limm(i32 %a) nounwind {
  %v = add i32 %a, 12345
  ret i32 %v
}

; CHECK-LABEL: mpy_r
; CHECK: mpy %r0, %r{{[01]}}, %r{{[01]}}
define i32 @mpy_r(i32 %a, i32 %b) nounwind {
entry:
  %v = mul i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: mpy_u6
; CHECK: mpy %r0, %r0, 10
define i32 @mpy_u6(i32 %a) nounwind {
  %v = mul i32 %a, 10
  ret i32 %v
}

; CHECK-LABEL: mpy_limm
; CHECK: mpy %r0, %r0, 12345
define i32 @mpy_limm(i32 %a) nounwind {
  %v = mul i32 %a, 12345
  ret i32 %v
}

; CHECK-LABEL: max_r
; CHECK: max %r0, %r{{[01]}}, %r{{[01]}}
define i32 @max_r(i32 %a, i32 %b) nounwind {
  %i = icmp sgt i32 %a, %b
  %v = select i1 %i, i32 %a, i32 %b
  ret i32 %v
}

; CHECK-LABEL: max_u6
; CHECK: max %r0, %r0, 12
define i32 @max_u6(i32 %a) nounwind {
  %i = icmp sgt i32 %a, 12
  %v = select i1 %i, i32 %a, i32 12
  ret i32 %v
}

; CHECK-LABEL: max_limm
; CHECK: max %r0, %r0, 2345
define i32 @max_limm(i32 %a) nounwind {
  %i = icmp sgt i32 %a, 2345
  %v = select i1 %i, i32 %a, i32 2345
  ret i32 %v
}

; CHECK-LABEL: min_r
; CHECK: min %r0, %r{{[01]}}, %r{{[01]}}
define i32 @min_r(i32 %a, i32 %b) nounwind {
  %i = icmp slt i32 %a, %b
  %v = select i1 %i, i32 %a, i32 %b
  ret i32 %v
}

; CHECK-LABEL: min_u6
; CHECK: min %r0, %r0, 20
define i32 @min_u6(i32 %a) nounwind {
  %i = icmp slt i32 %a, 20
  %v = select i1 %i, i32 %a, i32 20
  ret i32 %v
}

; CHECK-LABEL: min_limm
; CHECK: min %r0, %r0, 2040
define i32 @min_limm(i32 %a) nounwind {
  %i = icmp slt i32 %a, 2040
  %v = select i1 %i, i32 %a, i32 2040
  ret i32 %v
}

; CHECK-LABEL: and_r
; CHECK: and %r0, %r{{[01]}}, %r{{[01]}}
define i32 @and_r(i32 %a, i32 %b) nounwind {
  %v = and i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: and_u6
; CHECK: and %r0, %r0, 7
define i32 @and_u6(i32 %a) nounwind {
  %v = and i32 %a, 7
  ret i32 %v
}

; 0xfffff == 1048575
; CHECK-LABEL: and_limm
; CHECK: and %r0, %r0, 1048575 
define i32 @and_limm(i32 %a) nounwind {
  %v = and i32 %a, 1048575
  ret i32 %v
}

; CHECK-LABEL: or_r
; CHECK: or %r0, %r{{[01]}}, %r{{[01]}}
define i32 @or_r(i32 %a, i32 %b) nounwind {
  %v = or i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: or_u6
; CHECK: or %r0, %r0, 7
define i32 @or_u6(i32 %a) nounwind {
  %v = or i32 %a, 7
  ret i32 %v
}

; 0xf0f0f == 986895
; CHECK-LABEL: or_limm
define i32 @or_limm(i32 %a) nounwind {
  %v = or i32 %a, 986895
  ret i32 %v
}

; CHECK-LABEL: xor_r
; CHECK: xor %r0, %r{{[01]}}, %r{{[01]}}
define i32 @xor_r(i32 %a, i32 %b) nounwind {
  %v = xor i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: xor_u6
; CHECK: xor %r0, %r0, 3
define i32 @xor_u6(i32 %a) nounwind {
  %v = xor i32 %a, 3
  ret i32 %v
}

; CHECK-LABEL: xor_limm
; CHECK: xor %r0, %r0, 986895
define i32 @xor_limm(i32 %a) nounwind {
  %v = xor i32 %a, 986895
  ret i32 %v
}

; CHECK-LABEL: asl_r
; CHECK: asl %r0, %r{{[01]}}, %r{{[01]}}
define i32 @asl_r(i32 %a, i32 %b) nounwind {
  %v = shl i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: asl_u6
; CHECK: asl %r0, %r0, 4
define i32 @asl_u6(i32 %a) nounwind {
  %v = shl i32 %a, 4
  ret i32 %v
}

; CHECK-LABEL: lsr_r
; CHECK: lsr %r0, %r{{[01]}}, %r{{[01]}}
define i32 @lsr_r(i32 %a, i32 %b) nounwind {
  %v = lshr i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: lsr_u6
; CHECK: lsr %r0, %r0, 6
define i32 @lsr_u6(i32 %a) nounwind {
  %v = lshr i32 %a, 6
  ret i32 %v
}

; CHECK-LABEL: asr_r
; CHECK: asr %r0, %r{{[01]}}, %r{{[01]}}
define i32 @asr_r(i32 %a, i32 %b) nounwind {
  %v = ashr i32 %a, %b
  ret i32 %v
}

; CHECK-LABEL: asr_u6
; CHECK: asr %r0, %r0, 8
define i32 @asr_u6(i32 %a) nounwind {
  %v = ashr i32 %a, 8
  ret i32 %v
}

; CHECK-LABEL: ror_r
; CHECK: ror %r0, %r{{[01]}}, %r{{[01]}}
define i32 @ror_r(i32 %a, i32 %b) nounwind {
  %v1 = lshr i32 %a, %b
  %ls = sub i32 32, %b
  %v2 = shl i32 %a, %ls
  %v = or i32 %v1, %v2
  ret i32 %v
}

; CHECK-LABEL: ror_u6
; CHECK: ror %r0, %r0, 10
define i32 @ror_u6(i32 %a) nounwind {
  %v1 = lshr i32 %a, 10
  %v2 = shl i32 %a, 22
  %v = or i32 %v1, %v2
  ret i32 %v
}

; CHECK-LABEL: sexh_r
; CHECK: sexh %r0, %r0
define i32 @sexh_r(i32 %a) nounwind {
  %v1 = shl i32 %a, 16
  %v = ashr i32 %v1, 16
  ret i32 %v
}

; CHECK-LABEL: sexb_r
; CHECK: sexb %r0, %r0
define i32 @sexb_r(i32 %a) nounwind {
  %v1 = shl i32 %a, 24
  %v = ashr i32 %v1, 24
  ret i32 %v
}

; CHECK-LABEL: mulu64
; CHECK-DAG: mpy %r[[REG:[0-9]+]], %r{{[01]}}, %r{{[01]}}
; CHECK-DAG: mpymu %r[[REG:[0-9]+]], %r{{[01]}}, %r{{[01]}}
define i64 @mulu64(i32 %a, i32 %b) nounwind {
  %a64 = zext i32 %a to i64
  %b64 = zext i32 %b to i64
  %v = mul i64 %a64, %b64
  ret i64 %v
}

; CHECK-LABEL: muls64
; CHECK-DAG: mpy %r[[REG:[0-9]+]], %r{{[01]}}, %r{{[01]}}
; CHECK-DAG: mpym %r[[REG:[0-9]+]], %r{{[01]}}, %r{{[01]}}
define i64 @muls64(i32 %a, i32 %b) nounwind {
  %a64 = sext i32 %a to i64
  %b64 = sext i32 %b to i64
  %v = mul i64 %a64, %b64
  ret i64 %v
}

