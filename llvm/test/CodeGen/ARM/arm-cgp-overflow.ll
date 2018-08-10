; RUN: llc -mtriple=thumbv8.main -mcpu=cortex-m33 %s -arm-disable-cgp=false -o - | FileCheck %s

; CHECK: overflow_add
; CHECK: add
; CHECK: uxth
; CHECK: cmp
define zeroext i16 @overflow_add(i16 zeroext %a, i16 zeroext %b) {
  %add = add i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}

; CHECK-LABEL: overflow_sub
; CHECK: sub
; CHECK: uxth
; CHECK: cmp
define zeroext i16 @overflow_sub(i16 zeroext %a, i16 zeroext %b) {
  %add = sub i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}

; CHECK-LABEL: overflow_mul
; CHECK: mul
; CHECK: uxth
; CHECK: cmp
define zeroext i16 @overflow_mul(i16 zeroext %a, i16 zeroext %b) {
  %add = mul i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}

; CHECK-LABEL: overflow_shl
; CHECK-COMMON: lsl
; CHECK-COMMON: uxth
; CHECK-COMMON: cmp
define zeroext i16 @overflow_shl(i16 zeroext %a, i16 zeroext %b) {
  %add = shl i16 %a, %b
  %or = or i16 %add, 1
  %cmp = icmp ugt i16 %or, 1024
  %res = select i1 %cmp, i16 2, i16 5
  ret i16 %res
}
