; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: A2_combine_ll:
; CHECK: combine(r1.l,r0.l)
define i32 @A2_combine_ll(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = and i32 %a0, 65535
  %v4 = shl i32 %a1, 16
  %v5 = or i32 %v3, %v4
  ret i32 %v5
}

; CHECK-LABEL: A2_combine_lh:
; CHECK: combine(r1.l,r0.h)
define i32 @A2_combine_lh(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = lshr i32 %a0, 16
  %v4 = shl i32 %a1, 16
  %v5 = or i32 %v4, %v3
  ret i32 %v5
}

; CHECK-LABEL: A2_combine_hl:
; CHECK: combine(r1.h,r0.l)
define i32 @A2_combine_hl(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = and i32 %a0, 65535
  %v4 = and i32 %a1, 268431360
  %v5 = or i32 %v3, %v4
  ret i32 %v5
}

; CHECK-LABEL: A2_combine_hh:
; CHECK: combine(r1.h,r0.h)
define i32 @A2_combine_hh(i32 %a0, i32 %a1) #0 {
b2:
  %v3 = lshr i32 %a0, 16
  %v4 = and i32 %a1, 268431360
  %v5 = or i32 %v3, %v4
  ret i32 %v5
}

attributes #0 = { noinline nounwind optnone readnone }
