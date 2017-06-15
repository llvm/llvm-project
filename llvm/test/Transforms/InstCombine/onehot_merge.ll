; RUN: opt < %s -instcombine -S | FileCheck %s

;CHECK: @and_consts
;CHECK: and i32 %k, 12
;CHECK: icmp ne i32 %0, 12
;CHECK: ret
define i1 @and_consts(i32 %k, i32 %c1, i32 %c2) {
bb:
  %tmp1 = and i32 4, %k
  %tmp2 = icmp eq i32 %tmp1, 0
  %tmp5 = and i32 8, %k
  %tmp6 = icmp eq i32 %tmp5, 0
  %or = or i1 %tmp2, %tmp6
  ret i1 %or
}

;CHECK: @foo1_and
;CHECK:  shl i32 1, %c1
;CHECK-NEXT:  lshr i32 -2147483648, %c2
;CHECK-NEXT:  or i32
;CHECK-NEXT:  and i32
;CHECK-NEXT:  icmp ne i32 %1, %0
;CHECK: ret
define i1 @foo1_and(i32 %k, i32 %c1, i32 %c2) {
bb:
  %tmp = shl i32 1, %c1
  %tmp4 = lshr i32 -2147483648, %c2
  %tmp1 = and i32 %tmp, %k
  %tmp2 = icmp eq i32 %tmp1, 0
  %tmp5 = and i32 %tmp4, %k
  %tmp6 = icmp eq i32 %tmp5, 0
  %or = or i1 %tmp2, %tmp6
  ret i1 %or
}

; Same as above but with operands commuted one of the ands, but not the other.
; TODO handle this form correctly
define i1 @foo1_and_commuted(i32 %k, i32 %c1, i32 %c2) {
; CHECK-LABEL: @foo1_and_commuted(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[K2:%.*]] = mul i32 [[K:%.*]], [[K]]
; CHECK-NEXT:    [[TMP:%.*]] = shl i32 1, [[C1:%.*]]
; CHECK-NEXT:    [[TMP4:%.*]] = lshr i32 -2147483648, [[C2:%.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = and i32 [[K2]], [[TMP]]
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 0
; CHECK-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], [[K2]]
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i32 [[TMP5]], 0
; CHECK-NEXT:    [[OR:%.*]] = or i1 [[TMP2]], [[TMP6]]
; CHECK-NEXT:    ret i1 [[OR]]
;
bb:
  %k2 = mul i32 %k, %k ; to trick the complexity sorting
  %tmp = shl i32 1, %c1
  %tmp4 = lshr i32 -2147483648, %c2
  %tmp1 = and i32 %k2, %tmp
  %tmp2 = icmp eq i32 %tmp1, 0
  %tmp5 = and i32 %tmp4, %k2
  %tmp6 = icmp eq i32 %tmp5, 0
  %or = or i1 %tmp2, %tmp6
  ret i1 %or
}

