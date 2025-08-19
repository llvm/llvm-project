; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; Fold: (select C, (x | a), x) | b  ->  x | select C, (a | b), b

define i8 @src(i8 %x, i8 %y) {
; CHECK-LABEL: @src(
; CHECK-NEXT:    [[V0:%.*]] = icmp eq i8 [[Y:%.*]], -1
; CHECK-NEXT:    [[V1:%.*]] = or i8 [[X:%.*]], 4
; CHECK-NEXT:    [[V2:%.*]] = select i1 [[V0]], i8 [[V1]], i8 [[X]]
; CHECK-NEXT:    [[V3:%.*]] = or i8 [[V2]], 1
; CHECK-NEXT:    ret i8 [[V3]]
;
  %v0 = icmp eq i8 %y, -1
  %v1 = or i8 %x, 4
  %v2 = select i1 %v0, i8 %v1, i8 %x
  %v3 = or i8 %v2, 1
  ret i8 %v3
}

define i8 @tgt(i8 %x, i8 %y) {
; CHECK-LABEL: @tgt(
; CHECK-NEXT:    [[V0:%.*]] = icmp eq i8 [[Y:%.*]], -1
; CHECK-NEXT:    [[V1:%.*]] = select i1 [[V0]], i8 5, i8 1
; CHECK-NEXT:    [[V2:%.*]] = or i8 [[X:%.*]], [[V1]]
; CHECK-NEXT:    ret i8 [[V2]]
;
  %v0 = icmp eq i8 %y, -1
  %v1 = select i1 %v0, i8 5, i8 1
  %v2 = or i8 %x, %v1
  ret i8 %v2
}


