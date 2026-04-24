; RUN: opt < %s -passes=reassociate -S | FileCheck %s

define i32 @add_chain_lifts_constant(i32 %x, i32 %y) {
; CHECK-LABEL: @add_chain_lifts_constant(
; CHECK:       [[SUM:%.*]] = add i32
; CHECK-NEXT:  [[RES:%.*]] = add i32 [[SUM]], 7
; CHECK-NEXT:  ret i32 [[RES]]
;
  %inner = add i32 %x, 7
  %res = add i32 %inner, %y
  ret i32 %res
}

define i32 @sub_chain_lifts_constant(i32 %x, i32 %y) {
; CHECK-LABEL: @sub_chain_lifts_constant(
; CHECK:       [[NEG:%.*]] = sub i32 0, %y
; CHECK:       [[SUM:%.*]] = add i32
; CHECK-NEXT:  [[RES:%.*]] = add i32 [[SUM]], 7
; CHECK-NEXT:  ret i32 [[RES]]
;
  %inner = add i32 %x, 7
  %res = sub i32 %inner, %y
  ret i32 %res
}
