; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

; With sub reassociation, constant folding can eliminate the 12 and -12 constants.
define i32 @test1(i32 %A, i32 %B) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[Z:%.*]] = sub i32 %A, %B
; CHECK-NEXT:    ret i32 [[Z]]
;
  %X = add i32 -12, %A
  %Y = sub i32 %X, %B
  %Z = add i32 %Y, 12
  ret i32 %Z
}

; PR2047
; With sub reassociation, constant folding can eliminate the uses of %a.
define i32 @test2(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[SUM:%.*]] = add i32 %c, %b
; CHECK-NEXT:    [[TMP7:%.*]] = sub i32 0, [[SUM]]
; CHECK-NEXT:    ret i32 [[TMP7]]
;
  %tmp3 = sub i32 %a, %b
  %tmp5 = sub i32 %tmp3, %c
  %tmp7 = sub i32 %tmp5, %a
  ret i32 %tmp7
}

