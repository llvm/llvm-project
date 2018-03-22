; RUN: opt < %s -S -slp-vectorizer -instcombine -pass-remarks-output=%t | FileCheck %s
; RUN: cat %t | FileCheck -check-prefix=REMARK %s
; RUN: opt < %s -S -passes='slp-vectorizer,instcombine' -pass-remarks-output=%t | FileCheck %s
; RUN: cat %t | FileCheck -check-prefix=REMARK %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL:  @gather_multiple_use(
; CHECK-NEXT:     [[TMP00:%.*]] = lshr i32 [[A:%.*]], 15
; CHECK-NEXT:     [[TMP01:%.*]] = and i32 [[TMP00]], 65537
; CHECK-NEXT:     [[TMP02:%.*]] = mul nuw i32 [[TMP01]], 65535
; CHECK-NEXT:     [[TMP03:%.*]] = add i32 [[TMP02]], [[A]]
; CHECK-NEXT:     [[TMP04:%.*]] = xor i32 [[TMP03]], [[TMP02]]
; CHECK-NEXT:     [[TMP05:%.*]] = lshr i32 [[C:%.*]], 15
; CHECK-NEXT:     [[TMP06:%.*]] = and i32 [[TMP05]], 65537
; CHECK-NEXT:     [[TMP07:%.*]] = mul nuw i32 [[TMP06]], 65535
; CHECK-NEXT:     [[TMP08:%.*]] = add i32 [[TMP07]], [[C]]
; CHECK-NEXT:     [[TMP09:%.*]] = xor i32 [[TMP08]], [[TMP07]]
; CHECK-NEXT:     [[TMP10:%.*]] = lshr i32 [[B:%.*]], 15
; CHECK-NEXT:     [[TMP11:%.*]] = and i32 [[TMP10]], 65537
; CHECK-NEXT:     [[TMP12:%.*]] = mul nuw i32 [[TMP11]], 65535
; CHECK-NEXT:     [[TMP13:%.*]] = add i32 [[TMP12]], [[B]]
; CHECK-NEXT:     [[TMP14:%.*]] = xor i32 [[TMP13]], [[TMP12]]
; CHECK-NEXT:     [[TMP15:%.*]] = lshr i32 [[D:%.*]], 15
; CHECK-NEXT:     [[TMP16:%.*]] = and i32 [[TMP15]], 65537
; CHECK-NEXT:     [[TMP17:%.*]] = mul nuw i32 [[TMP16]], 65535
; CHECK-NEXT:     [[TMP18:%.*]] = add i32 [[TMP17]], [[D]]
; CHECK-NEXT:     [[TMP19:%.*]] = xor i32 [[TMP18]], [[TMP17]]
; CHECK-NEXT:     [[TMP20:%.*]] = add i32 [[TMP09]], [[TMP04]]
; CHECK-NEXT:     [[TMP21:%.*]] = add i32 [[TMP20]], [[TMP14]]
; CHECK-NEXT:     [[TMP22:%.*]] = add i32 [[TMP21]], [[TMP19]]
; CHECK-NEXT:     ret i32 [[TMP22]]
;
; REMARK-LABEL: Function: gather_multiple_use
; REMARK:       Args:
; REMARK-NEXT:    - String: Vectorizing horizontal reduction is possible
; REMARK-NEXT:    - String: 'but not beneficial with cost '
; REMARK-NEXT:    - Cost: '2'
;
define internal i32 @gather_multiple_use(i32 %a, i32 %b, i32 %c, i32 %d) {
  %tmp00 = lshr i32 %a, 15
  %tmp01 = and i32 %tmp00, 65537
  %tmp02 = mul nuw i32 %tmp01, 65535
  %tmp03 = add i32 %tmp02, %a
  %tmp04 = xor i32 %tmp03, %tmp02
  %tmp05 = lshr i32 %c, 15
  %tmp06 = and i32 %tmp05, 65537
  %tmp07 = mul nuw i32 %tmp06, 65535
  %tmp08 = add i32 %tmp07, %c
  %tmp09 = xor i32 %tmp08, %tmp07
  %tmp10 = lshr i32 %b, 15
  %tmp11 = and i32 %tmp10, 65537
  %tmp12 = mul nuw i32 %tmp11, 65535
  %tmp13 = add i32 %tmp12, %b
  %tmp14 = xor i32 %tmp13, %tmp12
  %tmp15 = lshr i32 %d, 15
  %tmp16 = and i32 %tmp15, 65537
  %tmp17 = mul nuw i32 %tmp16, 65535
  %tmp18 = add i32 %tmp17, %d
  %tmp19 = xor i32 %tmp18, %tmp17
  %tmp20 = add i32 %tmp09, %tmp04
  %tmp21 = add i32 %tmp20, %tmp14
  %tmp22 = add i32 %tmp21, %tmp19
  ret i32 %tmp22
}
