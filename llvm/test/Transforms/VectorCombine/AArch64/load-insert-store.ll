; RUN: opt -S -passes=vector-combine -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s
; RUN: opt -S -passes=vector-combine -mtriple=aarch64_be-unknown-linux-gnu < %s | FileCheck %s

; Use real little- and big-endian targets to verify that an insertelement lane
; is scalarized to the same memory element on both byte orders.
define void @insert_store2(ptr %p, i16 %x, i16 %y) {
; CHECK-LABEL: @insert_store2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP0:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P:%.*]], i32 0, i32 6
; CHECK-NEXT:    store i16 [[X:%.*]], ptr [[GEP0]], align 1
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P]], i32 0, i32 7
; CHECK-NEXT:    store i16 [[Y:%.*]], ptr [[GEP1]], align 1
; CHECK-NEXT:    ret void
;
entry:
  %load = load <8 x i16>, ptr %p, align 1
  %insert0 = insertelement <8 x i16> %load, i16 %x, i32 6
  %insert1 = insertelement <8 x i16> %insert0, i16 %y, i32 7
  store <8 x i16> %insert1, ptr %p, align 1
  ret void
}

; A duplicate lane must still be written in insertion order on both targets.
define void @insert_store_duplicate(ptr %p, i16 %x, i16 %y) {
; CHECK-LABEL: @insert_store_duplicate(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP0:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P:%.*]], i32 0, i32 3
; CHECK-NEXT:    store i16 [[X:%.*]], ptr [[GEP0]], align 1
; CHECK-NEXT:    [[GEP1:%.*]] = getelementptr inbounds <8 x i16>, ptr [[P]], i32 0, i32 3
; CHECK-NEXT:    store i16 [[Y:%.*]], ptr [[GEP1]], align 1
; CHECK-NEXT:    ret void
;
entry:
  %load = load <8 x i16>, ptr %p, align 1
  %insert0 = insertelement <8 x i16> %load, i16 %x, i32 3
  %insert1 = insertelement <8 x i16> %insert0, i16 %y, i32 3
  store <8 x i16> %insert1, ptr %p, align 1
  ret void
}
