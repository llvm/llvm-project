; RUN: opt -S -passes=mergefunc %s | FileCheck %s

target triple = "x86_64-unknown-windows-msvc19.42.34436"

@symbols = linkonce_odr global <{ ptr, ptr }> <{ ptr @f, ptr @g }>

$g = comdat any
$f = comdat any

define linkonce_odr hidden i32 @f(i32 %x, i32 %y) comdat {
  %sum = add i32 %x, %y
  %sum2 = add i32 %x, %sum
  %sum3 = add i32 %x, %sum
  ret i32 %sum3
}

define linkonce_odr hidden i32 @g(i32 %x, i32 %y) comdat {
  %sum = add i32 %x, %y
  %sum2 = add i32 %x, %sum
  %sum3 = add i32 %x, %sum
  ret i32 %sum3
}

; CHECK: $g = comdat any
; CHECK: $f = comdat any

;.
; CHECK: @symbols = linkonce_odr global <{ ptr, ptr }> <{ ptr @f, ptr @g }>
;.
; CHECK-LABEL: define private i32 @0(
; CHECK-SAME: i32 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:    [[SUM:%.*]] = add i32 [[X]], [[Y]]
; CHECK-NEXT:    [[SUM2:%.*]] = add i32 [[X]], [[SUM]]
; CHECK-NEXT:    [[SUM3:%.*]] = add i32 [[X]], [[SUM]]
; CHECK-NEXT:    ret i32 [[SUM3]]
;
;
; CHECK-LABEL: define linkonce_odr hidden i32 @g(
; CHECK-SAME: i32 [[TMP0:%.*]], i32 [[TMP1:%.*]]) comdat {
; CHECK-NEXT:    [[TMP3:%.*]] = tail call i32 @[[GLOB0:[0-9]+]](i32 [[TMP0]], i32 [[TMP1]])
; CHECK-NEXT:    ret i32 [[TMP3]]
;
;
; CHECK-LABEL: define linkonce_odr hidden i32 @f(
; CHECK-SAME: i32 [[TMP0:%.*]], i32 [[TMP1:%.*]]) comdat {
; CHECK-NEXT:    [[TMP3:%.*]] = tail call i32 @[[GLOB0]](i32 [[TMP0]], i32 [[TMP1]])
; CHECK-NEXT:    ret i32 [[TMP3]]
;
