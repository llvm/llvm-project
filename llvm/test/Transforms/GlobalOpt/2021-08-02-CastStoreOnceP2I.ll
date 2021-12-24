; RUN: opt -passes=globalopt -S < %s | FileCheck %s
; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; This tests the assignemnt of non-pointer to global address
; (assert due to D106589).

@a162 = internal global i16* null, align 1

define void @f363() {
; CHECK-LABEL: @f363(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load i16*, i16** @a162, align 1
; CHECK-NEXT:    store i16 0, i16* bitcast (i16** @a162 to i16*), align 1
; CHECK-NEXT:    ret void
;
entry:
  %0 = load i16*, i16** @a162, align 1
  store i16 0, i16* bitcast (i16** @a162 to i16*), align 1
  ret void
}
