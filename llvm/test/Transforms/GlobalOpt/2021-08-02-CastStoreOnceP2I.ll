; RUN: opt -passes=globalopt -S < %s | FileCheck %s

; This tests the assignemnt of non-pointer to global address
; (assert due to D106589).

@a162 = internal global ptr null, align 1

define void @f363() {
; CHECK-LABEL: @f363(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr @a162, align 1
; CHECK-NEXT:    store i16 0, ptr @a162, align 1
; CHECK-NEXT:    ret void
;
entry:
  %0 = load ptr, ptr @a162, align 1
  store i16 0, ptr @a162, align 1
  ret void
}
