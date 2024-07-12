;; Ensure that the MergeFunctions pass creates thunks with the appropriate debug
;; info format set (which would otherwise assert when inlining those thunks).
; RUN: opt -S -passes=mergefunc,inline --try-experimental-debuginfo-iterators < %s | FileCheck %s

declare void @f1()
declare void @f2()

define void @f3() {
  call void @f1()
  call void @f2()
  ret void
}

;; MergeFunctions will replace f4 with a thunk that calls f3. Inlining will
;; inline f3 into that thunk, which would assert if the thunk had the incorrect
;; debug info format.
define void @f4() {
  call void @f1()
  call void @f2()
  ret void
}

; CHECK-LABEL: define void @f4() {
; CHECK-NEXT:    call void @f1()
; CHECK-NEXT:    call void @f2()
; CHECK-NEXT:    ret void
; CHECK-NEXT: }

;; Both of these are interposable, so MergeFunctions will create a common thunk
;; that both will call. Inlining will inline that thunk back, which would assert
;; if the thunk had the incorrect debug info format.
define weak void @f5() {
  call void @f2()
  call void @f1()
  ret void
}

define weak void @f6() {
  call void @f2()
  call void @f1()
  ret void
}

; CHECK-LABEL: define weak void @f6() {
; CHECK-NEXT:    call void @f2()
; CHECK-NEXT:    call void @f1()
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

; CHECK-LABEL: define weak void @f5() {
; CHECK-NEXT:    call void @f2()
; CHECK-NEXT:    call void @f1()
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
