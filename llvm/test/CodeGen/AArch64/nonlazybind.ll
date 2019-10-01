; RUN: llc -mtriple=aarch64-apple-ios %s -o - -aarch64-enable-nonlazybind | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-ios %s -o - | FileCheck %s --check-prefix=CHECK-NORMAL

define void @local() nonlazybind {
  ret void
}

declare void @nonlocal() nonlazybind

define void @test_laziness() {
; CHECK-LABEL: test_laziness:

; CHECK: bl _local

; CHECK: adrp x[[TMP:[0-9]+]], _nonlocal@GOTPAGE
; CHECK: ldr [[FUNC:x[0-9]+]], [x[[TMP]], _nonlocal@GOTPAGEOFF]
; CHECK: blr [[FUNC]]

; CHECK-NORMAL-LABEL: test_laziness:
; CHECK-NORMAL: bl _local
; CHECK-NORMAL: bl _nonlocal

  call void @local()
  call void @nonlocal()
  ret void
}

define void @test_laziness_tail() {
; CHECK-LABEL: test_laziness_tail:

; CHECK: adrp x[[TMP:[0-9]+]], _nonlocal@GOTPAGE
; CHECK: ldr [[FUNC:x[0-9]+]], [x[[TMP]], _nonlocal@GOTPAGEOFF]
; CHECK: br [[FUNC]]

; CHECK-NORMAL-LABEL: test_laziness_tail:
; CHECK-NORMAL: b _nonlocal

  tail call void @nonlocal()
  ret void
}
