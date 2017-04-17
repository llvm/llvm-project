; RUN: llc -mtriple=aarch64-apple-ios %s -o - | FileCheck %s

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

  call void @local()
  call void @nonlocal()
  ret void
}

define void @test_laziness_tail() {
; CHECK-LABEL: test_laziness_tail:

; CHECK: adrp x[[TMP:[0-9]+]], _nonlocal@GOTPAGE
; CHECK: ldr [[FUNC:x[0-9]+]], [x[[TMP]], _nonlocal@GOTPAGEOFF]
; CHECK: br [[FUNC]]

  tail call void @nonlocal()
  ret void
}
