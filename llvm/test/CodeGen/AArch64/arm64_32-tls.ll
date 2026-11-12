; RUN: llc -mtriple=arm64_32-apple-ios %s -o - | FileCheck %s

define i32 @test_thread_local() {
; CHECK-LABEL: test_thread_local:
; CHECK: adrp x[[TMP:[0-9]+]], _var@TLVPPAGE
; CHECK: ldr w0, [x[[TMP]], _var@TLVPPAGEOFF]
; CHECK: ldr w[[DEST:[0-9]+]], [x0]
; CHECK: blr x[[DEST]]

  %val = load i32, ptr @var
  ret i32 %val
}

@var = thread_local global i32 zeroinitializer

; CHECK: .tbss _var$tlv$init, 4, 2

; CHECK-LABEL: __DATA,__thread_vars
; CHECK: _var:
; CHECK:    .long __tlv_bootstrap
; CHECK:    .long 0
; CHECK:    .long _var$tlv$init
