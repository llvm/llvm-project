; RUN: llc -mtriple=arm64-apple-macosx %s -o - | FileCheck %s --check-prefix=CHECK-DARWIN
; RUN: llc -mtriple=arm64-apple-macosx %s -o - -global-isel | FileCheck %s --check-prefix=CHECK-DARWIN
; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK-LINWIN
; RUN: llc -mtriple=aarch64-linux-gnu %s -o - -global-isel | FileCheck %s --check-prefix=CHECK-LINWIN
; RUN: llc -mtriple=aarch64-windows-msvc %s -o - | FileCheck %s --check-prefix=CHECK-LINWIN
; RUN: llc -mtriple=aarch64-windows-msvc %s -o - -global-isel | FileCheck %s --check-prefix=CHECK-LINWIN

declare i16 @foo([8 x i64], i16 signext, i16 signext %a, ...)

define void @bar() {
; CHECK-DARWIN-LABEL: bar:
; CHECK-LINWIN-LABEL: bar:

; CHECK-DARWIN: mov [[TMP:w[0-9]+]], #2752512
; CHECK-DARWIN: str [[TMP]], [sp]

; CHECK-LINWIN: mov [[TMP:w[0-9]+]], #42
; CHECK-LINWIN: str{{h?}} wzr, [sp]
; CHECK-LINWIN: str{{h?}} [[TMP]], [sp, #8]

  call i16([8 x i64], i16, i16, ...) @foo([8 x i64] poison, i16 signext 0, i16 signext 42)
  ret void
}
