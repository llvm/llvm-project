; RUN: not llc -mtriple=aarch64-windows-msvc -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error:
; CHECK-SAME: A-key return address signing is unsupported on AArch64 Windows

define void @a_key() "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" {
  call void @callee()
  ret void
}

declare void @callee()
