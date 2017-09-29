; RUN: opt -S -codegenprepare < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; We only use the div instruction -- the rem should be DCE'ed.
; CHECK-LABEL: @div_only
define void @div_only(i64 %a, i64 %b, i64* %retptr) {
  ; CHECK: udiv i32
  ; CHECK-NOT: urem
  ; CHECK: sdiv i64
  ; CHECK-NOT: rem
  %d = sdiv i64 %a, %b
  store i64 %d, i64* %retptr
  ret void
}

; We only use the rem instruction -- the div should be DCE'ed.
; CHECK-LABEL: @rem_only
define void @rem_only(i64 %a, i64 %b, i64* %retptr) {
  ; CHECK-NOT: div
  ; CHECK: urem i32
  ; CHECK-NOT: div
  ; CHECK: rem i64
  ; CHECK-NOT: div
  %d = srem i64 %a, %b
  store i64 %d, i64* %retptr
  ret void
}
