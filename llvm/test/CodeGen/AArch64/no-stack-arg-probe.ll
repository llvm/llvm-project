; RUN: llc -mtriple=aarch64-windows -verify-machineinstrs %s -o - \
; RUN:  | FileCheck %s

define void @check_watermark() "no-stack-arg-probe" {
entry:
  %buffer = alloca [4096 x i8], align 1
  ret void
}

; CHECK: check_watermark:
; CHECK: sub sp, sp, #1, lsl #12
; CHECK-NOT: bl __chkstk
