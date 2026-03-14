; RUN: llc -mtriple=thumbv7-windows -verify-machineinstrs %s -o - \
; RUN:  | FileCheck %s

define arm_aapcs_vfpcc void @check_watermark() "no-stack-arg-probe" {
entry:
  %buffer = alloca [4096 x i8], align 1
  ret void
}

; CHECK: check_watermark:
; CHECK-NOT: bl __chkstk
; CHECK: sub.w sp, sp, #4096
