; RUN: llc < %s -mtriple=armv4t-unknown-linux-gnueabi -verify-machineinstrs | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv4t-unknown-linux-gnueabi"

@a = global i32 0, align 4

define i32 @foo() {
entry:
  ret i32 and (i32 ptrtoint (ptr @a to i32), i32 255)
}

; CHECK-LABEL: foo:
; CHECK: ldrb    r0, .LCPI0_0
