; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Test that we can generate an fcmove, and also that it passes verification.

; CHECK-LABEL: cmove_f
; CHECK: fcmove %st({{[0-7]}}), %st
define x86_fp80 @cmove_f(x86_fp80 %a, x86_fp80 %b, i32 %c) {
  %test = icmp eq i32 %c, 0
  %add = fadd x86_fp80 %a, %b
  %ret = select i1 %test, x86_fp80 %add, x86_fp80 %b
  ret x86_fp80 %ret
}
