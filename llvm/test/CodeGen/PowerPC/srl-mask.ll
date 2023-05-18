; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define i64 @foo(i64 %x) #0 {
entry:
; CHECK-LABEL: @foo
  %a = lshr i64 %x, 35
  %b = and i64 %a, 65535
; CHECK: rldicl 3, 3, 29, 48
  ret i64 %b
; CHECK: blr
}

; for AND with an immediate like (x & ~0xFFFF)
; we should use rldicl instruction
define i64 @bar(i64 %x) #0 {
entry:
; CHECK-LABEL: @bar
  %a = and i64 %x, 18446744073709486080
; CHECK: rldicr 3, 3, 0, 47
  ret i64 %a
; CHECK: blr
}

attributes #0 = { nounwind }

