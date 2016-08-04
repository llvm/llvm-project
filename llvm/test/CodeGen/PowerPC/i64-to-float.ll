; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define float @foo(i64 %a) nounwind {
entry:
  %x = sitofp i64 %a to float
  ret float %x

; CHECK: @foo
; CHECK: std 3,
; CHECK: lfd [[REG:[0-9]+]],
; CHECK: fcfids 1, [[REG]]
; CHECK: blr

; CHECK-VSX: @foo
; CHECK-VSX: std 3,
; CHECK-VSX: lxsdx [[REG:[0-9]+]],
; CHECK-VSX: fcfids 1, [[REG]]
; CHECK-VSX: blr
}

define double @goo(i64 %a) nounwind {
entry:
  %x = sitofp i64 %a to double
  ret double %x

; CHECK: @goo
; CHECK: std 3,
; CHECK: lfd [[REG:[0-9]+]],
; CHECK: fcfid 1, [[REG]]
; CHECK: blr

; CHECK-VSX: @goo
; CHECK-VSX: std 3,
; CHECK-VSX: lxsdx [[REG:[0-9]+]],
; CHECK-VSX: xscvsxddp 1, [[REG]]
; CHECK-VSX: blr
}

define float @foou(i64 %a) nounwind {
entry:
  %x = uitofp i64 %a to float
  ret float %x

; CHECK: @foou
; CHECK: std 3,
; CHECK: lfd [[REG:[0-9]+]],
; CHECK: fcfidus 1, [[REG]]
; CHECK: blr

; CHECK-VSX: @foou
; CHECK-VSX: std 3,
; CHECK-VSX: lxsdx [[REG:[0-9]+]],
; CHECK-VSX: fcfidus 1, [[REG]]
; CHECK-VSX: blr
}

define double @goou(i64 %a) nounwind {
entry:
  %x = uitofp i64 %a to double
  ret double %x

; CHECK: @goou
; CHECK: std 3,
; CHECK: lfd [[REG:[0-9]+]],
; CHECK: fcfidu 1, [[REG]]
; CHECK: blr

; CHECK-VSX: @goou
; CHECK-VSX: std 3,
; CHECK-VSX: lxsdx [[REG:[0-9]+]],
; CHECK-VSX: xscvuxddp 1, [[REG]]
; CHECK-VSX: blr
}

