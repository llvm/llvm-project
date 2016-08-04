; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=g5 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr6 | FileCheck -check-prefix=CHECK-PWR6 %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 | FileCheck -check-prefix=CHECK-A2 %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=+vsx | FileCheck -check-prefix=CHECK-VSX %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define float @foo(i32 %a) nounwind {
entry:
  %x = sitofp i32 %a to float
  ret float %x

; CHECK: @foo
; CHECK: extsw [[REG:[0-9]+]], 3
; CHECK: std [[REG]],
; CHECK: lfd [[REG2:[0-9]+]],
; CHECK: fcfid [[REG3:[0-9]+]], [[REG2]]
; CHECK: frsp 1, [[REG3]]
; CHECK: blr

; CHECK-PWR6: @foo
; CHECK-PWR6: stw 3,
; CHECK-PWR6: lfiwax [[REG:[0-9]+]],
; CHECK-PWR6: fcfid [[REG2:[0-9]+]], [[REG]]
; CHECK-PWR6: frsp 1, [[REG2]]
; CHECK-PWR6: blr

; CHECK-A2: @foo
; CHECK-A2: stw 3,
; CHECK-A2: lfiwax [[REG:[0-9]+]],
; CHECK-A2: fcfids 1, [[REG]]
; CHECK-A2: blr

; CHECK-VSX: @foo
; CHECK-VSX: stw 3,
; CHECK-VSX: lfiwax [[REG:[0-9]+]],
; CHECK-VSX: fcfids 1, [[REG]]
; CHECK-VSX: blr
}

define double @goo(i32 %a) nounwind {
entry:
  %x = sitofp i32 %a to double
  ret double %x

; CHECK: @goo
; CHECK: extsw [[REG:[0-9]+]], 3
; CHECK: std [[REG]],
; CHECK: lfd [[REG2:[0-9]+]],
; CHECK: fcfid 1, [[REG2]]
; CHECK: blr

; CHECK-PWR6: @goo
; CHECK-PWR6: stw 3,
; CHECK-PWR6: lfiwax [[REG:[0-9]+]],
; CHECK-PWR6: fcfid 1, [[REG]]
; CHECK-PWR6: blr

; CHECK-A2: @goo
; CHECK-A2: stw 3,
; CHECK-A2: lfiwax [[REG:[0-9]+]],
; CHECK-A2: fcfid 1, [[REG]]
; CHECK-A2: blr

; CHECK-VSX: @goo
; CHECK-VSX: stw 3,
; CHECK-VSX: lfiwax [[REG:[0-9]+]],
; CHECK-VSX: xscvsxddp 1, [[REG]]
; CHECK-VSX: blr
}

define float @foou(i32 %a) nounwind {
entry:
  %x = uitofp i32 %a to float
  ret float %x

; CHECK-A2: @foou
; CHECK-A2: stw 3,
; CHECK-A2: lfiwzx [[REG:[0-9]+]],
; CHECK-A2: fcfidus 1, [[REG]]
; CHECK-A2: blr

; CHECK-VSX: @foou
; CHECK-VSX: stw 3,
; CHECK-VSX: lfiwzx [[REG:[0-9]+]],
; CHECK-VSX: fcfidus 1, [[REG]]
; CHECK-VSX: blr
}

define double @goou(i32 %a) nounwind {
entry:
  %x = uitofp i32 %a to double
  ret double %x

; CHECK-A2: @goou
; CHECK-A2: stw 3,
; CHECK-A2: lfiwzx [[REG:[0-9]+]],
; CHECK-A2: fcfidu 1, [[REG]]
; CHECK-A2: blr

; CHECK-VSX: @goou
; CHECK-VSX: stw 3,
; CHECK-VSX: lfiwzx [[REG:[0-9]+]],
; CHECK-VSX: xscvuxddp 1, [[REG]]
; CHECK-VSX: blr
}

