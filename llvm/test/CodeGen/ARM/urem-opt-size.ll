; When optimising for minimum size, we don't want to expand a div to a mul
; and a shift sequence. As a result, the urem instruction e.g. will not be
; expanded to a sequence of umull, lsrs, muls and sub instructions, but
; just a call to __aeabi_uidivmod.
;
; RUN: llc -mtriple=armv7a-eabi -mattr=-neon -verify-machineinstrs %s -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-none-eabi"

define i32 @foo1() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo1:
; CHECK:__aeabi_idiv
; CHECK-NOT: smmul
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %div = sdiv i32 %call, 1000000
  ret i32 %div
}

define i32 @foo2() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo2:
; CHECK: __aeabi_uidiv
; CHECK-NOT: umull
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %div = udiv i32 %call, 1000000
  ret i32 %div
}

define i32 @foo3() local_unnamed_addr #0 {
entry:
; CHECK-LABEL: foo3:
; CHECK: __aeabi_uidivmod
; CHECK-NOT: umull
  %call = tail call i32 bitcast (i32 (...)* @GetValue to i32 ()*)()
  %rem = urem i32 %call, 1000000
  %cmp = icmp eq i32 %rem, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

declare i32 @GetValue(...) local_unnamed_addr

attributes #0 = { minsize nounwind optsize }
