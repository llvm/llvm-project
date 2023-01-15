; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -mattr=use-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; CHECK:       ********** MI Scheduling **********
; We need second, post-ra scheduling to have VSTM instruction combined from single-stores
; CHECK:       ********** MI Scheduling **********
; CHECK:       schedule starting
; CHECK:       VSTMDIA
; CHECK:       rdefs left
; CHECK-NEXT:  Latency            : 2

%bigVec = type [2 x double]

@var = global %bigVec zeroinitializer

define void @bar(ptr %ptr) {

  %tmp = load %bigVec, ptr %ptr
  store %bigVec %tmp, ptr @var

  ret void
}

