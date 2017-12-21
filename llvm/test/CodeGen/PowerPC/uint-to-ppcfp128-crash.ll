; RUN: llc -verify-machineinstrs -mcpu=pwr9 \
; RUN:   -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s

; Ensure we don't crash by trying to convert directly from a subword load
; to a ppc_fp128 as we do for conversions to f32/f64.
define ppc_fp128 @test(i16* nocapture readonly %Ptr) {
entry:
  %0 = load i16, i16* %Ptr, align 2
  %conv = uitofp i16 %0 to ppc_fp128
  ret ppc_fp128 %conv
; CHECK: lhz [[LD:[0-9]+]], 0(3)
; CHECK: mtvsrwa [[MV:[0-9]+]], [[LD]]
; CHECK: xscvsxddp [[CONV:[0-9]+]], [[MV]]
; CHECK: bl __gcc_qadd
}
