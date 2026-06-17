; REQUIRES: arm-registered-target
; RUN: llc -mtriple=armv8a-unknown-linux -verify-machineinstrs < %s -o /dev/null

; Regression test for issue #202263. Lowering the REG_SEQUENCE that builds the
; <2 x i64> keeps a COPY of an undef value which feeds a later partial
; subregister redef. When RegisterCoalescer eliminates that undef copy, the
; redef must be turned into a read-undef def. Otherwise the coalescer asserts
; with "Instruction is reading nonexistent value" on targets that do not track
; subregister liveness (armv8a A-profile has no MVE).

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8a-unknown-linux"

define void @init(i64 %x, i1 %c, ptr %p) {
entry:
  %v0 = insertelement <2 x i64> poison, i64 %x, i64 1
  br i1 %c, label %exit, label %body

body:
  %v1 = insertelement <2 x i64> %v0, i64 1, i64 0
  store <2 x i64> %v1, ptr %p, align 8
  br label %exit

exit:
  ret void
}
