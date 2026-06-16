; RUN: llc -mtriple=armv8a-unknown-linux -verify-machineinstrs < %s -o /dev/null

; This used to assert in RegisterCoalescer:
;
;   (TrackSubRegLiveness || V.RedefVNI) &&
;   "Instruction is reading nonexistent value"
;
; When an undef copy value feeds a later partial redef, eliminating the copy
; can expose a partial redef with no incoming value.  Keep the undef value as
; an IMPLICIT_DEF instead.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8a-unknown-linux"

define void @init(i64 %x, i1 %min.iters.check, ptr %p) {
entry:
  %v0 = insertelement <2 x i64> poison, i64 %x, i64 1
  br i1 %min.iters.check, label %common.ret, label %vector.body

vector.body:
  %v1 = insertelement <2 x i64> %v0, i64 1, i64 0
  store <2 x i64> %v1, ptr %p, align 8
  br label %common.ret

common.ret:
  ret void
}
