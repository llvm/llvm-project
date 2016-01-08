; RUN: llc -float-abi soft -mattr=+fp16 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=SOFT
; RUN: llc -float-abi hard -mattr=+fp16 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=HARD

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7a--none-eabi"

define float @foo(float %a.coerce, float %b.coerce) {
entry:
  %0 = bitcast float %a.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %2 = bitcast float %b.coerce to i32
  %tmp1.0.extract.trunc = trunc i32 %2 to i16
  %3 = bitcast i16 %tmp1.0.extract.trunc to half
  %4 = fadd half %1, %3
  %5 = bitcast half %4 to i16
  %tmp5.0.insert.ext = zext i16 %5 to i32
  %6 = bitcast i32 %tmp5.0.insert.ext to float
  ret float %6
; CHECK: foo:

; SOFT: vmov    {{s[0-9]+}}, r1
; SOFT: vmov    {{s[0-9]+}}, r0
; SOFT: vcvtb.f32.f16   {{s[0-9]+}}, {{s[0-9]+}}
; SOFT: vcvtb.f32.f16   {{s[0-9]+}}, {{s[0-9]+}}
; SOFT: vadd.f32        {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; SOFT: vcvtb.f16.f32   {{s[0-9]+}}, {{s[0-9]+}}
; SOFT: vmov    r0, {{s[0-9]+}}

; HARD-NOT: vmov
; HARD-NOT: uxth
; HARD: vcvtb.f32.f16   {{s[0-9]+}}, s1
; HARD: vcvtb.f32.f16   {{s[0-9]+}}, s0
; HARD: vadd.f32        {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
; HARD: vcvtb.f16.f32   [[SREG:s[0-9]+]], {{s[0-9]+}}
; HARD-NEXT: vmov            [[REG0:r[0-9]+]], [[SREG]]
; HARD-NEXT: uxth            [[REG1:r[0-9]+]], [[REG0]]
; HARD-NEXT: vmov            s0, [[REG1]]

; CHECK: bx lr
}
