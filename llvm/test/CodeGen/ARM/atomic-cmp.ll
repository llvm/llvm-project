; RUN: llc < %s -mtriple=armv7-apple-darwin -verify-machineinstrs   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -verify-machineinstrs | FileCheck %s -check-prefix=T2
; rdar://8964854

define i8 @t(i8* %a, i8 %b, i8 %c) nounwind {
; ARM-LABEL: t:
; ARM: ldrexb
; ARM: strexb
; ARM: clrex

; T2-LABEL: t:
; T2: strexb
; T2: ldrexb
; T2: clrex
  %tmp0 = cmpxchg i8* %a, i8 %b, i8 %c monotonic monotonic
  %tmp1 = extractvalue { i8, i1 } %tmp0, 0
  ret i8 %tmp1
}
