; RUN: llc < %s -mtriple=armv7-apple-darwin -verify-machineinstrs   | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -verify-machineinstrs | FileCheck %s -check-prefix=T2
; rdar://8964854

define i8 @t(i8* %a, i8 %b, i8 %c) nounwind {
; ARM: t:
; ARM: ldrexb
; ARM: strexb

; T2: t:
; T2: ldrexb
; T2: strexb
  %tmp0 = cmpxchg i8* %a, i8 %b, i8 %c monotonic
  ret i8 %tmp0
}
