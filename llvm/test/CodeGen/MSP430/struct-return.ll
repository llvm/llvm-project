; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

; Allow simple structures to be returned by value.

%s = type { i64, i64 }

define %s @fred() #0 {
; CHECK-LABEL: fred:
; CHECK: mov.w	#2314, 14(r12)
; CHECK: mov.w	#2828, 12(r12)
; CHECK: mov.w	#3342, 10(r12)
; CHECK: mov.w	#3840, 8(r12)
; CHECK: mov.w	#258, 6(r12)
; CHECK: mov.w	#772, 4(r12)
; CHECK: mov.w	#1286, 2(r12)
; CHECK: mov.w	#1800, 0(r12)
  ret %s {i64 72623859790382856, i64 651345242494996224} 
}

attributes #0 = { nounwind }
