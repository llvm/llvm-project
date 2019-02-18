; RUN: llc -mtriple=ppc64-- -ppc-always-use-base-pointer < %s | FileCheck %s --check-prefix CHECK --check-prefix PPC64
; RUN: llc -ppc-always-use-base-pointer -relocation-model=static < %s | FileCheck %s --check-prefix CHECK --check-prefix PPC32
; RUN: llc -ppc-always-use-base-pointer -relocation-model=pic < %s | FileCheck %s --check-prefix CHECK --check-prefix PPC32PIC

; CHECK-LABEL: fred:

; Check for saving/restoring frame pointer (X31) and base pointer (X30)
; on ppc64:
; PPC64: std 31, -8(1)
; PPC64: std 30, -16(1)
; PPC64: ld 31, -8(1)
; PPC64: ld 30, -16(1)

; Check for saving/restoring frame pointer (R31) and base pointer (R30)
; on ppc32:
; PPC32: stwux 1, 1, 0
; PPC32; addic 0, 0, -4
; PPC32: stwx 31, 0, 0
; PPC32: addic 0, 0, -4
; PPC32: stwx 30, 0, 0
; The restore sequence:
; PPC32: lwz 31, 0(1)
; PPC32: addic 30, 0, 8
; PPC32: lwz 0, -4(31)
; PPC32: lwz 30, -8(31)
; PPC32: mr 1, 31
; PPC32: mr 31, 0

; Check for saving/restoring frame pointer (R31) and base pointer (R29)
; on ppc32/pic. This is mostly the same as without pic, except that base
; pointer is in R29.
; PPC32PIC: stwux 1, 1, 0
; PPC32PIC; addic 0, 0, -4
; PPC32PIC: stwx 31, 0, 0
; PPC32PIC: addic 0, 0, -8
; PPC32PIC: stwx 29, 0, 0
; The restore sequence:
; PPC32PIC: lwz 31, 0(1)
; PPC32PIC: addic 29, 0, 12
; PPC32PIC: lwz 0, -4(31)
; PPC32PIC: lwz 29, -12(31)
; PPC32PIC: mr 1, 31
; PPC32PIC: mr 31, 0


target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc-unknown-freebsd"

define i64 @fred() local_unnamed_addr #0 {
entry:
  ret i64 0
}

attributes #0 = { norecurse readnone nounwind sspstrong "no-frame-pointer-elim"="true" "target-cpu"="ppc" }
