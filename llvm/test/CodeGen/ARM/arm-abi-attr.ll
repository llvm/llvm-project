; RUN: llc -mtriple=arm-linux-gnu < %s | FileCheck %s --check-prefix=AAPCS
; RUN: llc -mtriple=arm-linux-gnu -target-abi=apcs < %s | \
; RUN: FileCheck %s --check-prefix=APCS
; RUN: llc -mtriple=arm-linux-gnueabi -target-abi=apcs < %s | \
; RUN: FileCheck %s --check-prefix=APCS

; RUN: llc -mtriple=arm-linux-gnueabi < %s | FileCheck %s --check-prefix=AAPCS
; RUN: llc -mtriple=arm-linux-gnueabi -target-abi=aapcs < %s | \
; RUN: FileCheck %s --check-prefix=AAPCS
; RUN: llc -mtriple=arm-linux-gnu -target-abi=aapcs < %s | \
; RUN: FileCheck %s --check-prefix=AAPCS

; The stack is 8 byte aligned on AAPCS and 4 on APCS, so we should get a BIC
; only on APCS.

define void @g() {
; APCS: sub	sp, sp, #8
; APCS: bic	sp, sp, #7

; AAPCS: sub	sp, sp, #8
; AAPCS-NOT: bic

  %c = alloca i8, align 8
  call void @f(i8* %c)
  ret void
}

declare void @f(i8*)
