; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-r52 | FileCheck %s --check-prefix=CHECK --check-prefix=USEAA
; RUN: llc < %s -mtriple=armv7m-eabi -mcpu=cortex-m4 | FileCheck %s --check-prefix=CHECK --check-prefix=USEAA
; RUN: llc < %s -mtriple=armv8m-eabi -mcpu=cortex-m33 | FileCheck %s --check-prefix=CHECK --check-prefix=USEAA
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=generic | FileCheck %s --check-prefix=CHECK --check-prefix=GENERIC

; Check we use AA during codegen, so can interleave these loads/stores.

; CHECK-LABEL: test
; GENERIC: ldr
; GENERIC: str
; GENERIC: ldr
; GENERIC: str
; USEAA: ldr
; USEAA: ldr
; USEAA: str
; USEAA: str

define void @test(i32* nocapture %a, i32* noalias nocapture %b) {
entry:
  %0 = load i32, i32* %a, align 4
  %add = add nsw i32 %0, 10
  store i32 %add, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add2 = add nsw i32 %1, 20
  store i32 %add2, i32* %b, align 4
  ret void
}

