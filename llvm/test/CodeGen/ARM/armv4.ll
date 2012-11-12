; RUN: llc < %s -mtriple=armv4t-unknown-eabi | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -mtriple=armv4-unknown-eabi -mcpu=strongarm | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv7-unknown-eabi -mcpu=cortex-a8 | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -mtriple=armv6-unknown-eabi | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -mtriple=armv4-unknown-eabi | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv4t-unknown-eabi | FileCheck %s -check-prefix=THUMB

define i32 @test(i32 %a) nounwind readnone {
entry:
; ARM: mov pc
; THUMB: bx
  ret i32 %a
}
