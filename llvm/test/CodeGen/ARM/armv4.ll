; RUN: llc < %s -mtriple=armv4t-unknown-eabi | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -mtriple=armv4-unknown-eabi -mcpu=strongarm | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv7-unknown-eabi -mcpu=cortex-a8 | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -mtriple=armv6-unknown-eabi | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -mtriple=armv4-unknown-eabi | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv4t-unknown-eabi | FileCheck %s -check-prefix=THUMB

define i32 @test_return(i32 %a) nounwind readnone {
entry:
; ARM-LABEL: test_return
; ARM: mov pc
; THUMB-LABEL: test_return
; THUMB: bx
  ret i32 %a
}

@helper = global i32 ()* null, align 4

define i32 @test_indirect() #0 {
entry:
; ARM-LABEL: test_indirect
; ARM: mov pc
; THUMB-LABEL: test_indirect
; THUMB: bx
  %0 = load i32 ()*, i32 ()** @helper, align 4
  %call = tail call i32 %0()
  ret i32 %call
}
