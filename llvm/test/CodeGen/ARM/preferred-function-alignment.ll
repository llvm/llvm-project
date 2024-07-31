; RUN: llc -mtriple=arm-none-eabi -mcpu=cortex-m85 < %s | FileCheck --check-prefixes=CHECK,ALIGN-16,ALIGN-CS-16 %s
; RUN: llc -mtriple=arm-none-eabi -mcpu=cortex-m23 < %s | FileCheck --check-prefixes=CHECK,ALIGN-16,ALIGN-CS-16 %s

; RUN: llc -mtriple=arm-none-eabi -mcpu=cortex-a5 < %s  | FileCheck --check-prefixes=CHECK,ALIGN-32,ALIGN-CS-32 %s
; RUN: llc -mtriple=arm-none-eabi -mcpu=cortex-m33 < %s | FileCheck --check-prefixes=CHECK,ALIGN-32,ALIGN-CS-16 %s
; RUN: llc -mtriple=arm-none-eabi -mcpu=cortex-m55 < %s | FileCheck --check-prefixes=CHECK,ALIGN-32,ALIGN-CS-16 %s


; CHECK-LABEL: test
; ALIGN-16: .p2align 1
; ALIGN-32: .p2align 2

define void @test() {
  ret void
}

; CHECK-LABEL: test_optsize
; ALIGN-CS-16: .p2align 1
; ALIGN-CS-32: .p2align 2

define void @test_optsize() optsize {
  ret void
}
