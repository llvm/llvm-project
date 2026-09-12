; RUN: llc -mtriple riscv32 -mattr=+experimental-y -target-abi il32pc64 %s -o - | FileCheck -check-prefix=RVY32 %s
; RUN: llc -mtriple riscv64 -mattr=+experimental-y -target-abi l64pc128 %s -o - | FileCheck -check-prefix=RVY64 %s
; RUN: llc -mtriple riscv64 -mattr=+xcheriot -target-abi cheriot %s -o - | FileCheck -check-prefix=CHERIOT %s

@global1 = global [6995 x i8] zeroinitializer, align 1

; RVY32-LABEL: .globl global1
; RVY32-NEXT:  .p2align 7, 0x0
; RVY32-NEXT:  global1:
; RVY32-NEXT:  .zero 6995
; RVY32-NEXT:  .size global1, 7040
; RVY32:  .p2align 7, 0x0

; RVY64-LABEL: .globl global1
; RVY64-NEXT:  .p2align 3, 0x0
; RVY64-NEXT:  global1:
; RVY64-NEXT:  .zero 6995
; RVY64-NEXT:  .size global1, 7000
; RVY64:  .p2align 3, 0x0

; CHERIOT-LABEL: .globl global1
; CHERIOT-NEXT:  .p2align 4, 0x0
; CHERIOT-NEXT:  global1:
; CHERIOT-NEXT:  .zero 6995
; CHERIOT-NEXT:  .size global1, 7008
; CHERIOT:  .p2align 4, 0x0
