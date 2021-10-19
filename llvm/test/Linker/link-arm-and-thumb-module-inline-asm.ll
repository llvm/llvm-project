; This test checks that proper directives to switch between ARM and Thumb mode
; are added when linking ARM and Thumb modules.

; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/thumb-module-inline-asm.ll -o %t2.bc
; RUN: llvm-link %t1.bc %t2.bc -S 2> %t3.out | FileCheck %s

target triple = "armv7-linux-gnueabihf"

module asm "add r1, r2, r2"

; CHECK:      module asm ".text"
; CHECK-NEXT: module asm ".balign 4"
; CHECK-NEXT: module asm ".arm"
; CHECK-NEXT: module asm "add r1, r2, r2"
; CHECK-NEXT: module asm ".text"
; CHECK-NEXT: module asm ".balign 2"
; CHECK-NEXT: module asm ".thumb"
; CHECK-NEXT: module asm "orn r1, r2, r2"
