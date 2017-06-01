; RUN: llc -mtriple thumbv7-unknown-none-eabi -float-abi soft -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MATCH
; RUN: llc -mtriple thumbv7-unknown-none-eabi -float-abi hard -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MISMATCH -check-prefix CHECK-TO-SOFT
; RUN: llc -mtriple thumbv7-unknown-none-eabihf -float-abi soft -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MISMATCH -check-prefix CHECK-TO-HARD
; RUN: llc -mtriple thumbv7-unknown-none-eabihf -float-abi hard -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MATCH

; RUN: llc -mtriple thumbv7-unknown-none-gnueabi -float-abi soft -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MATCH
; RUN: llc -mtriple thumbv7-unknown-none-gnueabi -float-abi hard -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MISMATCH -check-prefix CHECK-TO-SOFT
; RUN: llc -mtriple thumbv7-unknown-none-gnueabihf -float-abi soft -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MISMATCH -check-prefix CHECK-TO-HARD
; RUN: llc -mtriple thumbv7-unknown-none-gnueabihf -float-abi hard -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MATCH

; RUN: llc -mtriple thumbv7-unknown-none-musleabi -float-abi soft -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MATCH
; RUN: llc -mtriple thumbv7-unknown-none-musleabi -float-abi hard -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MISMATCH -check-prefix CHECK-TO-SOFT
; RUN: llc -mtriple thumbv7-unknown-none-musleabihf -float-abi soft -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MISMATCH -check-prefix CHECK-TO-HARD
; RUN: llc -mtriple thumbv7-unknown-none-musleabihf -float-abi hard -filetype asm -o - %s | FileCheck %s -check-prefix CHECK-MATCH

declare float @llvm.powi.f32(float, i32)

define float @f(float %f, i32 %i) {
entry:
  %0 = call float @llvm.powi.f32(float %f, i32 %i)
  ret float %0
}

; CHECK-MATCH: b __powisf2
; CHECK-MISMATCH: bl __powisf2
; CHECK-TO-SOFT: vmov s0, r0
; CHECK-TO-HARD: vmov r0, s0

declare double @llvm.powi.f64(double, i32)

define double @g(double %d, i32 %i) {
entry:
  %0 = call double @llvm.powi.f64(double %d, i32 %i)
  ret double %0
}

; CHECK-MATCH: b __powidf2
; CHECK-MISMATCH: bl __powidf2
; CHECK-TO-SOFT: vmov d0, r0, r1
; CHECK-TO-HARD: vmov r0, r1, d0

