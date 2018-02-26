; RUN: llc -mtriple thumbv7-unknown-none-eabi -float-abi soft -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-eabi -float-abi hard -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-eabihf -float-abi soft -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-eabihf -float-abi hard -filetype asm -o - %s | FileCheck %s

; RUN: llc -mtriple thumbv7-unknown-none-gnueabi -float-abi soft -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-gnueabi -float-abi hard -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-gnueabihf -float-abi soft -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-gnueabihf -float-abi hard -filetype asm -o - %s | FileCheck %s

; RUN: llc -mtriple thumbv7-unknown-none-musleabi -float-abi soft -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-musleabi -float-abi hard -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-musleabihf -float-abi soft -filetype asm -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-unknown-none-musleabihf -float-abi hard -filetype asm -o - %s | FileCheck %s

declare float @llvm.powi.f32(float, i32)

define float @powi_f32(float %f, i32 %i) {
entry:
  %0 = call float @llvm.powi.f32(float %f, i32 %i)
  ret float %0
}

; CHECK: b __powisf2

declare double @llvm.powi.f64(double, i32)

define double @powi_f64(double %d, i32 %i) {
entry:
  %0 = call double @llvm.powi.f64(double %d, i32 %i)
  ret double %0
}

; CHECK: b __powidf2

declare float @llvm.floor.f32(float)

define float @floor_f32(float %f) {
entry:
  %0 = call float @llvm.floor.f32(float %f)
  ret float %0
}

; CHECK: b floorf

declare double @llvm.floor.f64(double)

define double @floor_f64(double %d) {
entry:
  %0 = call double @llvm.floor.f64(double %d)
  ret double %0
}

; CHECK: b floor
