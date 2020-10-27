; RUN: llc < %s -mtriple=arm64_32-apple-watchos2.0.0 --aarch64-neon-syntax=generic | FileCheck %s

target datalayout = "e-m:o-p:32:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64_32-apple-watchos2.0.0"

; CHECK-LABEL: lCPI0_0:
; CHECK-NEXT:    .quad 36
; CHECK-NEXT:    .quad 4804

define <2 x i8*> @vector_gep(<2 x i8*> %0) {
; CHECK-LABEL: vector_gep:
; CHECK:         adrp x[[REG8:[123]?[0-9]]], lCPI0_0@PAGE
; CHECK:         ldr q[[REG1:[0-9]+]], [x[[REG8]], lCPI0_0@PAGEOFF]
; CHECK:         add v[[REG0:[0-9]+]].2d, v[[REG0]].2d, v[[REG1]].2d
; CHECK:         movi v[[REG1]].2d, #0x000000ffffffff
; CHECK:         and v[[REG0]].16b, v[[REG0]].16b, v[[REG1]].16b
; CHECK:         ret
entry:
  %1 = getelementptr i8, <2 x i8*> %0, <2 x i32> <i32 36, i32 4804>
  ret <2 x i8*> %1
}
