; RUN: llc < %s -mtriple=aarch64-apple-darwin | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL:  .section  __LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:   __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 1
; Num LargeConstants
; CHECK-NEXT:   .long   0
; Num Callsites
; CHECK-NEXT:   .long   1

; Functions and stack size
; CHECK-NEXT:   .quad _stackmap_liveness
; CHECK-NEXT:   .quad 16

; Test that the return register is recognized as an live-out.
define i64 @stackmap_liveness(i1 %c) {
; CHECK-LABEL:  .long L{{.*}}-_stackmap_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; Padding
; CHECK-NEXT:   .p2align  3
; CHECK-NEXT:   .short  0
; Num LiveOut Entries: 1
; CHECK-NEXT:   .short  2
; LiveOut Entry 0: X0
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 1: SP
; CHECK-NEXT:   .short 31
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; Align
; CHECK-NEXT:   .p2align  3
  %1 = select i1 %c, i64 1, i64 2
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 32, i8* null, i32 0)
  ret i64 %1
}

declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
