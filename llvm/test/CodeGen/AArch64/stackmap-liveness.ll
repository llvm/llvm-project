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
; Num LiveOut Entries: 20
; CHECK-NEXT:   .short  20
; LiveOut Entry 1: X0
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 2:
; CHECK-NEXT:   .short 19
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 3:
; CHECK-NEXT:   .short 20
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 4:
; CHECK-NEXT:   .short 21
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 5:
; CHECK-NEXT:   .short 22
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 6:
; CHECK-NEXT:   .short 23
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 7:
; CHECK-NEXT:   .short 24
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 8:
; CHECK-NEXT:   .short 25
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 9:
; CHECK-NEXT:   .short 26
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 10:
; CHECK-NEXT:   .short 27
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 11:
; CHECK-NEXT:   .short 28
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 12: SP
; CHECK-NEXT:   .short 31
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 13:
; CHECK-NEXT:   .short 72
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 14:
; CHECK-NEXT:   .short 73
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 15:
; CHECK-NEXT:   .short 74
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 16:
; CHECK-NEXT:   .short 75
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 17:
; CHECK-NEXT:   .short 76
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 18:
; CHECK-NEXT:   .short 77
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 19:
; CHECK-NEXT:   .short 78
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; LiveOut Entry 20:
; CHECK-NEXT:   .short 79
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .byte 8
; Align
; CHECK-NEXT:   .p2align  3
  %1 = select i1 %c, i64 1, i64 2
  call anyregcc void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 32, ptr null, i32 0)
  ret i64 %1
}

declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
