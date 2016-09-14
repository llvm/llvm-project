; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -enable-patchpoint-liveness=false | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx                                   | FileCheck -check-prefix=PATCH %s
;
; Note: Print verbose stackmaps using -debug-only=stackmaps.

; CHECK-LABEL:  .section  __LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT:   __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 2
; Num LargeConstants
; CHECK-NEXT:   .long   0
; Num Callsites
; CHECK-NEXT:   .long   5

; Functions and stack size
; CHECK-NEXT:   .quad _stackmap_liveness
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 3
; CHECK-NEXT:   .quad _mixed_liveness
; CHECK-NEXT:   .quad 8
; CHECK-NEXT:   .quad 2

define void @stackmap_liveness() {
entry:
  %a1 = call <2 x double> asm sideeffect "", "={xmm2}"() nounwind
; StackMap 1 (no liveness information available)
; CHECK-LABEL:  .long L{{.*}}-_stackmap_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; Padding
; CHECK-NEXT:   .short  0
; Num LiveOut Entries: 0
; CHECK-NEXT:   .short  0
; Align
; CHECK-NEXT:   .p2align  3

; StackMap 1 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_stackmap_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Padding
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 1
; PATCH-NEXT:   .short  1
; LiveOut Entry 1: %YMM2 (16 bytes) --> %XMM2
; PATCH-NEXT:   .short  19
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 16
; Align
; PATCH-NEXT:   .p2align  3
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 12, i8* null, i32 0)
  %a2 = call i64 asm sideeffect "", "={r8}"() nounwind
  %a3 = call i8 asm sideeffect "", "={ah}"() nounwind
  %a4 = call <4 x double> asm sideeffect "", "={ymm0}"() nounwind
  %a5 = call <4 x double> asm sideeffect "", "={ymm1}"() nounwind

; StackMap 2 (no liveness information available)
; CHECK-LABEL:  .long L{{.*}}-_stackmap_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; Padding
; CHECK-NEXT:   .short  0
; Num LiveOut Entries: 0
; CHECK-NEXT:   .short  0
; Align
; CHECK-NEXT:   .p2align  3

; StackMap 2 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_stackmap_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Padding
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 5
; PATCH-NEXT:   .short  5
; LiveOut Entry 1: %RAX (1 bytes) --> %AL or %AH
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 1
; LiveOut Entry 2: %R8 (8 bytes)
; PATCH-NEXT:   .short  8
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 8
; LiveOut Entry 3: %YMM0 (32 bytes)
; PATCH-NEXT:   .short  17
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 32
; LiveOut Entry 4: %YMM1 (32 bytes)
; PATCH-NEXT:   .short  18
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 32
; LiveOut Entry 5: %YMM2 (16 bytes) --> %XMM2
; PATCH-NEXT:   .short  19
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 16
; Align
; PATCH-NEXT:   .p2align  3
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 2, i32 12, i8* null, i32 0)
  call void asm sideeffect "", "{r8},{ah},{ymm0},{ymm1}"(i64 %a2, i8 %a3, <4 x double> %a4, <4 x double> %a5) nounwind

; StackMap 3 (no liveness information available)
; CHECK-LABEL:  .long L{{.*}}-_stackmap_liveness
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
; Padding
; CHECK-NEXT:   .short  0
; Num LiveOut Entries: 0
; CHECK-NEXT:   .short  0
; Align
; CHECK-NEXT:   .p2align  3

; StackMap 3 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_stackmap_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Padding
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 2
; PATCH-NEXT:   .short  2
; LiveOut Entry 1: %RSP (8 bytes)
; PATCH-NEXT:   .short  7
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 8
; LiveOut Entry 2: %YMM2 (16 bytes) --> %XMM2
; PATCH-NEXT:   .short  19
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 16
; Align
; PATCH-NEXT:   .p2align  3
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 3, i32 12, i8* null, i32 0)
  call void asm sideeffect "", "{xmm2}"(<2 x double> %a1) nounwind
  ret void
}

define void @mixed_liveness() {
entry:
  %a1 = call <2 x double> asm sideeffect "", "={xmm2}"() nounwind
; StackMap 4 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_mixed_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Padding
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 0
; PATCH-NEXT:   .short  0
; Align
; PATCH-NEXT:   .p2align  3

; StackMap 5 (patchpoint liveness information enabled)
; PATCH-LABEL:  .long L{{.*}}-_mixed_liveness
; PATCH-NEXT:   .short  0
; PATCH-NEXT:   .short  0
; Padding
; PATCH-NEXT:   .short  0
; Num LiveOut Entries: 2
; PATCH-NEXT:   .short  2
; LiveOut Entry 1: %RSP (8 bytes)
; PATCH-NEXT:   .short  7
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 8
; LiveOut Entry 2: %YMM2 (16 bytes) --> %XMM2
; PATCH-NEXT:   .short  19
; PATCH-NEXT:   .byte 0
; PATCH-NEXT:   .byte 16
; Align
; PATCH-NEXT:   .p2align  3
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 4, i32 5)
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 5, i32 0, i8* null, i32 0)
  call void asm sideeffect "", "{xmm2}"(<2 x double> %a1) nounwind
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
