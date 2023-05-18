
; RUN: not --crash llc < %s -mtriple=xcore -code-model=medium 2>&1 | FileCheck %s -check-prefix=BAD_CM
; RUN: not --crash llc < %s -mtriple=xcore -code-model=kernel 2>&1 | FileCheck %s -check-prefix=BAD_CM
; RUN: not --crash llc < %s -mtriple=xcore -code-model=tiny 2>&1 | FileCheck %s -check-prefix=BAD_CM
; BAD_CM: Target only supports CodeModel Small or Large


; RUN: llc < %s -mtriple=xcore | FileCheck %s
; RUN: llc < %s -mtriple=xcore -code-model=small | FileCheck %s
; RUN: llc < %s -mtriple=xcore -code-model=large | FileCheck %s -check-prefix=LARGE


; CHECK-LABEL: test:
; CHECK: zext r0, 1
; CHECK: bt r0, [[JUMP:.LBB[0-9_]*]]
; CHECK: ldaw r0, dp[A2]
; CHECK: retsp 0
; CHECK: [[JUMP]]
; CHECK: ldaw r0, dp[A1]
; CHECK: retsp 0
; LARGE-LABEL: test:
; LARGE: zext r0, 1
; LARGE: ldaw r11, cp[.LCPI{{[0-9_]*}}]
; LARGE: mov r1, r11
; LARGE: ldaw r11, cp[.LCPI{{[0-9_]*}}]
; LARGE: bt r0, [[JUMP:.LBB[0-9_]*]]
; LARGE: mov r11, r1
; LARGE: [[JUMP]]
; LARGE: ldw r0, r11[0]
; LARGE: retsp 0
@A1 = external dso_local global [50000 x i32]
@A2 = external dso_local global [50000 x i32]
define ptr @test(i1 %bool) nounwind {
entry:
  %Addr = select i1 %bool, ptr @A1, ptr @A2
  ret ptr %Addr
}


; CHECK: .section  .cp.rodata.cst4,"aMc",@progbits,4
; CHECK: .long 65536
; CHECK: .text
; CHECK-LABEL: f:
; CHECK: ldc r1, 65532
; CHECK: add r1, r0, r1
; CHECK: ldw r1, r1[0]
; CHECK: ldw r2, cp[.LCPI{{[0-9_]*}}]
; CHECK: add r0, r0, r2
; CHECK: ldw r0, r0[0]
; CHECK: add r0, r1, r0
; CHECK: ldw r1, dp[l]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[l+4]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[l+392]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[l+396]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[s]
; CHECK: add r0, r0, r1
; CHECK: ldw r1, dp[s+36]
; CHECK: add r0, r0, r1
; CHECK: retsp 0
;
; LARGE: .section .cp.rodata.cst4,"aMc",@progbits,4
; LARGE: .long 65536
; LARGE: .section .cp.rodata,"ac",@progbits
; LARGE: .long l
; LARGE: .long l+4
; LARGE: .long l+392
; LARGE: .long l+396
; LARGE: .text
; LARGE-LABEL: f:
; LARGE: ldc r1, 65532
; LARGE: add r1, r0, r1
; LARGE: ldw r1, r1[0]
; LARGE: ldw r2, cp[.LCPI{{[0-9_]*}}]
; LARGE: add r0, r0, r2
; LARGE: ldw r0, r0[0]
; LARGE: add r0, r1, r0
; LARGE: ldw r1, cp[.LCPI{{[0-9_]*}}]
; LARGE: ldw r1, r1[0]
; LARGE: add r0, r0, r1
; LARGE: ldw r1, cp[.LCPI{{[0-9_]*}}]
; LARGE: ldw r1, r1[0]
; LARGE: add r0, r0, r1
; LARGE: ldw r1, cp[.LCPI{{[0-9_]*}}]
; LARGE: ldw r1, r1[0]
; LARGE: add r0, r0, r1
; LARGE: ldw r1, cp[.LCPI{{[0-9_]*}}]
; LARGE: ldw r1, r1[0]
; LARGE: add r0, r0, r1
; LARGE: ldw r1, dp[s]
; LARGE: add r0, r0, r1
; LARGE: ldw r1, dp[s+36]
; LARGE: add r0, r0, r1
; LARGE: retsp 0
define dso_local i32 @f(ptr %i) {
entry:
  %0 = getelementptr inbounds i32, ptr %i, i32 16383
  %1 = load i32, ptr %0
  %2 = getelementptr inbounds i32, ptr %i, i32 16384
  %3 = load i32, ptr %2
  %4 = add nsw i32 %1, %3
  %5 = load i32, ptr @l
  %6 = add nsw i32 %4, %5
  %7 = load i32, ptr getelementptr inbounds ([100 x i32], ptr @l, i32 0, i32 1)
  %8 = add nsw i32 %6, %7
  %9 = load i32, ptr getelementptr inbounds ([100 x i32], ptr @l, i32 0, i32 98)
  %10 = add nsw i32 %8, %9
  %11 = load i32, ptr getelementptr inbounds ([100 x i32], ptr @l, i32 0, i32 99)
  %12 = add nsw i32 %10, %11
  %13 = load i32, ptr @s
  %14 = add nsw i32 %12, %13
  %15 = load i32, ptr getelementptr inbounds ([10 x i32], ptr @s, i32 0, i32 9)
  %16 = add nsw i32 %14, %15
  ret i32 %16
}


; CHECK-LABEL: UnknownSize:
; CHECK: ldw r0, dp[NoSize+40]
; CHECK-NEXT: retsp 0
;
; LARGE: .section .cp.rodata,"ac",@progbits
; LARGE: .LCPI{{[0-9_]*}}
; LARGE-NEXT: .long NoSize
; LARGE-NEXT: .text
; LARGE-LABEL: UnknownSize:
; LARGE: ldw r0, cp[.LCPI{{[0-9_]*}}]
; LARGE-NEXT: ldw r0, r0[0]
; LARGE-NEXT: retsp 0
@NoSize = external dso_local global [0 x i32]
define dso_local i32 @UnknownSize() nounwind {
entry:
  %0 = load i32, ptr getelementptr inbounds ([0 x i32], ptr @NoSize, i32 0, i32 10)
  ret i32 %0
}


; CHECK-LABEL: UnknownStruct:
; CHECK: ldaw r0, dp[Unknown]
; CHECK-NEXT: retsp 0
;
; LARGE: .section .cp.rodata,"ac",@progbits
; LARGE: .LCPI{{[0-9_]*}}
; LARGE-NEXT: .long Unknown
; LARGE-NEXT: .text
; LARGE-LABEL: UnknownStruct:
; LARGE: ldw r0, cp[.LCPI{{[0-9_]*}}]
; LARGE-NEXT: retsp 0
%Struct = type opaque
@Unknown = external dso_local global %Struct
define ptr @UnknownStruct() nounwind {
entry:
  ret ptr @Unknown
}


; CHECK: .section .dp.bss,"awd",@nobits
; CHECK-LABEL: l:
; CHECK: .space 400
; LARGE: .section  .dp.bss.large,"awd",@nobits
; LARGE-LABEL: l:
; LARGE: .space  400
@l = dso_local global [100 x i32] zeroinitializer

; CHECK-LABEL: s:
; CHECK: .space 40
; LARGE: .section  .dp.bss,"awd",@nobits
; LARGE-LABEL: s:
; LARGE: .space  40
@s = dso_local global [10 x i32] zeroinitializer

; CHECK: .section .dp.rodata,"awd",@progbits
; CHECK-LABEL: cl:
; CHECK: .space 400
; LARGE: .section .dp.rodata.large,"awd",@progbits
; LARGE-LABEL: cl:
; LARGE: .space 400
@cl = dso_local constant  [100 x i32] zeroinitializer

; CHECK-LABEL: cs:
; CHECK: .space 40
; LARGE: .section .dp.rodata,"awd",@progbits
; LARGE-LABEL: cs:
; LARGE: .space 40
@cs = dso_local constant  [10 x i32] zeroinitializer

; CHECK: .section .cp.rodata,"ac",@progbits
; CHECK-LABEL: icl:
; CHECK: .space 400
; LARGE: .section .cp.rodata.large,"ac",@progbits
; LARGE-LABEL: icl:
; LARGE: .space 400
@icl = internal constant  [100 x i32] zeroinitializer

; CHECK-LABEL: cs:
; CHECK: .space 40
; LARGE: .section .cp.rodata,"ac",@progbits
; LARGE-LABEL: cs:
; LARGE: .space 40
@ics = internal constant  [10 x i32] zeroinitializer

; CHECK: .section  .cp.namedsection,"ac",@progbits
; CHECK-LABEL: cpsec:
; CHECK: .long 0
@cpsec = dso_local constant i32 0, section ".cp.namedsection"

; CHECK: .section  .dp.namedsection,"awd",@progbits
; CHECK-LABEL: dpsec:
; CHECK: .long 0
@dpsec = dso_local global i32 0, section ".dp.namedsection"

