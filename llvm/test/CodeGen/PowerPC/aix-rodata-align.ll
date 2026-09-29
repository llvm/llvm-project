; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr7 \
; RUN:     -mattr=+altivec -data-sections=false < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr7 \
; RUN:     -mattr=+altivec -data-sections=false < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr7 \
; RUN:     -mattr=+altivec -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=OBJ %s

define <16 x i8> @test() {
entry:
  ret <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>
}
; CHECK:        .csect .rodata.16[RO],4
; CHECK-NEXT:   .align  4
; CHECK-NEXT: L..CPI0_0:
; CHECK-NEXT:   .byte   0                               # 0x0
; CHECK-NEXT:   .byte   1                               # 0x1
; CHECK-NEXT:   .byte   2                               # 0x2
; CHECK-NEXT:   .byte   3                               # 0x3
; CHECK-NEXT:   .byte   4                               # 0x4
; CHECK-NEXT:   .byte   5                               # 0x5
; CHECK-NEXT:   .byte   6                               # 0x6
; CHECK-NEXT:   .byte   7                               # 0x7
; CHECK-NEXT:   .byte   8                               # 0x8
; CHECK-NEXT:   .byte   9                               # 0x9
; CHECK-NEXT:   .byte   10                              # 0xa
; CHECK-NEXT:   .byte   11                              # 0xb
; CHECK-NEXT:   .byte   12                              # 0xc
; CHECK-NEXT:   .byte   13                              # 0xd
; CHECK-NEXT:   .byte   14                              # 0xe
; CHECK-NEXT:   .byte   15                              # 0xf

; OBJ-LABEL: <.rodata.16>:
; OBJ-NEXT:    00 01 02 03
; OBJ-NEXT:    04 05 06 07
; OBJ-NEXT:    08 09 0a 0b
; OBJ-NEXT:    0c 0d 0e 0f
