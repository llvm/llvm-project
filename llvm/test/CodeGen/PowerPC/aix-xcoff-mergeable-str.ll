; This file tests the codegen of mergeable strings in AIX assembly only.
; Once the codegen of mergeable strings for XCOFF object files is supported
; the test in this file should be merged into aix-xcoff-data.ll with additional
; tests for XCOFF object files.

; RUN: llc -verify-machineinstrs -mcpu=pwr4 \
; RUN:     -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 \
; RUN:     -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=CHECKOBJ %s

@magic16 = private unnamed_addr constant [4 x i16] [i16 264, i16 272, i16 213, i16 0], align 2
@magic32 =  private unnamed_addr constant [4 x i32] [i32 464, i32 472, i32 413, i32 0], align 4
@strA = private unnamed_addr constant [14 x i8] c"hello world!\0A\00", align 1
@.str = private unnamed_addr constant [9 x i8] c"abcdefgh\00", align 1
@p = global i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0), align 4

; Function Attrs: noinline nounwind optnone
define i8 @foo() #0 {
entry:
  %0 = load i8*, i8** @p, align 4
  %1 = load i8, i8* %0, align 1
  ret i8 %1
}

; CHECK:   .csect .rodata.str2.2[RO],2
; CHECK-NEXT:   .align  1
; CHECK-NEXT: .Lmagic16:
; CHECK-NEXT:   .vbyte	2, 264                     # 0x108
; CHECK-NEXT:   .vbyte	2, 272                     # 0x110
; CHECK-NEXT:   .vbyte	2, 213                     # 0xd5
; CHECK-NEXT:   .vbyte	2, 0                       # 0x0
; CHECK-NEXT:   .csect .rodata.str4.4[RO],2
; CHECK-NEXT:   .align  2
; CHECK-NEXT: .Lmagic32:
; CHECK-NEXT:   .vbyte	4, 464                     # 0x1d0
; CHECK-NEXT:   .vbyte	4, 472                     # 0x1d8
; CHECK-NEXT:   .vbyte	4, 413                     # 0x19d
; CHECK-NEXT:   .vbyte	4, 0                       # 0x0
; CHECK-NEXT:   .csect .rodata.str1.1[RO],2
; CHECK-NEXT: .LstrA:
; CHECK-NEXT: .byte   104
; CHECK-NEXT: .byte   101
; CHECK-NEXT: .byte   108
; CHECK-NEXT: .byte   108
; CHECK-NEXT: .byte   111
; CHECK-NEXT: .byte   32
; CHECK-NEXT: .byte   119
; CHECK-NEXT: .byte   111
; CHECK-NEXT: .byte   114
; CHECK-NEXT: .byte   108
; CHECK-NEXT: .byte   100
; CHECK-NEXT: .byte   33
; CHECK-NEXT: .byte   10
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .L.str:
; CHECK-NEXT: .byte   97
; CHECK-NEXT: .byte   98
; CHECK-NEXT: .byte   99
; CHECK-NEXT: .byte   100
; CHECK-NEXT: .byte   101
; CHECK-NEXT: .byte   102
; CHECK-NEXT: .byte   103
; CHECK-NEXT: .byte   104
; CHECK-NEXT: .byte   0

; CHECKOBJ:     00000010 <.rodata.str2.2>:
; CHECKOBJ-NEXT:       10: 01 08 01 10
; CHECKOBJ-NEXT:       14: 00 d5 00 00    {{.*}}{{[[:space:]] *}}
; CHECKOBJ-NEXT: 00000018 <.rodata.str4.4>:
; CHECKOBJ-NEXT:       18: 00 00 01 d0
; CHECKOBJ-NEXT:       1c: 00 00 01 d8
; CHECKOBJ-NEXT:       20: 00 00 01 9d
; CHECKOBJ-NEXT:       24: 00 00 00 00    {{.*}}{{[[:space:]] *}}
; CHECKOBJ-NEXT: 00000028 <.rodata.str1.1>:
; CHECKOBJ-NEXT:       28: 68 65 6c 6c
; CHECKOBJ-NEXT:       2c: 6f 20 77 6f
; CHECKOBJ-NEXT:       30: 72 6c 64 21
; CHECKOBJ-NEXT:       34: 0a 00 61 62
; CHECKOBJ-NEXT:       38: 63 64 65 66
; CHECKOBJ-NEXT:       3c: 67 68 00 00
