; This file tests the codegen of mergeable strings in AIX assembly only.
; Once the codegen of mergeable strings for XCOFF object files is supported
; the test in this file should be merged into aix-xcoff-data.ll with additional
; tests for XCOFF object files.

; RUN: llc -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 \
; RUN:     -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=CHECKOBJ %s

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

; CHECK:      .csect .rodata.str1.1[RO]
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

; CHECKOBJ: 00000010 .rodata.str1.1:
; CHECKOBJ-NEXT:       10: 68 65 6c 6c                   xori 5, 3, 27756
; CHECKOBJ-NEXT:       14: 6f 20 77 6f                   xoris 0, 25, 30575
; CHECKOBJ-NEXT:       18: 72 6c 64 21                   andi. 12, 19, 25633
; CHECKOBJ-NEXT:       1c: 0a 00 61 62                   tdlti   0, 24930{{[[:space:]] *}}
; CHECKOBJ-NEXT: 0000001e .L.str:
; CHECKOBJ-NEXT:       1e: 61 62 63 64                   ori 2, 11, 25444
; CHECKOBJ-NEXT:       22: 65 66 67 68                   oris 6, 11, 26472
; CHECKOBJ-NEXT:       26: 00                            <unknown>
; CHECKOBJ-NEXT:       27: 00                            <unknown>
