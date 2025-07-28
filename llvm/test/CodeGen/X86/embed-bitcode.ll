; RUN: llc -filetype=obj -mtriple=x86_64 %s -o %t
; RUN: llvm-readelf -S %t | FileCheck %s
; RUN: llc -filetype=obj -mtriple=x86_64-pc-windows-msvc %s -o %t
; RUN: llvm-readobj -S %t | FileCheck %s --check-prefix=COFF

; CHECK:      .text    PROGBITS 0000000000000000 [[#%x,OFF:]] 000000 00 AX 0
; CHECK-NEXT: .llvmbc  PROGBITS 0000000000000000 [[#%x,OFF:]] 000004 00    0
; CHECK-NEXT: .llvmcmd PROGBITS 0000000000000000 [[#%x,OFF:]] 000005 00    0

; COFF:      Name: .llvmbc (2E 6C 6C 76 6D 62 63 00)
; COFF:      Characteristics [
; COFF-NEXT:   IMAGE_SCN_ALIGN_1BYTES
; COFF-NEXT:   IMAGE_SCN_MEM_DISCARDABLE
; COFF-NEXT: ]
; COFF:      Name: .llvmcmd (2E 6C 6C 76 6D 63 6D 64)
; COFF:      Characteristics [
; COFF-NEXT:   IMAGE_SCN_ALIGN_1BYTES
; COFF-NEXT:   IMAGE_SCN_MEM_DISCARDABLE
; COFF-NEXT: ]

@llvm.embedded.module = private constant [4 x i8] c"BC\C0\DE", section ".llvmbc", align 1
@llvm.cmdline = private constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
@llvm.compiler.used = appending global [2 x ptr] [ptr @llvm.embedded.module, ptr @llvm.cmdline], section "llvm.metadata"
