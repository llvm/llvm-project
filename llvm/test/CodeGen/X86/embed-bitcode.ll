; RUN: llc -filetype=obj -mtriple=x86_64 %s -o %t
; RUN: llvm-readelf -S %t | FileCheck %s

; CHECK:      .text    PROGBITS 0000000000000000 [[#%x,OFF:]] 000000 00 AX 0
; CHECK-NEXT: .llvmbc  PROGBITS 0000000000000000 [[#%x,OFF:]] 000004 00    0
; CHECK-NEXT: .llvmcmd PROGBITS 0000000000000000 [[#%x,OFF:]] 000005 00    0

@llvm.embedded.module = private constant [4 x i8] c"BC\C0\DE", section ".llvmbc", align 1
@llvm.cmdline = private constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
@llvm.compiler.used = appending global [2 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @llvm.embedded.module, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @llvm.cmdline, i32 0, i32 0)], section "llvm.metadata"
