; RUN: llc -mtriple x86_64-linux-gnu < %s | FileCheck %s
@a = internal unnamed_addr constant [1 x [1 x i32]] zeroinitializer, section ".init.rodata", align 4
; CHECK: .init.rodata,"aM",{{[@%]}}progbits,4
