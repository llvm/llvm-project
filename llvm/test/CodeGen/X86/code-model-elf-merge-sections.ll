; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -large-data-threshold=0 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -large-data-threshold=99 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL

; LARGE: .lrodata.str4.4 {{.*}} AMSl
; LARGE: .lrodata.cst8 {{.*}} AMl

; SMALL: .rodata.str4.4 {{.*}} AMS
; SMALL: .rodata.cst8 {{.*}} AM

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

@str = internal unnamed_addr constant [3 x i32] [i32 1, i32 2, i32 0]
@merge = internal unnamed_addr constant i64 2
