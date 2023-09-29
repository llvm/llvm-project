; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=small -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -large-data-threshold=79 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -large-data-threshold=80 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL

; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=small -data-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL-DS
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -data-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE-DS
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -data-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL-DS

; SMALL: .data {{.*}} WA {{.*}}
; SMALL: .bss {{.*}} WA {{.*}}
; SMALL: .rodata {{.*}} A {{.*}}
; SMALL: .data.rel.ro {{.*}} WA {{.*}}

; SMALL-DS: .data.data {{.*}} WA {{.*}}
; SMALL-DS: .bss.bss {{.*}} WA {{.*}}
; SMALL-DS: .rodata.rodata {{.*}} A {{.*}}
; SMALL-DS: .data.rel.ro.relro {{.*}} WA {{.*}}

; LARGE: .ldata {{.*}} WAl {{.*}}
; LARGE: .lbss {{.*}} WAl {{.*}}
; LARGE: .lrodata {{.*}} Al {{.*}}
; LARGE: .ldata.rel.ro {{.*}} WAl {{.*}}

; LARGE-DS: .ldata.data {{.*}} WAl {{.*}}
; LARGE-DS: .lbss.bss {{.*}} WAl {{.*}}
; LARGE-DS: .lrodata.rodata {{.*}} Al {{.*}}
; LARGE-DS: .ldata.rel.ro.relro {{.*}} WAl {{.*}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

@data = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0]
@bss = internal global [10 x i64] zeroinitializer
@rodata = internal constant [10 x i64] zeroinitializer
@relro = internal constant [10 x ptr] [ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func]

declare void @func()
