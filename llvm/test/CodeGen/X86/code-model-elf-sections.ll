; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=small -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -large-data-threshold=79 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -large-data-threshold=80 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -large-data-threshold=79 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -large-data-threshold=80 -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL

; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=small -data-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=SMALL-DS
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=medium -data-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE-DS
; RUN: llc < %s -relocation-model=pic -filetype=obj -code-model=large -data-sections -o %t
; RUN: llvm-readelf -S %t | FileCheck %s --check-prefix=LARGE-DS

; SMALL: .data {{.*}} WA {{.*}}
; SMALL: .data.x {{.*}} WA {{.*}}
; SMALL: .data0 {{.*}} WA {{.*}}
; SMALL: .ldata {{.*}} WAl {{.*}}
; SMALL: .ldata.x {{.*}} WAl {{.*}}
; SMALL: .ldata0 {{.*}} WA {{.*}}
; SMALL: force_small {{.*}} WA {{.*}}
; SMALL: force_large {{.*}} WAl {{.*}}
; SMALL: foo {{.*}} WA {{.*}}
; SMALL: .bss {{.*}} WA {{.*}}
; SMALL: .lbss {{.*}} WAl {{.*}}
; SMALL: .rodata {{.*}} A {{.*}}
; SMALL: .lrodata {{.*}} Al {{.*}}
; SMALL: .data.rel.ro {{.*}} WA {{.*}}
; SMALL: .tbss {{.*}} WAT {{.*}}
; SMALL: .tdata {{.*}} WAT {{.*}}

; SMALL-DS: .data {{.*}} WA {{.*}}
; SMALL-DS: .data.x {{.*}} WA {{.*}}
; SMALL-DS: .data0 {{.*}} WA {{.*}}
; SMALL-DS: .ldata {{.*}} WAl {{.*}}
; SMALL-DS: .ldata.x {{.*}} WAl {{.*}}
; SMALL-DS: .ldata0 {{.*}} WA {{.*}}
; SMALL-DS: .data.data {{.*}} WA {{.*}}
; SMALL-DS: force_small {{.*}} WA {{.*}}
; SMALL-DS: force_large {{.*}} WAl {{.*}}
; SMALL-DS: foo {{.*}} WA {{.*}}
; SMALL-DS: .lbss {{.*}} WAl {{.*}}
; SMALL-DS: .bss.bss {{.*}} WA {{.*}}
; SMALL-DS: .lrodata {{.*}} Al {{.*}}
; SMALL-DS: .rodata.rodata {{.*}} A {{.*}}
; SMALL-DS: .data.rel.ro.relro {{.*}} WA {{.*}}
; SMALL-DS: .tbss.tbss {{.*}} WAT {{.*}}
; SMALL-DS: .tdata.tdata {{.*}} WAT {{.*}}

; LARGE: .data {{.*}} WA {{.*}}
; LARGE: .data.x {{.*}} WA {{.*}}
; LARGE: .data0 {{.*}} WAl {{.*}}
; LARGE: .ldata {{.*}} WAl {{.*}}
; LARGE: .ldata.x {{.*}} WAl {{.*}}
; LARGE: .ldata0 {{.*}} WAl {{.*}}
; LARGE: force_small {{.*}} WA {{.*}}
; LARGE: force_large {{.*}} WAl {{.*}}
; LARGE: foo {{.*}} WAl {{.*}}
; LARGE: .bss {{.*}} WA {{.*}}
; LARGE: .lbss {{.*}} WAl {{.*}}
; LARGE: .rodata {{.*}} A {{.*}}
; LARGE: .lrodata {{.*}} Al {{.*}}
; LARGE: .ldata.rel.ro {{.*}} WAl {{.*}}
; LARGE: .tbss {{.*}} WAT {{.*}}
; LARGE: .tdata {{.*}} WAT {{.*}}

; LARGE-DS: .data {{.*}} WA {{.*}}
; LARGE-DS: .data.x {{.*}} WA {{.*}}
; LARGE-DS: .data0 {{.*}} WAl {{.*}}
; LARGE-DS: .ldata {{.*}} WAl {{.*}}
; LARGE-DS: .ldata.x {{.*}} WAl {{.*}}
; LARGE-DS: .ldata0 {{.*}} WAl {{.*}}
; LARGE-DS: .ldata.data {{.*}} WAl {{.*}}
; LARGE-DS: force_small {{.*}} WA {{.*}}
; LARGE-DS: force_large {{.*}} WAl {{.*}}
; LARGE-DS: foo {{.*}} WAl {{.*}}
; LARGE-DS: .bss {{.*}} WA {{.*}}
; LARGE-DS: .lbss.bss {{.*}} WAl {{.*}}
; LARGE-DS: .rodata {{.*}} A {{.*}}
; LARGE-DS: .lrodata.rodata {{.*}} Al {{.*}}
; LARGE-DS: .ldata.rel.ro.relro {{.*}} WAl {{.*}}
; LARGE-DS: .tbss.tbss {{.*}} WAT {{.*}}
; LARGE-DS: .tdata.tdata {{.*}} WAT {{.*}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

@data_with_explicit_section = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], section ".data"
@data_with_explicit_section2 = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], section ".data.x"
@data_with_explicit_section0 = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], section ".data0"
@ldata_with_explicit_section = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], section ".ldata"
@ldata_with_explicit_section2 = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], section ".ldata.x"
@ldata_with_explicit_section0 = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], section ".ldata0"
@data = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0]
@data_force_small = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], code_model "small", section "force_small"
@data_force_large = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], code_model "large", section "force_large"
@data_force_small_ldata = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], code_model "small", section ".ldata"
@data_force_large_data = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], code_model "large", section ".data"
@foo_with_explicit_section = internal global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0], section "foo"
@bss_with_explicit_section = internal global [10 x i64] zeroinitializer, section ".bss"
@lbss_with_explicit_section = internal global [10 x i64] zeroinitializer, section ".lbss"
@bss = internal global [10 x i64] zeroinitializer
@rodata_with_explicit_section = internal constant [10 x i64] zeroinitializer, section ".rodata"
@lrodata_with_explicit_section = internal constant [10 x i64] zeroinitializer, section ".lrodata"
@rodata = internal constant [10 x i64] zeroinitializer
@relro = internal constant [10 x ptr] [ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func, ptr @func]
@tbss = internal thread_local global [10 x i64] zeroinitializer
@tdata = internal thread_local global [10 x i64] [i64 1, i64 2, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0]

declare void @func()
