; RUN: llc -mtriple=arm64-apple-macosx %s -o - | FileCheck %s

; CHECK-LABEL: .section __MY_DATA,__my_data
; CHECK:       .globl _data
; CHECK: _data:
; CHECK:       .long 42
@data = global i32 42 #0

; CHECK-LABEL: .section __MY_BSS,__my_bss
; CHECK:       .globl _bss
; CHECK: _bss:
; CHECK:       .long 0
@bss = global i32 0 #0

; CHECK-LABEL: .section __MY_RODATA,__my_rodata
; CHECK:       .globl _const
; CHECK: _const:
; CHECK:       .long 42
@const = constant i32 42 #0

; CHECK-LABEL: .section __MY_RELRO,__my_relro
; CHECK:       .globl _vars_relro
; CHECK: _vars_relro:
; CHECK:       .quad _data
; CHECK:       .quad _bss
@vars_relro = hidden constant [2 x ptr] [ptr @data, ptr @bss], align 16 #0

attributes #0 = { "data-section"="__MY_DATA,__my_data" "bss-section"="__MY_BSS,__my_bss" "rodata-section"="__MY_RODATA,__my_rodata" "relro-section"="__MY_RELRO,__my_relro" }
