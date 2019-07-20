; RUN: llc -relocation-model=pic -data-sections -o - %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@hidden = external hidden global i8
@default = external global i8

; CHECK: .section .rodata.rodata
; CHECK: rodata:
; CHECK: .long hidden-rodata
@rodata = hidden constant i32 trunc (i64 sub (i64 ptrtoint (i8* @hidden to i64), i64 ptrtoint (i32* @rodata to i64)) to i32)

; CHECK: .section .data.rel.ro.relro1
; CHECK: relro1:
; CHECK: .long default-relro1
@relro1 = hidden constant i32 trunc (i64 sub (i64 ptrtoint (i8* @default to i64), i64 ptrtoint (i32* @relro1 to i64)) to i32)

; CHECK: .section .data.rel.ro.relro2
; CHECK: relro2:
; CHECK: .long hidden-relro2
@relro2 = constant i32 trunc (i64 sub (i64 ptrtoint (i8* @hidden to i64), i64 ptrtoint (i32* @relro2 to i64)) to i32)
