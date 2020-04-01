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

; CHECK:      .section .rodata.cst8
; CHECK-NEXT: .globl obj
; CHECK:      obj:
; CHECK:      .long 0
; CHECK:      .long (hidden_func-obj)-4

declare hidden void @hidden_func()

; Ensure that inbound GEPs with constant offsets are also resolved.
@obj = dso_local unnamed_addr constant { { i32, i32 } } {
  { i32, i32 } {
    i32 0,
    i32 trunc (i64 sub (i64 ptrtoint (void ()* dso_local_equivalent @hidden_func to i64), i64 ptrtoint (i32* getelementptr inbounds ({ { i32, i32 } }, { { i32, i32 } }* @obj, i32 0, i32 0, i32 1) to i64)) to i32)
  } }, align 4

; CHECK:      .section .rodata.rodata2
; CHECK-NEXT: .globl rodata2
; CHECK:      rodata2:
; CHECK:      .long extern_func@PLT-rodata2

declare void @extern_func()

@rodata2 = dso_local constant i32 trunc (i64 sub (i64 ptrtoint (void ()* dso_local_equivalent @extern_func to i64), i64 ptrtoint (i32* @rodata2 to i64)) to i32)
