; RUN: llc -relocation-model=pic -data-sections -o - %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@hidden = external hidden global i8
@default = external global i8

; CHECK: .section .rodata.rodata
; CHECK: rodata:
; CHECK: .long hidden-rodata
@rodata = hidden constant i32 trunc (i64 sub (i64 ptrtoint (ptr @hidden to i64), i64 ptrtoint (ptr @rodata to i64)) to i32)

; CHECK: .section .rodata.rodata_ptrtoaddr
; CHECK: rodata_ptrtoaddr:
; CHECK: .long hidden-rodata_ptrtoaddr
@rodata_ptrtoaddr = hidden constant i32 trunc (i64 sub (i64 ptrtoaddr (ptr @hidden to i64), i64 ptrtoaddr (ptr @rodata_ptrtoaddr to i64)) to i32)

; CHECK: .section .data.rel.ro.relro1
; CHECK: relro1:
; CHECK: .long default-relro1
@relro1 = hidden constant i32 trunc (i64 sub (i64 ptrtoint (ptr @default to i64), i64 ptrtoint (ptr @relro1 to i64)) to i32)

; CHECK: .section .data.rel.ro.relro1_ptrtoaddr
; CHECK: relro1_ptrtoaddr:
; CHECK: .long default-relro1_ptrtoaddr
@relro1_ptrtoaddr = hidden constant i32 trunc (i64 sub (i64 ptrtoaddr (ptr @default to i64), i64 ptrtoaddr (ptr @relro1_ptrtoaddr to i64)) to i32)

; CHECK: .section .data.rel.ro.relro2
; CHECK: relro2:
; CHECK: .long hidden-relro2
@relro2 = constant i32 trunc (i64 sub (i64 ptrtoint (ptr @hidden to i64), i64 ptrtoint (ptr @relro2 to i64)) to i32)

; CHECK: .section .data.rel.ro.relro2_ptrtoaddr
; CHECK: relro2_ptrtoaddr:
; CHECK: .long hidden-relro2_ptrtoaddr
@relro2_ptrtoaddr = constant i32 trunc (i64 sub (i64 ptrtoaddr (ptr @hidden to i64), i64 ptrtoaddr (ptr @relro2_ptrtoaddr to i64)) to i32)

; CHECK:      .section .rodata.obj
; CHECK-NEXT: .globl obj
; CHECK:      obj:
; CHECK:      .long 0
; CHECK:      .long hidden_func@PLT-obj-4

declare hidden void @hidden_func()

; Ensure that inbound GEPs with constant offsets are also resolved.
@obj = dso_local unnamed_addr constant { { i32, i32 } } {
  { i32, i32 } {
    i32 0,
    i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @hidden_func to i64), i64 ptrtoint (ptr getelementptr inbounds ({ { i32, i32 } }, ptr @obj, i32 0, i32 0, i32 1) to i64)) to i32)
  } }, align 4

; CHECK:      .section .rodata.rodata2
; CHECK-NEXT: .globl rodata2
; CHECK:      rodata2:
; CHECK:      .long extern_func@PLT-rodata2

declare void @extern_func()

@rodata2 = dso_local constant i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @extern_func to i64), i64 ptrtoint (ptr @rodata2 to i64)) to i32)
