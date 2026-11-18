; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu -filetype=obj | llvm-objdump -s -j .data - | FileCheck %s --check-prefix=OBJ

; A trunc in a global initializer is emitted as the operand expression and
; the assembler truncates it to the emitted size. For a type that is not
; byte-sized the expression has to be masked to the type's width, otherwise
; the padding bits of the byte are not zero.
; b - a = 130 (0x82).

@a = global [130 x i8] zeroinitializer
@b = global [2 x i8] zeroinitializer

; CHECK-LABEL: g1:
; CHECK-NEXT:    .byte (b-a)&1
@g1 = global i1 trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr @a to i64)) to i1)

; CHECK-LABEL: g7:
; CHECK-NEXT:    .byte (b-a)&127
@g7 = global i7 trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr @a to i64)) to i7)

; Byte-sized types are left to the assembler.
; CHECK-LABEL: g8:
; CHECK-NEXT:    .byte b-a
@g8 = global i8 trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr @a to i64)) to i8)

; OBJ:      Contents of section .data:
; OBJ-NEXT: 0000 000282
