; RUN: not llc < %s -mtriple=x86_64-linux-gnu 2>&1 | FileCheck %s

; A trunc in a global initializer is emitted as the operand expression and
; the assembler truncates it to the emitted size, a whole number of bytes.
; For a type that is not byte-sized that would leave the padding bits set,
; so it is reported as unsupported.

@a = global [130 x i8] zeroinitializer
@b = global [2 x i8] zeroinitializer

; CHECK: error: unsupported expression in static initializer: trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr @a to i64)) to i1)
@g1 = global i1 trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr @a to i64)) to i1)

; CHECK: error: unsupported expression in static initializer: trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr @a to i64)) to i7)
@g7 = global i7 trunc (i64 sub (i64 ptrtoint (ptr @b to i64), i64 ptrtoint (ptr @a to i64)) to i7)
