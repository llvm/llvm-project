; RUN: llc -mtriple=mips -mcpu=mips32r2 < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -mtriple=mips -mcpu=mips32r6 < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 < %s | \
; RUN:    FileCheck %s -check-prefixes=ALL,M64
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 < %s | \
; RUN:    FileCheck %s -check-prefixes=ALL,M64

define i8 @load_i8(ptr %ptr) {
; ALL-LABEL: load_i8

; ALL: lb $2, 0($4)
; ALL: sync
  %val = load atomic i8, ptr %ptr acquire, align 1
  ret i8 %val
}

define i16 @load_i16(ptr %ptr) {
; ALL-LABEL: load_i16

; ALL: lh $2, 0($4)
; ALL: sync
  %val = load atomic i16, ptr %ptr acquire, align 2
  ret i16 %val
}

define i32 @load_i32(ptr %ptr) {
; ALL-LABEL: load_i32

; ALL: lw $2, 0($4)
; ALL: sync
  %val = load atomic i32, ptr %ptr acquire, align 4
  ret i32 %val
}

define i64 @load_i64(ptr %ptr) {
; M64-LABEL: load_i64

; M64: ld $2, 0($4)
; M64: sync
  %val = load atomic i64, ptr %ptr acquire, align 8
  ret i64 %val
}
