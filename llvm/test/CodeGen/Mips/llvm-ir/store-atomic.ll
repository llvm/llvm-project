; RUN: llc -march=mips -mcpu=mips32r2 < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips -mcpu=mips32r6 < %s | FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips64 -mcpu=mips64r2 < %s | \
; RUN:    FileCheck %s -check-prefix=ALL -check-prefix=M64
; RUN: llc -march=mips64 -mcpu=mips64r6 < %s | \
; RUN:    FileCheck %s -check-prefix=ALL -check-prefix=M64

define void @store_i8(i8* %ptr, i8 signext %v) {
; ALL-LABEL: store_i8

; ALL: sync
; ALL: sb $5, 0($4)
  store atomic i8 %v, i8* %ptr release, align 1
  ret void
}

define void @store_i16(i16* %ptr, i16 signext %v) {
; ALL-LABEL: store_i16

; ALL: sync
; ALL: sh $5, 0($4)
  store atomic i16 %v, i16* %ptr release, align 2
  ret void
}

define void @store_i32(i32* %ptr, i32 signext %v) {
; ALL-LABEL: store_i32

; ALL: sync
; ALL: sw $5, 0($4)
  store atomic i32 %v, i32* %ptr release, align 4
  ret void
}

define void @store_i64(i64* %ptr, i64 %v) {
; M64-LABEL: store_i64

; M64: sync
; M64: sd $5, 0($4)
  store atomic i64 %v, i64* %ptr release, align 8
  ret void
}
