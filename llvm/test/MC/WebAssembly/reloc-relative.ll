; RUN: llc -O0 -filetype=obj %s -o - | llvm-readobj -r --expand-relocs - | FileCheck %s

; CHECK:      Format: WASM
; CHECK:      Relocations [
; CHECK-NEXT:   Section (3) DATA {
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I32 (23)
; CHECK-NEXT:       Offset: 0x6
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I64 (27)
; CHECK-NEXT:       Offset: 0xE
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I32 (23)
; CHECK-NEXT:       Offset: 0x16
; CHECK-NEXT:       Symbol: fizz
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I64 (27)
; CHECK-NEXT:       Offset: 0x1E
; CHECK-NEXT:       Symbol: fizz
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I32 (23)
; CHECK-NEXT:       Offset: 0x2F
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 4
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I64 (27)
; CHECK-NEXT:       Offset: 0x33
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 8
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]

target triple = "wasm32-unknown-unknown"

; @foo - @bar
@foo = external global i32, align 4
@bar = constant i32 sub (
    i32 ptrtoint (ptr @foo to i32),
    i32 ptrtoint (ptr @bar to i32)
), section ".sec1"

@bar64 = constant i64 sub (
    i64 ptrtoint (ptr @foo to i64),
    i64 ptrtoint (ptr @bar64 to i64)
), section ".sec1"


; @foo - @addend + 4
@fizz = constant i32 42, align 4, section ".sec2"
@addend = constant i32 sub (
    i32 ptrtoint (ptr @foo to i32),
    i32 ptrtoint (ptr @fizz to i32)
), section ".sec2"

@addend64 = constant i64 sub (
    i64 ptrtoint (ptr @foo to i64),
    i64 ptrtoint (ptr @fizz to i64)
), section ".sec2"

@x_sec = constant i32 sub (
    i32 ptrtoint (ptr @fizz to i32),
    i32 ptrtoint (ptr @x_sec to i32)
), section ".sec1"

@x_sec64 = constant i64 sub (
    i64 ptrtoint (ptr @fizz to i64),
    i64 ptrtoint (ptr @x_sec64 to i64)
), section ".sec1"
