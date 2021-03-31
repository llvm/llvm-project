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
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I32 (23)
; CHECK-NEXT:       Offset: 0xA
; CHECK-NEXT:       Symbol: fizz
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LOCREL_I32 (23)
; CHECK-NEXT:       Offset: 0x17
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 4
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]

target triple = "wasm32-unknown-unknown"


; @foo - @bar
@foo = external global i32, align 4
@bar = constant i32 sub (
    i32 ptrtoint (i32* @foo to i32),
    i32 ptrtoint (i32* @bar to i32)
), section ".sec1"


; @foo - @addend + 4
@fizz = constant i32 42, align 4, section ".sec2"
@addend = constant i32 sub (
    i32 ptrtoint (i32* @foo to i32),
    i32 ptrtoint (i32* @fizz to i32)
), section ".sec2"

@x_sec = constant i32 sub (
    i32 ptrtoint (i32* @fizz to i32),
    i32 ptrtoint (i32* @x_sec to i32)
), section ".sec1"
