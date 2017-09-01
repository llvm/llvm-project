; RUN: llc -O0 -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | llvm-readobj -r -expand-relocs | FileCheck %s

; foo and bar are external and internal symbols.  a and b are pointers
; initialized to these locations offset by 2 and -2 elements respecitively.
@foo = external global i32, align 4
@bar = global i64 7, align 4
@a = global i32* getelementptr (i32, i32* @foo, i32 2), align 8
@b = global i64* getelementptr (i64, i64* @bar, i64 -2), align 8

; CHECK: Format: WASM
; CHECK: Relocations [
; CHECK:   Section (6) DATA {
; CHECK:     Relocation {
; CHECK:       Type: R_WEBASSEMBLY_MEMORY_ADDR_I32 (5)
; CHECK:       Offset: 0xE
; CHECK:       Index: 0x0
; CHECK:       Addend: 8
; CHECK:     }
; CHECK:     Relocation {
; CHECK:       Type: R_WEBASSEMBLY_MEMORY_ADDR_I32 (5)
; CHECK:       Offset: 0x16
; CHECK:       Index: 0x1
; CHECK:       Addend: -16
; CHECK:     }
; CHECK:   }
; CHECK: ]
