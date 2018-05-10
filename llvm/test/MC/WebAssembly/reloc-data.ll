; RUN: llc -O0 -filetype=obj %s -o - | llvm-readobj -r -expand-relocs | FileCheck %s

target triple = "wasm32-unknown-unknown"

; foo and bar are external and internal symbols.  a and b are pointers
; initialized to these locations offset by 2 and -2 elements respecitively.

@foo = external global i32, align 4
@bar = global i64 7, align 4
@a = global i32* getelementptr (i32, i32* @foo, i32 2), align 8
@b = global i64* getelementptr (i64, i64* @bar, i64 -2), align 8
@c = global [3 x i32*] [i32* @foo, i32* @foo, i32* @foo], align 16

; CHECK:      Format: WASM
; CHECK:      Relocations [
; CHECK-NEXT:   Section (2) DATA {
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WEBASSEMBLY_MEMORY_ADDR_I32 (5)
; CHECK-NEXT:       Offset: 0x13
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 8
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WEBASSEMBLY_MEMORY_ADDR_I32 (5)
; CHECK-NEXT:       Offset: 0x1C
; CHECK-NEXT:       Symbol: bar
; CHECK-NEXT:       Addend: -16
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WEBASSEMBLY_MEMORY_ADDR_I32 (5)
; CHECK-NEXT:       Offset: 0x25
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WEBASSEMBLY_MEMORY_ADDR_I32 (5)
; CHECK-NEXT:       Offset: 0x29
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WEBASSEMBLY_MEMORY_ADDR_I32 (5)
; CHECK-NEXT:       Offset: 0x2D
; CHECK-NEXT:       Symbol: foo
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]
