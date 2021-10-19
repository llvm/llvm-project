; RUN: llc -filetype=obj %s -o - | llvm-readobj -r --expand-relocs - | FileCheck %s
; RUN: llc -filetype=obj -mattr=+reference-types %s -o - | llvm-readobj -r --expand-relocs - | FileCheck --check-prefix=REF %s

target triple = "wasm32-unknown-unknown"

; Pointers to functions of two different types
@a = global i64 ()* inttoptr (i64 5 to i64 ()*), align 8
@b = global i32 ()* inttoptr (i32 7 to i32 ()*), align 8

; External functions
declare i32 @c()
declare i32 @d()

define i32 @f1() {
entry:
    %aa = load i64 ()*, i64 ()** @a, align 8
    %bb = load i32 ()*, i32 ()** @b, align 8
    %tmp1 = call i64 %aa()
    %tmp2 = call i32 %bb()
    %tmp3 = call i32 @c()
    %tmp4 = call i32 @d()
    ret i32 %tmp2
}


; CHECK: Format: WASM
; CHECK: Relocations [
; CHECK-NEXT:   Section (5) CODE {
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
; CHECK-NEXT:       Offset: 0x9
; CHECK-NEXT:       Symbol: b
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
; CHECK-NEXT:       Offset: 0x14
; CHECK-NEXT:       Symbol: a
; CHECK-NEXT:       Addend: 0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
; CHECK-NEXT:       Offset: 0x1A
; CHECK-NEXT:       Index: 0x1
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
; CHECK-NEXT:       Offset: 0x24
; CHECK-NEXT:       Index: 0x0
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
; CHECK-NEXT:       Offset: 0x2D
; CHECK-NEXT:       Symbol: c
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
; CHECK-NEXT:       Offset: 0x34
; CHECK-NEXT:       Symbol: d
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]

; REF: Format: WASM
; REF: Relocations [
; REF-NEXT:   Section (5) CODE {
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
; REF-NEXT:       Offset: 0x9
; REF-NEXT:       Symbol: b
; REF-NEXT:       Addend: 0
; REF-NEXT:     }
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_MEMORY_ADDR_LEB (3)
; REF-NEXT:       Offset: 0x14
; REF-NEXT:       Symbol: a
; REF-NEXT:       Addend: 0
; REF-NEXT:     }
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
; REF-NEXT:       Offset: 0x1A
; REF-NEXT:       Index: 0x1
; REF-NEXT:     }
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_TABLE_NUMBER_LEB (20)
; REF-NEXT:       Offset: 0x1F
; REF-NEXT:       Symbol: __indirect_function_table
; REF-NEXT:     }
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_TYPE_INDEX_LEB (6)
; REF-NEXT:       Offset: 0x28
; REF-NEXT:       Index: 0x0
; REF-NEXT:     }
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_TABLE_NUMBER_LEB (20)
; REF-NEXT:       Offset: 0x2D
; REF-NEXT:       Symbol: __indirect_function_table
; REF-NEXT:     }
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
; REF-NEXT:       Offset: 0x35
; REF-NEXT:       Symbol: c
; REF-NEXT:     }
; REF-NEXT:     Relocation {
; REF-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
; REF-NEXT:       Offset: 0x3C
; REF-NEXT:       Symbol: d
; REF-NEXT:     }
; REF-NEXT:   }
; REF-NEXT: ]
