; RUN: llc < %s -asm-verbose=false | FileCheck --check-prefix=ASM %s
; RUN: llc < %s --filetype=obj | obj2yaml | FileCheck --check-prefix=OBJ %s

target triple = "wasm32-unknown-unknown"

@exported_g = addrspace(1) global i32 42, !wasm.export.name !0

; ASM: .globaltype exported_g, i32
; ASM: exported_g:
; ASM-NEXT: .export_name exported_g, "global_g"

; OBJ:      - Type:            EXPORT
; OBJ:        Exports:
; OBJ:          - Name:            global_g
; OBJ-NEXT:       Kind:            GLOBAL
; OBJ-NEXT:       Index:           0

!0 = !{!"global_g"}
