; RUN: llc < %s -asm-verbose=false | FileCheck --check-prefix=ASM %s
; RUN: llc < %s --filetype=obj | obj2yaml | FileCheck --check-prefix=OBJ %s

target triple = "wasm32-unknown-unknown"

@exported_g = addrspace(1) global i32 42 #0
@exported_mem = global i32 100 #1

; ASM: .globaltype exported_g, i32
; ASM: exported_g:
; ASM-NEXT: .export_name exported_g, "global_g"

; ASM: .export_name exported_mem, "mem_g"
; ASM: exported_mem:
; ASM-NEXT: .int32 100

; OBJ:      - Type:            EXPORT
; OBJ:        Exports:
; OBJ:          - Name:            global_g
; OBJ-NEXT:       Kind:            GLOBAL
; OBJ-NEXT:       Index:           0

; OBJ:      - Type:            CUSTOM
; OBJ:        Name:            linking
; OBJ:        SymbolTable:
; OBJ:          - Index:           0
; OBJ:            Kind:            GLOBAL
; OBJ:            Name:            exported_g
; OBJ:            Flags:           [ EXPORTED ]
; OBJ:          - Index:           1
; OBJ:            Kind:            DATA
; OBJ:            Name:            exported_mem
; OBJ:            Flags:           [ EXPORTED ]

attributes #0 = { "wasm-export-name"="global_g" }
attributes #1 = { "wasm-export-name"="mem_g" }

