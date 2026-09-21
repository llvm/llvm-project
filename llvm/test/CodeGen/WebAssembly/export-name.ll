; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s
; RUN: llc < %s --filetype=obj | obj2yaml | FileCheck --check-prefix=OBJ %s

target triple = "wasm32-unknown-unknown"

@exported_g = addrspace(1) global i32 42 "wasm-export-name"="global_g"
@exported_mem = global i32 100 "wasm-export-name"="mem_g"

define void @test() "wasm-export-name"="foo" {
  ret void
}

declare void @test2() "wasm-export-name"="bar"

; CHECK: .export_name test, "foo"
; CHECK: .export_name test2, "bar"

; CHECK: .globaltype exported_g, i32
; CHECK: exported_g:
; CHECK-NEXT: .export_name exported_g, "global_g"

; CHECK: .export_name exported_mem, "mem_g"
; CHECK: exported_mem:
; CHECK-NEXT: .int32 100

; OBJ:      - Type:            EXPORT
; OBJ:        Exports:
; OBJ:          - Name:            foo
; OBJ-NEXT:       Kind:            FUNCTION
; OBJ-NEXT:       Index:           0
; OBJ:          - Name:            global_g
; OBJ-NEXT:       Kind:            GLOBAL
; OBJ-NEXT:       Index:           0

; OBJ:      - Type:            CUSTOM
; OBJ:        Name:            linking
; OBJ:        SymbolTable:
; OBJ:          - Index:           0
; OBJ:            Kind:            FUNCTION
; OBJ:            Name:            test
; OBJ:            Flags:           [ EXPORTED ]
; OBJ:          - Index:           1
; OBJ:            Kind:            GLOBAL
; OBJ:            Name:            exported_g
; OBJ:            Flags:           [ EXPORTED ]
; OBJ:          - Index:           2
; OBJ:            Kind:            DATA
; OBJ:            Name:            exported_mem
; OBJ:            Flags:           [ EXPORTED ]
