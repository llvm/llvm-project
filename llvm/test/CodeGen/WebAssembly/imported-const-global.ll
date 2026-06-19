; RUN: llc < %s --mtriple=wasm32-unknown-unknown -asm-verbose=false | FileCheck %s --check-prefix=ASM
; RUN: llc < %s --mtriple=wasm32-unknown-unknown --filetype=obj | obj2yaml | FileCheck %s --check-prefix=OBJ

@imported_g = external addrspace(1) constant i32

define i32 @goo() {
; ASM-LABEL: goo:
; ASM-NEXT:  functype       goo () -> (i32)
; ASM-NEXT:  global.get imported_g
; ASM-NEXT:  end_function
  %v = load i32, ptr addrspace(1) @imported_g
  ret i32 %v
}

; ASM:       .globaltype imported_g, i32, immutable

; OBJ:       --- !WASM
; OBJ:       FileHeader:
; OBJ:         Version:         0x1
; OBJ:       Sections:
; OBJ:         - Type:            TYPE
; OBJ:         - Type:            IMPORT
; OBJ:           Imports:
; OBJ:             - Module:          env
; OBJ:               Field:           imported_g
; OBJ:               Kind:            GLOBAL
; OBJ:               GlobalType:      I32
; OBJ:               GlobalMutable:   false