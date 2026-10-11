; RUN: llc < %s -mtriple=wasm32-unknown-unknown -mattr=+reference-types -filetype=obj | obj2yaml | FileCheck %s --implicit-check-not=error:

; undef and zeroinitializer both give an externref global ref.null, with no
; diagnostic.

@undefref = addrspace(1) global target("wasm.externref") undef
@nullref = addrspace(1) global target("wasm.externref") zeroinitializer

; CHECK:      - Type:            GLOBAL
; CHECK-NEXT:   Globals:
; CHECK-NEXT:     - Index:           0
; CHECK-NEXT:       Type:            EXTERNREF
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          REF_NULL
; CHECK-NEXT:         Type:            EXTERNREF
; CHECK-NEXT:     - Index:           1
; CHECK-NEXT:       Type:            EXTERNREF
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          REF_NULL
; CHECK-NEXT:         Type:            EXTERNREF
