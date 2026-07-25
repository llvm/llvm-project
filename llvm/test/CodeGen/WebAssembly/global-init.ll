; RUN: llc < %s -mtriple=wasm32-unknown-unknown -filetype=obj | obj2yaml | FileCheck %s

; No function is defined here, so the AsmPrinter's Subtarget is null: emitting
; these address-space-1 globals must not crash, and each initializer must reach
; the Global section instead of the type's default 0.

@i32v = addrspace(1) global i32 33554435
@i64v = addrspace(1) global i64 8589934595
@f32v = addrspace(1) global float 2.0
@f64v = addrspace(1) global double 2.0
@constv = addrspace(1) constant i32 42

; CHECK:      - Type:            GLOBAL
; CHECK-NEXT:   Globals:
; CHECK-NEXT:     - Index:           0
; CHECK-NEXT:       Type:            I32
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I32_CONST
; CHECK-NEXT:         Value:           33554435
; CHECK-NEXT:     - Index:           1
; CHECK-NEXT:       Type:            I64
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I64_CONST
; CHECK-NEXT:         Value:           8589934595
; CHECK-NEXT:     - Index:           2
; CHECK-NEXT:       Type:            F32
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          F32_CONST
; CHECK-NEXT:         Value:           1073741824
; CHECK-NEXT:     - Index:           3
; CHECK-NEXT:       Type:            F64
; CHECK-NEXT:       Mutable:         true
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          F64_CONST
; CHECK-NEXT:         Value:           4611686018427387904
; CHECK-NEXT:     - Index:           4
; CHECK-NEXT:       Type:            I32
; CHECK-NEXT:       Mutable:         false
; CHECK-NEXT:       InitExpr:
; CHECK-NEXT:         Opcode:          I32_CONST
; CHECK-NEXT:         Value:           42
