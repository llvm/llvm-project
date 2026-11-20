; RUN: llc < %s -mtriple=wasm32-unknown-unknown -filetype=obj | obj2yaml | FileCheck %s --check-prefix=OBJ --implicit-check-not=error:
; RUN: llc < %s -mtriple=wasm32-unknown-unknown | llvm-mc -triple=wasm32-unknown-unknown -filetype=obj | obj2yaml | FileCheck %s --check-prefix=OBJ
; RUN: llc < %s -mtriple=wasm32-unknown-unknown | FileCheck %s --check-prefix=ASM

; No function is defined here, so the AsmPrinter has no Subtarget. Emitting
; these globals must not crash, and each initializer must reach the Global
; section.

@i32v = addrspace(1) global i32 33554435
@i64v = addrspace(1) global i64 8589934595
@f32v = addrspace(1) global float 2.0
@f64v = addrspace(1) global double 2.0
@constv = addrspace(1) constant i32 42

; undef, poison and zero keep the type's default and are not diagnosed.
@undefv = addrspace(1) global i32 undef
@poisonv = addrspace(1) global i32 poison
@nullv = addrspace(1) global ptr null
@zagg = addrspace(1) global {i32} zeroinitializer
@znarrow = addrspace(1) global i8 0

; A single-element aggregate holds its element's value.
@agg = addrspace(1) global {i32} {i32 5}
@arr = addrspace(1) global [1 x i64] [i64 6]
@vec = addrspace(1) global <1 x float> <float 2.0>
@nest = addrspace(1) global {[1 x double]} {[1 x double] [double 2.0]}

; ASM:      .globaltype i32v, i32
; ASM-NEXT: .globalinit i32v, i32.const 33554435
; ASM:      .globaltype i64v, i64
; ASM-NEXT: .globalinit i64v, i64.const 8589934595
; ASM:      .globaltype f32v, f32
; ASM-NEXT: .globalinit f32v, f32.const 0x1p1
; ASM:      .globaltype f64v, f64
; ASM-NEXT: .globalinit f64v, f64.const 0x1p1
; ASM:      .globaltype constv, i32, immutable
; ASM-NEXT: .globalinit constv, i32.const 42
; A default value needs no .globalinit line.
; ASM:      .globaltype undefv, i32
; ASM-NOT:  .globalinit undefv
; ASM:      .globaltype agg, i32
; ASM-NEXT: .globalinit agg, i32.const 5
; ASM:      .globaltype arr, i64
; ASM-NEXT: .globalinit arr, i64.const 6
; ASM:      .globaltype vec, f32
; ASM-NEXT: .globalinit vec, f32.const 0x1p1
; ASM:      .globaltype nest, f64
; ASM-NEXT: .globalinit nest, f64.const 0x1p1

; OBJ:      - Type:            GLOBAL
; OBJ-NEXT:   Globals:
; OBJ-NEXT:     - Index:           0
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           33554435
; OBJ-NEXT:     - Index:           1
; OBJ-NEXT:       Type:            I64
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I64_CONST
; OBJ-NEXT:         Value:           8589934595
; OBJ-NEXT:     - Index:           2
; OBJ-NEXT:       Type:            F32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          F32_CONST
; OBJ-NEXT:         Value:           1073741824
; OBJ-NEXT:     - Index:           3
; OBJ-NEXT:       Type:            F64
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          F64_CONST
; OBJ-NEXT:         Value:           4611686018427387904
; OBJ-NEXT:     - Index:           4
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         false
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           42
; OBJ-NEXT:     - Index:           5
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           0
; OBJ-NEXT:     - Index:           6
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           0
; OBJ-NEXT:     - Index:           7
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           0
; OBJ-NEXT:     - Index:           8
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           0
; OBJ-NEXT:     - Index:           9
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           0
; OBJ-NEXT:     - Index:           10
; OBJ-NEXT:       Type:            I32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I32_CONST
; OBJ-NEXT:         Value:           5
; OBJ-NEXT:     - Index:           11
; OBJ-NEXT:       Type:            I64
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          I64_CONST
; OBJ-NEXT:         Value:           6
; OBJ-NEXT:     - Index:           12
; OBJ-NEXT:       Type:            F32
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          F32_CONST
; OBJ-NEXT:         Value:           1073741824
; OBJ-NEXT:     - Index:           13
; OBJ-NEXT:       Type:            F64
; OBJ-NEXT:       Mutable:         true
; OBJ-NEXT:       InitExpr:
; OBJ-NEXT:         Opcode:          F64_CONST
; OBJ-NEXT:         Value:           4611686018427387904
