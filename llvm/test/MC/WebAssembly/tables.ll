; RUN: llc --mtriple=wasm32-unknown-unknown -filetype=obj %s -o - | obj2yaml | FileCheck --check-prefixes CHECK %s
; RUN: llc --mtriple=wasm64-unknown-unknown -filetype=obj %s -o - | obj2yaml | FileCheck --check-prefixes CHECK,WASM64 %s

; Ensure that tables generated through `llc` are in the correct address mode


%externref = type ptr addrspace(10)
@externref_table = local_unnamed_addr addrspace(1) global [0 x %externref] undef

%funcref = type ptr addrspace(20)
@funcref_table = local_unnamed_addr addrspace(1) global [0 x %funcref] undef

; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x1
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Memory:
; WASM64-NEXT:          Flags:           [ IS_64 ]
; CHECK-NEXT:           Minimum:         0x0
; CHECK-NEXT:   - Type:            TABLE
; CHECK-NEXT:     Tables:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ElemType:        EXTERNREF
; CHECK-NEXT:         Limits:
; WASM64-NEXT:          Flags:           [ IS_64 ]
; CHECK-NEXT:           Minimum:         0x0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ElemType:        FUNCREF
; CHECK-NEXT:         Limits:
; WASM64-NEXT:          Flags:           [ IS_64 ]
; CHECK-NEXT:           Minimum:         0x0
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Name:            externref_table
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:         Table:           0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Name:            funcref_table
; CHECK-NEXT:        Flags:           [  ]
; CHECK-NEXT:         Table:           1
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            target_features
; CHECK-NEXT:     Features:
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            bulk-memory
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            bulk-memory-opt
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            call-indirect-overlong
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            multivalue
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            mutable-globals
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            nontrapping-fptoint
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            reference-types
; CHECK-NEXT:       - Prefix:          USED
; CHECK-NEXT:         Name:            sign-ext
; WASM64-NEXT:      - Prefix:          USED
; WASM64-NEXT:        Name:            memory64
; CHECK-NEXT: ...
