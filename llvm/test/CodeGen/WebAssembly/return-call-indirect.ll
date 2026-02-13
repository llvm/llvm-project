; RUN: llc < %s -asm-verbose=false -mattr=-reference-types,+tail-call -O2 | FileCheck --check-prefixes=CHECK,NOREF %s
; RUN: llc < %s -asm-verbose=false -mattr=+reference-types,+tail-call -O2 | FileCheck --check-prefixes=CHECK,REF %s
; RUN: llc < %s -asm-verbose=false -mattr=+tail-call -O2 --filetype=obj | obj2yaml | FileCheck --check-prefix=OBJ %s

; Test that compilation units with return_call_indirect but without any
; function pointer declarations still get a table.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: return_call_indirect:
; CHECK-NEXT: .functype return_call_indirect (i32) -> (i32)
; CHECK-NEXT: local.get 0
; REF:       return_call_indirect     __indirect_function_table, () -> (i32)
; NOREF:     return_call_indirect     () -> (i32)
; CHECK-NEXT: end_function
define i32 @return_call_indirect(ptr %callee) {
  %r = tail call i32 %callee()
  ret i32 %r
}

; OBJ:    Imports:
; OBJ-NEXT:      - Module:          env
; OBJ-NEXT:        Field:           __linear_memory
; OBJ-NEXT:        Kind:            MEMORY
; OBJ-NEXT:        Memory:
; OBJ-NEXT:          Minimum:         0x0
; OBJ-NEXT:      - Module:          env
; OBJ-NEXT:        Field:           __indirect_function_table
; OBJ-NEXT:        Kind:            TABLE
