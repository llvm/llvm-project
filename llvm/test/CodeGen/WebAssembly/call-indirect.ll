; RUN: llc < %s -asm-verbose=false -O2 | FileCheck --check-prefixes=CHECK,NOREF %s
; RUN: llc < %s -asm-verbose=false -mattr=+reference-types -O2 | FileCheck --check-prefixes=CHECK,REF %s
; RUN: llc < %s -asm-verbose=false -O2 --filetype=obj | obj2yaml | FileCheck --check-prefix=OBJ %s

; Test that compilation units with call_indirect but without any
; function pointer declarations still get a table.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: call_indirect_void:
; CHECK-NEXT: .functype call_indirect_void (i32) -> ()
; CHECK-NEXT: local.get 0
; REF:        call_indirect __indirect_function_table, () -> ()
; NOREF:      call_indirect () -> ()
; CHECK-NEXT: end_function
define void @call_indirect_void(ptr %callee) {
  call void %callee()
  ret void
}

; CHECK-LABEL: call_indirect_alloca:
; CHECK-NEXT: .functype call_indirect_alloca () -> ()
; CHECK:      local.tee  0
; CHECK-NEXT: global.set  __stack_pointer
; CHECK-NEXT: local.get  0
; CHECK-NEXT: i32.const  12
; CHECK-NEXT: i32.add
; REF:        call_indirect __indirect_function_table, () -> ()
; NOREF:      call_indirect () -> ()
define void @call_indirect_alloca() {
entry:
  %ptr = alloca i32, align 4
  call void %ptr()
  ret void
}

; OBJ:    Imports:
; OBJ-NEXT:      - Module:          env
; OBJ-NEXT:        Field:           __linear_memory
; OBJ-NEXT:        Kind:            MEMORY
; OBJ-NEXT:        Memory:
; OBJ-NEXT:          Minimum:         0x0
; OBJ-NEXT:      - Module:          env
; OBJ-NEXT:        Field:           __stack_pointer
; OBJ-NEXT:        Kind:            GLOBAL
; OBJ-NEXT:        GlobalType:      I32
; OBJ-NEXT:        GlobalMutable:   true
; OBJ-NEXT:      - Module:          env
; OBJ-NEXT:        Field:           __indirect_function_table
; OBJ-NEXT:        Kind:            TABLE
