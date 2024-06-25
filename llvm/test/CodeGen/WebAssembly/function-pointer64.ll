; RUN: llc < %s -asm-verbose=false -O2 | FileCheck %s
; RUN: llc < %s -asm-verbose=false -mattr=+reference-types -O2 | FileCheck --check-prefix=REF %s
; RUN: llc < %s -asm-verbose=false -O2 --filetype=obj | obj2yaml | FileCheck --check-prefix=YAML %s

; This tests pointer features that may codegen differently in wasm64.

target triple = "wasm64-unknown-unknown"

define void @bar(i32 %n) {
entry:
  ret void
}

define void @foo(ptr %fp) {
entry:
  call void %fp(i32 1)
  ret void
}

define void @test() {
entry:
  call void @foo(ptr @bar)
  store ptr @bar, ptr @fptr
  ret void
}

@fptr = global ptr @bar

; For simplicity (and compatibility with UB C/C++ code) we keep all types
; of pointers the same size, so function pointers (which are 32-bit indices
; in Wasm) are represented as 64-bit until called.

; CHECK-LABEL: foo:
; CHECK:      .functype foo (i64) -> ()
; CHECK-NEXT: i32.const 1
; CHECK-NEXT: local.get 0
; CHECK-NEXT: call_indirect (i32) -> ()
; REF:        call_indirect __indirect_function_table, (i32) -> ()

; CHECK:      .functype test () -> ()
; CHECK-NEXT: i64.const bar
; CHECK-NEXT: call foo


; Check we're emitting a 64-bit relocs for the call_indirect, the
; `i64.const bar` reference in code, and the global.

; YAML:      Memory:
; YAML-NEXT:   Flags:   [ IS_64 ]
; YAML-NEXT:   Minimum: 0x1

; YAML:      - Type:   CODE
; YAML:      - Type:   R_WASM_TABLE_INDEX_SLEB64
; YAML-NEXT:   Index:  0
; YAML-NEXT:   Offset: 0x15
; YAML:      - Type:   R_WASM_TABLE_INDEX_SLEB64
; YAML-NEXT:   Index:  0
; YAML-NEXT:   Offset: 0x28

; YAML:      - Type:   DATA
; YAML:      - Type:   R_WASM_TABLE_INDEX_I64
; YAML-NEXT:   Index:  0
; YAML-NEXT:   Offset: 0x6
