; RUN: llc -filetype=obj -mattr=-bulk-memory,-bulk-memory-opt %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

define hidden void @call_memcpy(ptr align 4 %a, ptr align 4 %b) {
entry:
  tail call void @llvm.memcpy.p0.p0.i32(ptr align 4 %a, ptr align 4 %b, i32 512, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1)

; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x1
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            TYPE
; CHECK-NEXT:     Signatures:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:           - I32
; CHECK-NEXT:         ReturnTypes:     []
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         ParamTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:           - I32
; CHECK-NEXT:           - I32
; CHECK-NEXT:         ReturnTypes:
; CHECK-NEXT:           - I32
; CHECK-NEXT:   - Type:            IMPORT
