; RUN: llc < %s -mtriple=wasm32-wasip3 | FileCheck %s --check-prefix=WASIP3
; RUN: llc < %s -mtriple=wasm32-wasip3 -mattr=-component-model-thread-context | FileCheck %s --check-prefix=EXPLICIT-DISABLE
; RUN: llc < %s -mtriple=wasm32-wasip1 | FileCheck %s --check-prefix=WASIP1
; RUN: llc < %s -mtriple=wasm32-wasip2 | FileCheck %s --check-prefix=WASIP2

; Test that wasip3 target automatically enables component-model-thread-context

; WASIP3:        .section        .custom_section.target_features,"",@
; WASIP3-NEXT:   .int8   9
; WASIP3-NEXT:   .int8   43
; WASIP3-NEXT:   .int8   11
; WASIP3-NEXT:   .ascii  "bulk-memory"
; WASIP3-NEXT:   .int8   43
; WASIP3-NEXT:   .int8   15
; WASIP3-NEXT:   .ascii  "bulk-memory-opt"
; WASIP3-NEXT:   .int8   43
; WASIP3-NEXT:   .int8   22
; WASIP3-NEXT:   .ascii  "call-indirect-overlong"
; WASIP3-NEXT:   .int8   43
; WASIP3-NEXT:   .int8   30
; WASIP3-NEXT:   .ascii  "component-model-thread-context"

; EXPLICIT-DISABLE:        .section        .custom_section.target_features,"",@
; EXPLICIT-DISABLE-NEXT:   .int8   9
; EXPLICIT-DISABLE-NEXT:   .int8   43
; EXPLICIT-DISABLE-NEXT:   .int8   11
; EXPLICIT-DISABLE-NEXT:   .ascii  "bulk-memory"
; EXPLICIT-DISABLE-NEXT:   .int8   43
; EXPLICIT-DISABLE-NEXT:   .int8   15
; EXPLICIT-DISABLE-NEXT:   .ascii  "bulk-memory-opt"
; EXPLICIT-DISABLE-NEXT:   .int8   43
; EXPLICIT-DISABLE-NEXT:   .int8   22
; EXPLICIT-DISABLE-NEXT:   .ascii  "call-indirect-overlong"
; EXPLICIT-DISABLE-NEXT:   .int8   45
; EXPLICIT-DISABLE-NEXT:   .int8   30
; EXPLICIT-DISABLE-NEXT:   .ascii  "component-model-thread-context"

; WASIP1:        .section        .custom_section.target_features,"",@
; WASIP1-NEXT:   .int8   9
; WASIP1-NEXT:   .int8   43
; WASIP1-NEXT:   .int8   11
; WASIP1-NEXT:   .ascii  "bulk-memory"
; WASIP1-NEXT:   .int8   43
; WASIP1-NEXT:   .int8   15
; WASIP1-NEXT:   .ascii  "bulk-memory-opt"
; WASIP1-NEXT:   .int8   43
; WASIP1-NEXT:   .int8   22
; WASIP1-NEXT:   .ascii  "call-indirect-overlong"
; WASIP1-NEXT:   .int8   45
; WASIP1-NEXT:   .int8   30
; WASIP1-NEXT:   .ascii  "component-model-thread-context"

; WASIP2:        .section        .custom_section.target_features,"",@
; WASIP2-NEXT:   .int8   9
; WASIP2-NEXT:   .int8   43
; WASIP2-NEXT:   .int8   11
; WASIP2-NEXT:   .ascii  "bulk-memory"
; WASIP2-NEXT:   .int8   43
; WASIP2-NEXT:   .int8   15
; WASIP2-NEXT:   .ascii  "bulk-memory-opt"
; WASIP2-NEXT:   .int8   43
; WASIP2-NEXT:   .int8   22
; WASIP2-NEXT:   .ascii  "call-indirect-overlong"
; WASIP2-NEXT:   .int8   45
; WASIP2-NEXT:   .int8   30
; WASIP2-NEXT:   .ascii  "component-model-thread-context"

define void @test() {
  ret void
}