; This test makes sure we can extract the instrumentation map from an
; XRay-instrumented object file.
;
; RUN: llvm-xray extract %S/Inputs/instr-map-mach.o -s | FileCheck %s

; CHECK:      ---
; CHECK-NEXT: - { id: 1, address: 0x0, function: 0x0, kind: function-enter, always-instrument: true, function-name: 'task(void*)' }
; CHECK-NEXT: - { id: 1, address: 0x162, function: 0x0, kind: function-exit, always-instrument: true, function-name: 'task(void*)' }
; CHECK-NEXT: ...
