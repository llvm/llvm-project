; This test makes sure we can extract the instrumentation map from an
; XRay-instrumented object file.
;
; RUN: llvm-xray extract %S/Inputs/elf64-example.bin | FileCheck %s

; CHECK:      ---
; CHECK-NEXT: - { id: 1, address: 0x000000000041C900, function: 0x000000000041C900, kind: function-enter, always-instrument: true }
; CHECK-NEXT: - { id: 1, address: 0x000000000041C912, function: 0x000000000041C900, kind: function-exit, always-instrument: true }
; CHECK-NEXT: - { id: 2, address: 0x000000000041C930, function: 0x000000000041C930, kind: function-enter, always-instrument: true }
; CHECK-NEXT: - { id: 2, address: 0x000000000041C946, function: 0x000000000041C930, kind: function-exit, always-instrument: true }
; CHECK-NEXT: ...
