; This test makes sure we can extract the instrumentation map from an
; XRay-instrumented object file.
;
; RUN: llvm-xray extract %S/Inputs/elf64-example.bin | FileCheck %s

; CHECK:      ---
; CHECK-NEXT: - { id: 1, address: 0x41C900, function: 0x41C900, kind: function-enter, always-instrument: true{{.*}} }
; CHECK-NEXT: - { id: 1, address: 0x41C912, function: 0x41C900, kind: function-exit, always-instrument: true{{.*}} }
; CHECK-NEXT: - { id: 2, address: 0x41C930, function: 0x41C930, kind: function-enter, always-instrument: true{{.*}} }
; CHECK-NEXT: - { id: 2, address: 0x41C946, function: 0x41C930, kind: function-exit, always-instrument: true{{.*}} }
; CHECK-NEXT: ...
