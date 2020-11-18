; This test makes sure we can extract the instrumentation map from an
; XRay-instrumented PIE file.
;
; RUN: llvm-xray extract %S/Inputs/elf64-pie.bin -s | FileCheck %s

; CHECK:      ---
; CHECK-NEXT: - { id: 1, address: 0x00000000000299C0, function: 0x00000000000299C0, kind: function-enter, always-instrument: true, function-name: {{.*foo.*}} }
; CHECK-NEXT: - { id: 1, address: 0x00000000000299D0, function: 0x00000000000299C0, kind: function-exit, always-instrument: true, function-name: {{.*foo.*}} }
; CHECK-NEXT: - { id: 2, address: 0x00000000000299E0, function: 0x00000000000299E0, kind: function-enter, always-instrument: true, function-name: {{.*bar.*}} }
; CHECK-NEXT: - { id: 2, address: 0x00000000000299F6, function: 0x00000000000299E0, kind: function-exit, always-instrument: true, function-name: {{.*bar.*}} }
; CHECK-NEXT: ...
