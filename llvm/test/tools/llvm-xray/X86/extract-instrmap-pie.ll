; This test makes sure we can extract the instrumentation map from an
; XRay-instrumented PIE file.
;
; RUN: llvm-xray extract %S/Inputs/elf64-pie.bin -s | FileCheck %s

; CHECK:      ---
; CHECK-NEXT: - { id: 1, address: 0x299C0, function: 0x299C0, kind: function-enter, always-instrument: true, function-name: {{.*foo.*}} }
; CHECK-NEXT: - { id: 1, address: 0x299D0, function: 0x299C0, kind: function-exit, always-instrument: true, function-name: {{.*foo.*}} }
; CHECK-NEXT: - { id: 2, address: 0x299E0, function: 0x299E0, kind: function-enter, always-instrument: true, function-name: {{.*bar.*}} }
; CHECK-NEXT: - { id: 2, address: 0x299F6, function: 0x299E0, kind: function-exit, always-instrument: true, function-name: {{.*bar.*}} }
; CHECK-NEXT: ...
