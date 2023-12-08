; This tests that we can extract the instrumentation map and symbolize the
; function addresses.
; RUN: llvm-xray extract %S/Inputs/elf64-example.bin -s | FileCheck %s

; CHECK:      ---
; CHECK-NEXT: - { id: 1, address: 0x41C900, function: 0x41C900, kind: function-enter, always-instrument: true, function-name: {{.*foo.*}} }
; CHECK-NEXT: - { id: 1, address: 0x41C912, function: 0x41C900, kind: function-exit, always-instrument: true, function-name: {{.*foo.*}}  }
; CHECK-NEXT: - { id: 2, address: 0x41C930, function: 0x41C930, kind: function-enter, always-instrument: true, function-name: {{.*bar.*}}  }
; CHECK-NEXT: - { id: 2, address: 0x41C946, function: 0x41C930, kind: function-exit, always-instrument: true, function-name: {{.*bar.*}}  }
; CHECK-NEXT: ...
