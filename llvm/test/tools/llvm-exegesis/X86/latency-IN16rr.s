# RUN: llvm-exegesis -mode=latency -opcode-name=IN16rr -repetition-mode=duplicate | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     - 'IN16rr'
CHECK-NEXT:     - {{.*}}
CHECK-NEXT: config: ''
CHECK-NEXT: register_initial_values:
CHECK-DAG: - '[[REG1:[A-Z0-9]+]]=0x0'
CHECK-LAST: ...
