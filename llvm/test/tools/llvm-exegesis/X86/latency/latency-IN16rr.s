# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --skip-measurements -opcode-name=IN16rr -repetition-mode=duplicate | FileCheck %s

# FIXME: Sometimes fails with: 'unimplemented operand type'
# ALLOW_RETRIES: 2

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
