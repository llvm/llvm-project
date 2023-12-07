# RUN: llvm-exegesis -mode=latency -opcode-name=ADD | FileCheck %s
# REQUIRES: exegesis-can-execute-mips, exegesis-can-measure-latency

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADD
CHECK-NEXT: config: ''
CHECK-NEXT: register_initial_values:
CHECK-DAG: - '[[REG1:[A-Z0-9]+]]=0x0'
CHECK-LAST: ...
