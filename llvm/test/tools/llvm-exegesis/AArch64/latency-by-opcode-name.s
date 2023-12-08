# RUN: llvm-exegesis -mode=latency -opcode-name=ADDXrr | FileCheck %s
# REQUIRES: exegesis-can-execute-aarch64, exegesis-can-measure-latency

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     ADDXrr [[REG1:X[0-9]+|LR]] [[REG2:X[0-9]+|LR]] [[REG3:X[0-9]+|LR]]
CHECK-NEXT: config: ''
CHECK-NEXT: register_initial_values:
CHECK-DAG: - '[[REG2]]=0x0'
# We don't check REG3 because in the case that REG2=REG3 the check would fail
CHECK-LAST: ...
