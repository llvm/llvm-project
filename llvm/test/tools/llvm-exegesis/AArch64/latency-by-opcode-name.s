# RUN: llvm-exegesis -mode=latency -opcode-name=ADCXr | FileCheck %s
# REQUIRES: exegesis-can-execute-aarch64, exegesis-can-measure-latency

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     - 'ADCXr [[REG1:X[0-9]+|LR]] [[REG2:X[0-9]+|LR]] [[REG3:X[0-9]+|LR]]'
CHECK-NEXT: config: ''
CHECK-NEXT: register_initial_values:
CHECK-DAG: - '[[REG2]]=0x0'
CHECK-DAG: - '[[REG3]]=0x0'
CHECK-DAG: - 'NZCV=0x0'
CHECK-DAG: ...
