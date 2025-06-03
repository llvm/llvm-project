# RUN: llvm-exegesis -mtriple=riscv64-unknown-linux-gnu --mcpu=generic -mode=latency --benchmark-phase=assemble-measured-code -mattr=+d -opcode-name=FADD_D | FileCheck %s

CHECK:      ---
CHECK-NEXT: mode: latency
CHECK-NEXT: key:
CHECK-NEXT:   instructions:
CHECK-NEXT:     - 'FADD_D [[REG1:F[0-9]+_D]] [[REG2:F[0-9]+_D]] [[REG3:F[0-9]+_D]] i_0x7'
CHECK-NEXT: config: ''
CHECK-NEXT: register_initial_values:
CHECK-DAG: - '[[REG1]]=0x0'
CHECK-DAG: ...
