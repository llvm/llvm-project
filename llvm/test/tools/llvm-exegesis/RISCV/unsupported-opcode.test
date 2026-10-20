# RUN: llvm-exegesis -mtriple=riscv64-unknown-linux-gnu -mcpu=generic --benchmark-phase=assemble-measured-code -mode=inverse_throughput -opcode-name=InsnB 2>&1 | FileCheck %s

CHECK: Unsupported opcode: No Sched Class
