# RUN: not llvm-exegesis -mtriple=riscv64-unknown-linux-gnu -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADD 2>&1 | FileCheck %s
# REQUIRES: riscv-registered-target

# CHECK: llvm-exegesis error: A CPU must be explicitly specified when cross compiling. To see all possible options for riscv64-unknown-linux-gnu triple use -mcpu=help
