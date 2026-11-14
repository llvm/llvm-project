## Library options such as -print-after-isel reach the target machine.
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --benchmark-phase=assemble-measured-code -opcode-name=ADD64rr -print-after-isel 2>&1 >/dev/null | FileCheck %s

# CHECK: # MachineFunctionGenerator::assemble:
