# RUN: llvm-exegesis --opcode-name=ADDv16i8 --mcpu=cortex-a78 --mtriple=aarch64-linux-gnu \
# RUN:               --mode=latency --benchmark-phase=assemble-measured-code 2>&1 | FileCheck %s

# Check that we add ret (bx lr) instr to snippet
# CHECK: assembled_snippet: {{.*}}C0035FD6
