# RUN: llvm-exegesis -mcpu=neoverse-v2 -mode=latency -opcode-name=FADDV_VPZ_D | FileCheck %s
# REQUIRES: aarch64-registered-target

# Check that warning of not initializing registers is not printed
# CHECK-NOT: setRegTo is not implemented, results will be unreliable

# Check that we add ret (bx lr) instr to snippet
# CHECK: assembled_snippet: {{.*}}C0035FD6