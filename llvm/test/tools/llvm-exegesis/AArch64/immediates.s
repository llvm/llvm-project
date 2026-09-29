# RUN: llvm-exegesis --mtriple=aarch64 --mcpu=neoverse-v2 --mode=latency --opcode-name=EXTRWrri  --benchmark-phase=prepare-and-assemble-snippet | FileCheck %s

# CHECK: 'EXTRWrri {{W.*}} {{W.*}} {{W.*}} i_0x1'
