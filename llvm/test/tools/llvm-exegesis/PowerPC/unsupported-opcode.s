# RUN: llvm-exegesis -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 --skip-measurements -mode=latency -opcode-name=SELECT_I8 2>&1 | FileCheck %s

CHECK: Unsupported opcode: isPseudo/usesCustomInserter
