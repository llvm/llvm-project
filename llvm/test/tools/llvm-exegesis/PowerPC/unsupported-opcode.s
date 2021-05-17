# RUN: llvm-exegesis -mode=latency -opcode-name=SELECT_I8 2>&1 | FileCheck %s

CHECK: Unsupported opcode: isPseudo/usesCustomInserter
