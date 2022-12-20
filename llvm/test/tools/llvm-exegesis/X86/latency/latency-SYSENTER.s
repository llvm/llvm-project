# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --skip-measurements -opcode-name=SYSENTER -repetition-mode=duplicate 2>&1 | FileCheck %s
# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=latency --skip-measurements -opcode-name=SYSENTER -repetition-mode=loop 2>&1 | FileCheck %s

CHECK: SYSENTER: unsupported opcode
