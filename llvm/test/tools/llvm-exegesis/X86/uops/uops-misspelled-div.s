# RUN: not llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mode=uops --skip-measurements -snippets-file=%s 2>&1 | FileCheck %s

# llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64-DEFREG CL 1
# llvm-exegesis -mtriple=x86_64-unknown-unknown -mcpu=x86-64-DEFREG AX 1
div8r cl

CHECK: error: invalid instruction mnemonic 'div8r'
CHECK: llvm-exegesis error: cannot parse asm file
