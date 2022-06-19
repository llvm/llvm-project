# RUN: not llvm-exegesis -mode=uops -snippets-file=%s 2>&1 | FileCheck %s

# LLVM-EXEGESIS-DEFREG CL 1
# LLVM-EXEGESIS-DEFREG AX 1
div8r cl

CHECK: error: invalid instruction mnemonic 'div8r'
CHECK: llvm-exegesis error: cannot parse asm file
