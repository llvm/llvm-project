## Tested invalid statement start tokens. X86 supports "{". Use a different target.
# REQUIRES: aarch64-registered-target

# RUN: not llvm-mc -triple=aarch64 %s 2>&1 | FileCheck %s

# CHECK: [[#@LINE+1]]:2: error: unexpected token at start of statement
 {insn}
