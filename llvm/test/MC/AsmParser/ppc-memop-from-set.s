# RUN: llvm-mc -triple ppc64-unknown-elf %s | FileCheck %s
.set SP,1
std 3, -8(SP)

# CHECK: SP = 1
# CHECK: std 3, -8(1)
