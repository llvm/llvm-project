# RUN: not llvm-mc -triple ppc64-unknown-elf %s 2>&1 | FileCheck %s
.set SP,r1
.set SP2,%r1
std 3, -8(SP)
std 3, -8(SP2)

# CHECK: unknown token in expression
# CHECK: missing expression
# CHECK: identifier expands to invalid register number
# CHECK: identifier expands to invalid register number
