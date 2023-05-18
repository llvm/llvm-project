# RUN: not llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s -o /dev/null 2>&1 | FileCheck %s

# Check the assembler rejects hi and lo expressions with constant expressions
# involving labels when diff expressions are emitted as relocation pairs.
# Test case derived from test/MC/Mips/hilo-addressing.s

tmp1:
tmp2:
# CHECK: :[[#@LINE+1]]:[[#]]: error: expected relocatable expression
  lui t0, %hi(tmp3-tmp1)
# CHECK: :[[#@LINE+1]]:[[#]]: error: expected relocatable expression
  lw ra, %lo(tmp3-tmp1)(t0)

tmp3:
# CHECK: :[[#@LINE+1]]:[[#]]: error: expected relocatable expression
  lui t1, %hi(tmp2-tmp3)
# CHECK: :[[#@LINE+1]]:[[#]]: error: expected relocatable expression
  lw sp, %lo(tmp2-tmp3)(t1)
