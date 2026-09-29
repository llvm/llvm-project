# RUN: not llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o /dev/null 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s | llvm-objdump -d - | FileCheck %s --check-prefix=NORELAX

# Check the assembler rejects hi and lo expressions with constant expressions
# involving labels when diff expressions are emitted as relocation pairs.
# Test case derived from test/MC/Mips/hilo-addressing.s

# NORELAX:      lui t0, 0x0
# NORELAX-NEXT: lw ra, 0x8(t0)
# NORELAX:      lui t1, 0x0
# NORELAX-NEXT: lw sp, -0x8(t1)

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
