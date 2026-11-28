# RUN: llvm-mc --verify -filetype=obj -triple=riscv32 -mattr=+relax %s -o /dev/null
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
  lui t0, %hi(tmp3-tmp1) # expected-error {{expected relocatable expression}}
  lw ra, %lo(tmp3-tmp1)(t0) # expected-error {{expected relocatable expression}}

tmp3:
  lui t1, %hi(tmp2-tmp3) # expected-error {{expected relocatable expression}}
  lw sp, %lo(tmp2-tmp3)(t1) # expected-error {{expected relocatable expression}}
