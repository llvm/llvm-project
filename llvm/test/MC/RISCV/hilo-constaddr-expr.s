# RUN: not llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s 2>&1 \
# RUN:     | FileCheck %s -check-prefix=CHECK-RELAX
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s 2>&1 \
# RUN:     | llvm-objdump -d - | FileCheck %s -check-prefix=CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s 2>&1 \
# RUN:     | llvm-objdump -r - | FileCheck %s -check-prefix=CHECK-REL

# Check the assembler rejects hi and lo expressions with constant expressions
# involving labels when diff expressions are emitted as relocation pairs.
# Test case derived from test/MC/Mips/hilo-addressing.s

tmp1:
  # Emit zeros so that difference between tmp1 and tmp3 is 0x30124 bytes.
  .fill 0x30124-8
tmp2:
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lui t0, %hi(tmp3-tmp1)
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lw ra, %lo(tmp3-tmp1)(t0)
# CHECK-INSTR: lui t0, 48
# CHECK-INSTR: lw ra, 292(t0)

tmp3:
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lui t1, %hi(tmp2-tmp3)
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lw sp, %lo(tmp2-tmp3)(t1)
# CHECK-INSTR: lui t1, 0
# CHECK-INSTR: lw sp, -8(t1)

# CHECK-REL-NOT: R_RISCV