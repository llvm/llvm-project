# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:  | llvm-objdump -d - | FileCheck %s -check-prefix=CHECK-INSTR

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:  | llvm-readobj -r | FileCheck %s -check-prefix=CHECK-REL

# Check the assembler can handle hi and lo expressions with a constant 
# address, and constant expressions involving labels. Test case derived from 
# test/MC/Mips/hilo-addressing.s

# Check that 1 is added to the high 20 bits if bit 11 of the low part is 1.
.equ addr, 0xdeadbeef
  lui t0, %hi(addr)
  lw ra, %lo(addr)(t0)
# CHECK-INSTR: lui t0, 912092
# CHECK-INSTR: lw ra, -273(t0)

# Check that assembler can handle %hi(label1 - label2) and %lo(label1 - label2)
# expressions.

tmp1:
  # Emit zeros so that difference between tmp1 and tmp3 is 0x30124 bytes.
  .fill 0x30124-8
tmp2:
  lui t0, %hi(tmp3-tmp1)
  lw ra, %lo(tmp3-tmp1)(t0)
# CHECK-INSTR: lui t0, 48
# CHECK-INSTR: lw ra, 292(t0)

tmp3:
  lui t1, %hi(tmp2-tmp3)
  lw sp, %lo(tmp2-tmp3)(t1)
# CHECK-INSTR: lui t1, 0
# CHECK-INSTR: lw sp, -8(t1)

# Check that a relocation isn't emitted for %hi(label1 - label2) and
# %lo(label1 - label2) expressions.

# CHECK-REL-NOT: R_RISCV
