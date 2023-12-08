# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-INSTR

# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:   | llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-REL

# Check the assembler can handle hi and lo expressions with a constant
# address. Test case derived from test/MC/Mips/hilo-addressing.s

# Check that 1 is added to the high 20 bits if bit 11 of the low part is 1.
.equ addr, 0xdeadbeef
  lui t0, %hi(addr)
  lw ra, %lo(addr)(t0)
# CHECK-INSTR: lui t0, 912092
# CHECK-INSTR: lw ra, -273(t0)

# CHECK-REL-NOT: R_RISCV
