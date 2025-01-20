# RUN: not llvm-mc %s -triple=riscv32 -mattr=+experimental-zicfiss,+c -M no-aliases -show-encoding \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ERR %s

# CHECK-ERR: error: invalid operand for instruction
sspopchk a1

# CHECK-ERR: error: invalid operand for instruction
c.sspush t0

# CHECK-ERR: error: invalid operand for instruction
c.sspopchk ra

# CHECK-ERR: error: invalid operand for instruction
sspush a0

# CHECK-ERR: error: invalid operand for instruction
ssrdp zero
