# RUN: not llvm-mc %s -triple=riscv32 -mattr=+experimental-zicfiss,+zcmop,+c -M no-aliases -show-encoding \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ERR %s
# RUN: not llvm-mc %s -triple=riscv64 -mattr=+experimental-zicfiss,+zcmop,+c -M no-aliases -show-encoding \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ERR %s

# CHECK-ERR: error: register must be ra or t0 (x1 or x5)
sspopchk a1

# CHECK-ERR: error: register must be ra (x1)
c.sspush t0

# CHECK-ERR: error: register must be t0 (x5)
c.sspopchk ra

# CHECK-ERR: error: register must be ra or t0 (x1 or x5)
sspush a0

# CHECK-ERR: error: invalid operand for instruction
ssrdp zero
