# RUN: llvm-mc %s -triple=riscv32 -mattr=+zcmop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zcmop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+zcmop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+zcmop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-OBJ %s

# c.mop.1 is an alias for c.sspush ra.
# CHECK-OBJ: c.sspush ra
# CHECK-ASM: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
c.mop.1

# CHECK-OBJ: c.mop.3
# CHECK-ASM: c.mop.3
# CHECK-ASM: encoding: [0x81,0x61]
c.mop.3

# c.mop.5 is an alias for c.sspopchk t0.
# CHECK-OBJ: c.sspopchk t0
# CHECK-ASM: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
c.mop.5

# CHECK-OBJ: c.mop.7
# CHECK-ASM: c.mop.7
# CHECK-ASM: encoding: [0x81,0x63]
c.mop.7

# CHECK-OBJ: c.mop.9
# CHECK-ASM: c.mop.9
# CHECK-ASM: encoding: [0x81,0x64]
c.mop.9

# CHECK-OBJ: c.mop.11
# CHECK-ASM: c.mop.11
# CHECK-ASM: encoding: [0x81,0x65]
c.mop.11

# CHECK-OBJ: c.mop.13
# CHECK-ASM: c.mop.13
# CHECK-ASM: encoding: [0x81,0x66]
c.mop.13

# CHECK-OBJ: c.mop.15
# CHECK-ASM: c.mop.15
# CHECK-ASM: encoding: [0x81,0x67]
c.mop.15
