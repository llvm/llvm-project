# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zicfiss,+zcmop -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfiss,+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zicfiss,+zcmop -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zicfiss,+zcmop -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfiss,+zcmop < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zicfiss,+zcmop -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk x5

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk t0

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush x1

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush ra

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
# CHECK-NO-EXT: error: instruction requires the following: 'Zcmop' (Compressed May-Be-Operations), 'Zicfiss' (Shadow stack)
c.sspush x1

# CHECK-ASM-AND-OBJ: c.sspush ra
# CHECK-ASM: encoding: [0x81,0x60]
# CHECK-NO-EXT: error: instruction requires the following: 'Zcmop' (Compressed May-Be-Operations), 'Zicfiss' (Shadow stack)
c.sspush ra

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
# CHECK-NO-EXT: error: instruction requires the following: 'Zcmop' (Compressed May-Be-Operations), 'Zicfiss' (Shadow stack)
c.sspopchk x5

# CHECK-ASM-AND-OBJ: c.sspopchk t0
# CHECK-ASM: encoding: [0x81,0x62]
# CHECK-NO-EXT: error: instruction requires the following: 'Zcmop' (Compressed May-Be-Operations), 'Zicfiss' (Shadow stack)
c.sspopchk t0
