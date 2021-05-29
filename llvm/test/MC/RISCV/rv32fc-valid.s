# RUN: llvm-mc %s -triple=riscv32 -mattr=+c,+f -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+f < %s \
# RUN:     | llvm-objdump --mattr=+c,+f -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+c \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT-F %s
# RUN: not llvm-mc -triple riscv32 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT-FC %s
# RUN: not llvm-mc -triple riscv64 -mattr=+c,+f \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-RV32 %s

# FIXME: error messages for rv64fc are misleading

# CHECK-ASM-AND-OBJ: c.flwsp  fs0, 252(sp)
# CHECK-ASM: encoding: [0x7e,0x74]
# CHECK-NO-EXT-F:  error: instruction requires the following: 'F' (Single-Precision Floating-Point)
# CHECK-NO-EXT-FC:  error: instruction requires the following: 'C' (Compressed Instructions), 'F' (Single-Precision Floating-Point)
# CHECK-NO-RV32:  error: instruction requires the following: RV32I Base Instruction Set
c.flwsp  fs0, 252(sp)
# CHECK-ASM-AND-OBJ: c.fswsp  fa7, 252(sp)
# CHECK-ASM: encoding: [0xc6,0xff]
# CHECK-NO-EXT-F:  error: instruction requires the following: 'F' (Single-Precision Floating-Point)
# CHECK-NO-EXT-FC:  error: instruction requires the following: 'C' (Compressed Instructions), 'F' (Single-Precision Floating-Point)
# CHECK-NO-RV32:  error: instruction requires the following: RV32I Base Instruction Set
c.fswsp  fa7, 252(sp)

# CHECK-ASM-AND-OBJ: c.flw  fa3, 124(a5)
# CHECK-ASM: encoding: [0xf4,0x7f]
# CHECK-NO-EXT-F:  error: instruction requires the following: 'F' (Single-Precision Floating-Point)
# CHECK-NO-EXT-FC:  error: instruction requires the following: 'C' (Compressed Instructions), 'F' (Single-Precision Floating-Point)
# CHECK-NO-RV32:  error: instruction requires the following: RV32I Base Instruction Set
c.flw  fa3, 124(a5)
# CHECK-ASM-AND-OBJ: c.fsw  fa2, 124(a1)
# CHECK-ASM: encoding: [0xf0,0xfd]
# CHECK-NO-EXT-F:  error: instruction requires the following: 'F' (Single-Precision Floating-Point)
# CHECK-NO-EXT-FC:  error: instruction requires the following: 'C' (Compressed Instructions), 'F' (Single-Precision Floating-Point)
# CHECK-NO-RV32:  error: instruction requires the following: RV32I Base Instruction Set
c.fsw  fa2, 124(a1)
