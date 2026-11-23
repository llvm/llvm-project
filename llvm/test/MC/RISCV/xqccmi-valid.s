# Xqccmi - Qualcomm 16-bit Instruction Lookup Table
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqccmi -M no-aliases -show-encoding | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-xqccmi < %s | llvm-objdump --mattr=-c,+experimental-xqccmi --no-print-imm-hex -M no-aliases -d -r - | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: qc.cm.ilut 0
# CHECK-ASM: encoding: [0x00,0x20]
qc.cm.ilut 0

# CHECK-ASM-AND-OBJ: qc.cm.ilut 1
# CHECK-ASM: encoding: [0x04,0x20]
qc.cm.ilut 1

# CHECK-ASM-AND-OBJ: qc.cm.ilut 2047
# CHECK-ASM: encoding: [0xfc,0x3f]
qc.cm.ilut 2047

# CHECK-ASM-AND-OBJ: csrrs t2, qc.itba, zero
# CHECK-ASM: encoding: [0xf3,0x23,0x00,0x80]
csrrs t2, qc.itba, zero

# CHECK-ASM-AND-OBJ: csrrs t2, qc.itdec, zero
# CHECK-ASM: encoding: [0xf3,0x23,0x10,0x80]
csrrs t2, qc.itdec, zero
