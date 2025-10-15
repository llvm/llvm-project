# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+mul16,mul32high,mul32 \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRR
# CHECK-INST: mul16s a3, a4, a5
# CHECK: encoding: [0x50,0x34,0xd1]
mul16s a3, a4, a5

# Instruction format RRR
# CHECK-INST: mul16u a3, a4, a5
# CHECK: encoding: [0x50,0x34,0xc1]
mul16u a3, a4, a5

# Instruction format RRR
# CHECK-INST: mull a3, a4, a5
# CHECK: encoding:  [0x50,0x34,0x82]
mull a3, a4, a5

# Instruction format RRR
# CHECK-INST: muluh a3, a4, a5
# CHECK: encoding:  [0x50,0x34,0xa2]
muluh a3, a4, a5

# Instruction format RRR
# CHECK-INST: mulsh a3, a4, a5
# CHECK: encoding:  [0x50,0x34,0xb2]
mulsh a3, a4, a5
