# RUN: llvm-mc %s -triple=xtensa -show-encoding --mattr=+regprotect \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

# Instruction format RRR
# CHECK-INST: idtlb a3
# CHECK: encoding: [0x00,0xc3,0x50]
idtlb a3

# Instruction format RRR
# CHECK-INST: iitlb a3
# CHECK: encoding: [0x00,0x43,0x50]
iitlb a3

# Instruction format RRR
# CHECK-INST: pdtlb a3, a4
# CHECK: encoding: [0x30,0xd4,0x50]
pdtlb a3, a4

# Instruction format RRR
# CHECK-INST: pitlb a3, a4
# CHECK: encoding: [0x30,0x54,0x50]
pitlb a3, a4

# Instruction format RRR
# CHECK-INST: rdtlb0 a3, a4
# CHECK: encoding: [0x30,0xb4,0x50]
rdtlb0 a3, a4

# Instruction format RRR
# CHECK-INST: rdtlb1 a3, a4
# CHECK: encoding: [0x30,0xf4,0x50]
rdtlb1 a3, a4

# Instruction format RRR
# CHECK-INST: ritlb0 a3, a4
# CHECK: encoding: [0x30,0x34,0x50]
ritlb0 a3, a4

# Instruction format RRR
# CHECK-INST: ritlb1 a3, a4
# CHECK: encoding: [0x30,0x74,0x50]
ritlb1 a3, a4

# Instruction format RRR
# CHECK-INST: wdtlb a3, a4
# CHECK: encoding: [0x30,0xe4,0x50]
wdtlb a3, a4

# Instruction format RRR
# CHECK-INST: witlb a3, a4
# CHECK: encoding: [0x30,0x64,0x50]
witlb a3, a4
