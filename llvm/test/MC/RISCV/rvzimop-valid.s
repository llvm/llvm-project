# RUN: llvm-mc %s -triple=riscv32 -mattr=+zimop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zimop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zimop < %s \
# RUN:     | llvm-objdump --mattr=+zimop -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zimop < %s \
# RUN:     | llvm-objdump --mattr=+zimop -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: mop.r.0 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0x81]
mop.r.0 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.1 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0x81]
mop.r.1 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.2 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0x81]
mop.r.2 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.3 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0x81]
mop.r.3 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.4 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0x85]
mop.r.4 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.5 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0x85]
mop.r.5 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.6 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0x85]
mop.r.6 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.7 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0x85]
mop.r.7 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.8 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0x89]
mop.r.8 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.9 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0x89]
mop.r.9 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.10 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0x89]
mop.r.10 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.11 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0x89]
mop.r.11 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.12 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0x8d]
mop.r.12 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.13 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0x8d]
mop.r.13 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.14 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0x8d]
mop.r.14 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.15 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0x8d]
mop.r.15 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.16 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0xc1]
mop.r.16 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.17 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0xc1]
mop.r.17 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.18 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0xc1]
mop.r.18 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.19 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0xc1]
mop.r.19 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.20 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0xc5]
mop.r.20 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.21 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0xc5]
mop.r.21 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.22 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0xc5]
mop.r.22 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.23 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0xc5]
mop.r.23 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.24 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0xc9]
mop.r.24 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.25 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0xc9]
mop.r.25 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.26 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0xc9]
mop.r.26 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.27 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0xc9]
mop.r.27 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.28 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0xcd]
mop.r.28 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.29 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xd5,0xcd]
mop.r.29 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.30 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xe5,0xcd]
mop.r.30 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.31 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0xcd]
mop.r.31 a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.0 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0x82]
mop.rr.0 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.1 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0x86]
mop.rr.1 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.2 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0x8a]
mop.rr.2 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.3 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0x8e]
mop.rr.3 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.4 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0xc2]
mop.rr.4 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.5 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0xc6]
mop.rr.5 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.6 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0xca]
mop.rr.6 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.7 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0xce]
mop.rr.7 a3, a2, a1

# mop.r.28 and mop.rr.7 share their encoding with sspopchk/ssrdp and sspush
# respectively (see RISCVInstrInfoZicfiss.td), which are also gated solely
# on Zimop. sspopchk/sspush only accept x1 or x5 for their free register,
# and ssrdp rejects x0. When those constraints aren't met, the disassembler
# falls back to decoding the generic mop.r.28/mop.rr.7 instruction instead
# (via hasCompleteDecoder = 0 on GPRX1X5/GPRNoX0Zicfiss).
# CHECK-ASM-AND-OBJ: mop.r.28 zero, gp
# CHECK-ASM: encoding: [0x73,0xc0,0xc1,0xcd]
mop.r.28 x0, x3

# CHECK-ASM-AND-OBJ: mop.r.28 zero, zero
# CHECK-ASM: encoding: [0x73,0x40,0xc0,0xcd]
mop.r.28 x0, x0

# CHECK-ASM-AND-OBJ: mop.rr.7 zero, zero, gp
# CHECK-ASM: encoding: [0x73,0x40,0x30,0xce]
mop.rr.7 x0, x0, x3