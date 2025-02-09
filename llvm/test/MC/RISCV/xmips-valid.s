# RUN: llvm-mc %s -triple=riscv64 -mattr=+xmipslsp,+xmipscmove -M no-aliases -show-encoding \
# RUN:   | FileCheck -check-prefixes=CHECK-INST,CHECK-ENC %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xmipslsp,+xmipscmove < %s \
# RUN:   | llvm-objdump --mattr=+xmipslsp,+xmipscmove -M no-aliases -d - \
# RUN:   | FileCheck -check-prefix=CHECK-DIS %s

# CHECK-INST: mips.ccmov	s0, s1, s2, s3
# CHECK-ENC:  encoding: [0x0b,0x34,0x99,0x9e]
mips.ccmov s0, s1, s2, s3

# CHECK-DIS: mips.ccmov	s0, s1, s2, s3

# CHECK-INST: mips.swp s3, s2, 0(sp)
# CHECK-ENC: encoding: [0x8b,0x50,0x31,0x91]
mips.swp s3, s2, 0(sp)

# CHECK-DIS: mips.swp s3, s2, 0x0(sp)

# CHECK-INST: mips.sdp s5, s6, 16(s7)
# CHECK-ENC: encoding: [0x0b,0xd8,0x5b,0xb1]
mips.sdp s5, s6, 16(s7)

# CHECK-DIS: mips.sdp s5, s6, 0x10(s7)

# CHECK-INST: mips.ldp s1, s2, 8(sp)
# CHECK-ENC: encoding: [0x8b,0x44,0x81,0x90]
mips.ldp s1, s2, 8(sp)

# CHECK-DIS: mips.ldp s1, s2, 0x8(sp)

# CHECK-INST: mips.lwp a0, a1, 20(a2)
# CHECK-ENC: encoding: [0x0b,0x45,0x56,0x59]
mips.lwp x10, x11, 20(x12)

# CHECK-DIS: mips.lwp a0, a1, 0x14(a2)
