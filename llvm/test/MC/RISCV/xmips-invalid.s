# RUN: llvm-mc --verify=feature -triple=riscv64 < %s
# RUN: llvm-mc --verify=check -triple=riscv64 -mattr=+xmipslsp,+xmipscmov,+xmipscbop,+xmipsexectl < %s
# RUN: llvm-mc --verify=rv32 -triple=riscv32 -mattr=+xmipslsp,+xmipscmov,+xmipscbop,+xmipsexectl < %s

mips.pause 10
# check-error@-1 {{unexpected extra operand for instruction}}
# feature-error@-2 {{invalid instruction}}
# rv32-error@-3 {{unexpected extra operand for instruction}}

mips.ehb 10
# check-error@-1 {{unexpected extra operand for instruction}}
# feature-error@-2 {{invalid instruction}}
# rv32-error@-3 {{unexpected extra operand for instruction}}

mips.ihb 10
# check-error@-1 {{unexpected extra operand for instruction}}
# feature-error@-2 {{invalid instruction}}
# rv32-error@-3 {{unexpected extra operand for instruction}}

mips.pref   8, 512(a0)
# check-error@-1 {{immediate offset must be in the range [0, 511]}}
# feature-error@-2 {{invalid instruction}}
# rv32-error@-3 {{immediate offset must be in the range [0, 511]}}

mips.pref	8
# check-error@-1 {{too few operands for instruction}}
# feature-error@-2 {{invalid instruction}}
# rv32-error@-3 {{too few operands for instruction}}

mips.pref	8, 511(a0) # feature-error {{instruction requires the following: 'Xmipscbop' (MIPS hardware prefetch)}}

mips.ccmov x0, x1, 0x10
# check-error@-1 {{register must be a GPR}}
# feature-error@-2 {{instruction requires the following: 'Xmipscmov' ('mips.ccmov' instruction)}}
# rv32-error@-3 {{register must be a GPR}}

mips.ccmov x10
# check-error@-1 {{too few operands for instruction}}
# feature-error@-2 {{invalid instruction}}
# rv32-error@-3 {{too few operands for instruction}}

mips.ccmov	s0, s1, s2, s3 # feature-error {{instruction requires the following: 'Xmipscmov' ('mips.ccmov' instruction)}}

mips.lwp x10, x11
# check-error@-1 {{too few operands for instruction}}
# feature-error@-2 {{invalid instruction}}
# rv32-error@-3 {{too few operands for instruction}}

mips.ldp x9, 0x20
# check-error@-1 {{register must be a GPR}}
# feature-error@-2 {{instruction requires the following: 'Xmipslsp' (load and store pair instructions)}}
# rv32-error@-3 {{instruction requires the following: RV64I Base Instruction Set}}

mips.lwp x11, x12, 0(x13) # feature-error {{instruction requires the following: 'Xmipslsp' (load and store pair instructions)}}

mips.swp x18, x19, 8(x2) # feature-error {{instruction requires the following: 'Xmipslsp' (load and store pair instructions)}}

mips.sdp 0x10, x3, 12(x4)
# check-error@-1 {{register must be a GPR}}
# feature-error@-2 {{instruction requires the following: 'Xmipslsp' (load and store pair instructions)}}
# rv32-error@-3 {{instruction requires the following: RV64I Base Instruction Set}}

mips.ldp s1, s2, 8(sp)
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# feature-error@-2 {{instruction requires the following: 'Xmipslsp' (load and store pair instructions)}}

mips.sdp s5, s6, 16(s7)
# rv32-error@-1:1 {{instruction requires the following: RV64I Base Instruction Set}}
# feature-error@-2 {{instruction requires the following: 'Xmipslsp' (load and store pair instructions)}}
