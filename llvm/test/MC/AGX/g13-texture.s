# RUN: llvm-mc -triple=agx -mcpu=agx-g13g -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s --check-prefix=DIS

# Bindless texture_sample (texture descriptor at u0_u1 plus a 32-bit offset
# register, register sampler index, automatic LOD): the short form, and the
# long one for registers above the 64th half and scoreboard group 1. Both
# decode to the same operands in applegpu's disassembler (AGX Performance
# Simulator agx_asm).

texture_sample 0, r7_r8_r9_r10, r4, r3l, r5_r6, 2, xyzw
# CHECK: texture_sample 0, r7_r8_r9_r10, r4, r3l, r5_r6, 2, xyzw{{ +}}// encoding: [0x31,0x1d,0x0a,0x00,0xc8,0x62,0x0f,0x46]
texture_sample 1, r39_r40_r41_r42, r4, r99l, r37, 0, xyzw
# CHECK: texture_sample 1, r39_r40_r41_r42, r4, r99l, r37, 0, xyzw{{ +}}// encoding: [0x31,0x9d,0x0a,0x00,0xc8,0x60,0x0f,0xc6,0x00,0x05,0x00,0x30]

# DIS: 31 1d 0a 00 c8 62 0f 46 texture_sample 0, r7_r8_r9_r10, r4, r3l, r5, 2, xyzw
# DIS: 31 9d 0a 00 c8 60 0f c6 00 05 00 30 texture_sample 1, r39_r40_r41_r42, r4, r99l, r37, 0, xyzw
