# RUN: llvm-mc -triple=agx -mcpu=agx-g13g -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -show-encoding %s | FileCheck %s

# Encodings cross-checked against the applegpu reference assembler and
# disassembler (dougallj/applegpu): each case is a random canonical encoding
# that applegpu disassembles to the same text.

atomic umin, 0, r2, r37_r38, -28392, signed, r118, 0
# CHECK: atomic umin, 0, r2, r37_r38, -28392, signed, r118, 0{{ +}}// encoding: [0x15,0x11,0x8a,0x85,0x41,0xd0,0xec,0x91]
atomic and, 0, r39, r121_r122, -12555, unsigned, r21, 1
# CHECK: atomic and, 0, r39, r121_r122, -12555, unsigned, r21, 1{{ +}}// encoding: [0x15,0x3a,0x52,0x87,0xff,0xf1,0x2a,0xce]
atomic 11, 1, r74, r10_r11, -17279, signed, r76, 0
# CHECK: atomic 11, 1, r74, r10_r11, -17279, signed, r76, 0{{ +}}// encoding: [0xd5,0x52,0x14,0xc5,0x18,0xd2,0x98,0xbc]
atomic add, 0, r45, u43_u44, -1657, signed, r45, 1
# CHECK: atomic add, 0, r45, u43_u44, -1657, signed, r45, 1{{ +}}// encoding: [0x15,0x68,0x76,0x8d,0x58,0xf1,0x5a,0xf9]


device_store 1, i8, xy, r92l_r92h, r25_r26, 29863, signed, lsl 1
# CHECK: device_store 1, i8, xy, r92l_r92h, r25_r26, 29863, signed, lsl 1{{ +}}// encoding: [0x45,0xe0,0x72,0x45,0x3a,0xc6,0x30,0x74]
device_store 1, rgb10a2, xz, r97h_r98l, r107_r108, 14332, signed, lsl 1
# CHECK: device_store 1, rgb10a2, xz, r97h_r98l, r107_r108, 14332, signed, lsl 1{{ +}}// encoding: [0x45,0x0c,0xc6,0x45,0xdf,0xc7,0x51,0x37]
device_store 0, rg11b10f.rtz, xyw, r88h_r89l_r89h, r90_r91, 24244, signed
# CHECK: device_store 0, rg11b10f.rtz, xyw, r88h_r89l_r89h, r90_r91, 24244, signed{{ +}}// encoding: [0x45,0xc7,0x44,0x05,0xbb,0xc2,0xb1,0x5e]
device_store 0, srgba8, xyzw, r47h_r48l_r48h_r49l, u73_u74, -13603, signed, lsl 1
# CHECK: device_store 0, srgba8, xyzw, r47h_r48l_r48h_r49l, u73_u74, -13603, signed, lsl 1{{ +}}// encoding: [0x45,0x7d,0xd2,0x0d,0x9d,0xc5,0xf1,0xca]

threadgroup_load u8norm, xy, r92_r93, u217h, 17515
# CHECK: threadgroup_load u8norm, xy, r92_r93, u217h, 17515{{ +}}// encoding: [0x69,0xf1,0xf3,0xb4,0x36,0x00,0x11,0x29]
threadgroup_load i16, xyw, r105h_r106l_r106h, u110l, -27664
# CHECK: threadgroup_load i16, xyw, r105h_r106l_r106h, u110l, -27664{{ +}}// encoding: [0x69,0xa6,0x5c,0x01,0xb7,0x00,0x4f,0x3e]
threadgroup_load srgba8, x, r60h, r32l, -2730
# CHECK: threadgroup_load srgba8, x, r60h, r32l, -2730{{ +}}// encoding: [0x69,0xf2,0x00,0x6a,0x15,0x00,0xd5,0x17]
threadgroup_load i16, z, r10l, r103h, -28133
# CHECK: threadgroup_load i16, z, r10l, r103h, -28133{{ +}}// encoding: [0x69,0xa8,0x0f,0xb1,0x45,0x00,0x48,0x0e]

threadgroup_store rg11b10f.rtz, xy, r0h_r1l, r40l, -10643
# CHECK: threadgroup_store rg11b10f.rtz, xy, r0h_r1l, r40l, -10643{{ +}}// encoding: [0x29,0x82,0x10,0xde,0x36,0x00,0x59,0x07]
threadgroup_store u16norm, xy, r89h_r90l, r50l, -14279
# CHECK: threadgroup_store u16norm, xy, r89h_r90l, r50l, -14279{{ +}}// encoding: [0x29,0xe6,0x24,0x96,0x37,0x00,0x20,0x27]
threadgroup_store i32, xz, r110l_r110h, u65h, -17937
# CHECK: threadgroup_store i32, xz, r110l_r110h, u65h, -17937{{ +}}// encoding: [0x29,0xb8,0x43,0xf2,0x56,0x00,0xe7,0x3a]
threadgroup_store rg11b10f.rtz, xzw, r100l_r100h_r101l, u65h, 30145
# CHECK: threadgroup_store rg11b10f.rtz, xzw, r100l_r100h_r101l, u65h, 30145{{ +}}// encoding: [0x29,0x90,0x43,0x1e,0xd4,0x00,0xd7,0x39]


# Bits 45-46 of the device atomic (G14C measurements, Apple-M2-Playground
# 2026-10-17-atomic-variants): mode 3 clears bit 46 (u64 min with cmpxchg),
# and "_" leaves the result unwritten.
atomic cmpxchg, 0, _, r10_r11, 0, signed, r4_r5, 3
# CHECK: atomic cmpxchg, 0, _, r10_r11, 0, signed, r4_r5, 3{{ +}}// encoding: [0xd5,0x00,0x04,0x85,0x10,0x30,0x08,0x00]
atomic cmpxchg, 0, _, r10_r11, 0, signed, r4_r5, 1
# CHECK: atomic cmpxchg, 0, _, r10_r11, 0, signed, r4_r5, 1{{ +}}// encoding: [0xd5,0x00,0x04,0x85,0x10,0x70,0x08,0x00]

# The coherency operand of device loads and stores (bits 44-45; 3 is Mesa's
# coherent access, u4 = 7).
device_load 0, i32, x, r1, u0_u1, 0, signed, 3
# CHECK: device_load 0, i32, x, r1, u0_u1, 0, signed, 3 // encoding: [0x05,0x09,0x00,0x0d,0x00,0xf0,0x12,0x00]
device_store 0, i32, x, r1, u2_u3, 0, signed, 3
# CHECK: device_store 0, i32, x, r1, u2_u3, 0, signed, 3 // encoding: [0x45,0x09,0x04,0x0d,0x00,0xf0,0x12,0x00]
