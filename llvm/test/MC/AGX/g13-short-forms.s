# RUN: llvm-mc -triple=agx -mcpu=agx-g13g -show-encoding %s | FileCheck %s

# Short forms of the memory instructions, threadgroup atomics with a 16-bit
# result, the device store flag (bit 44) and relative branch targets, found by
# the differential test against the AGX Sail model (PTX-formalization,
# sail/agx/gen/mccheck.py). Encodings cross-checked against applegpu.

threadgroup_load i16, xw, r22_r23, 0, r23h
# CHECK: threadgroup_load i16, xw, r22_r23, 0, r23h{{ +}}// encoding: [0x69,0x59,0x80,0xf1,0x92,0x00]
threadgroup_store i32, xyw, r26_r27_r28, 0, r16l
# CHECK: threadgroup_store i32, xyw, r26_r27_r28, 0, r16l{{ +}}// encoding: [0x29,0x69,0x80,0x02,0xb2,0x00]
threadgroup_load i32, x, r4, u3l, 12
# CHECK: threadgroup_load i32, x, r4, u3l, 12{{ +}}// encoding: [0x69,0x11,0x46,0xc2,0x14,0x00]
device_load 0, i16, 0, r112l, r114_r115, r126, signed, lsl 1
# CHECK: device_load 0, i16, 0, r112l, r114_r115, r126, signed, lsl 1{{ +}}// encoding: [0x85,0x80,0xc4,0x04,0xef,0x47]
threadgroup_atomic xor, r1h, 0, 0, r1
# CHECK: threadgroup_atomic xor, r1h, 0, 0, r1{{ +}}// encoding: [0x19,0x86,0x80,0x0a,0x54,0x80,0x00,0x00,0x02,0x00]
threadgroup_atomic xchg, r60l, u116h, r126l, r7
# CHECK: threadgroup_atomic xchg, r60l, u116h, r126l, r7{{ +}}// encoding: [0x19,0xf0,0x69,0xc2,0x53,0x80,0x03,0x1c,0x0e,0x00]
threadgroup_atomic add, r5, 0, 3, r0
# CHECK: threadgroup_atomic add, r5, 0, 3, r0{{ +}}// encoding: [0x19,0x15,0x80,0x30,0x54,0x80]
threadgroup_atomic umax, _, r2l, 7, r0
# CHECK: threadgroup_atomic umax, _, r2l, 7, r0{{ +}}// encoding: [0x19,0x00,0x04,0x76,0x14,0x80]
device_store 0, i32, x, r76, r0_r1, r6, signed, 1
# CHECK: device_store 0, i32, x, r76, r0_r1, r6, signed, 1{{ +}}// encoding: [0x45,0x61,0xc0,0x04,0x00,0xd2,0x12,0x00]
device_store 1, i16, y, r2l, r57_r58, 19471, signed, lsl 2, 1
# CHECK: device_store 1, i16, y, r2l, r57_r58, 19471, signed, lsl 2, 1{{ +}}// encoding: [0xc5,0x10,0xf2,0x45,0x70,0xd8,0x20,0x4c]
jmp_exec_any pc-222
# CHECK: jmp_exec_any pc-222{{ +}}// encoding: [0x00,0xc0,0x22,0xff,0xff,0xff]
jmp_exec_none pc+1545958588
# CHECK: jmp_exec_none pc+1545958588{{ +}}// encoding: [0x20,0xc0,0xbc,0x74,0x25,0x5c]
