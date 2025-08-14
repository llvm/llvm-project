# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-p -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj --triple=riscv64 -mattr=+experimental-p < %s \
# RUN:     | llvm-objdump --triple=riscv64 --mattr=+experimental-p -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: clz a0, a1
# CHECK-ASM: encoding: [0x13,0x95,0x05,0x60]
clz a0, a1
# CHECK-ASM-AND-OBJ: cls a1, a2
# CHECK-ASM: encoding: [0x93,0x15,0x36,0x60]
cls a1, a2
# CHECK-ASM-AND-OBJ: sext.b a2, a3
# CHECK-ASM: encoding: [0x13,0x96,0x46,0x60]
sext.b a2, a3
# CHECK-ASM-AND-OBJ: sext.h t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x53,0x60]
sext.h t0, t1
# CHECK-ASM-AND-OBJ: abs a4, a5
# CHECK-ASM: encoding: [0x13,0x97,0x77,0x60]
abs a4, a5
# CHECK-ASM-AND-OBJ: rev16 s0, s1
# CHECK-ASM: encoding: [0x13,0xd4,0x04,0x6b]
rev16 s0, s1
# CHECK-ASM-AND-OBJ: rev8 s0, s1
# CHECK-ASM: encoding: [0x13,0xd4,0x84,0x6b]
rev8 s0, s1
# CHECK-ASM-AND-OBJ: rev s2, s3
# CHECK-ASM: encoding: [0x13,0xd9,0xf9,0x6b]
rev s2, s3
# CHECK-ASM-AND-OBJ: clzw s0, s1
# CHECK-ASM: encoding: [0x1b,0x94,0x04,0x60]
clzw s0, s1
# CHECK-ASM-AND-OBJ: clsw s2, s3
# CHECK-ASM: encoding: [0x1b,0x99,0x39,0x60]
clsw s2, s3
# CHECK-ASM-AND-OBJ: absw s2, s3
# CHECK-ASM: encoding: [0x1b,0x99,0x79,0x60]
absw s2, s3
# CHECK-ASM-AND-OBJ: sh1add a0, a1, a2
# CHECK-ASM: encoding: [0x33,0xa5,0xc5,0x20]
sh1add a0, a1, a2
# CHECK-ASM-AND-OBJ: pack s0, s1, s2
# CHECK-ASM: encoding: [0x33,0xc4,0x24,0x09]
pack s0, s1, s2
# CHECK-ASM-AND-OBJ: min t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x0a]
min t0, t1, t2
# CHECK-ASM-AND-OBJ: minu ra, sp, gp
# CHECK-ASM: encoding: [0xb3,0x50,0x31,0x0a]
minu ra, sp, gp
# CHECK-ASM-AND-OBJ: max t3, t4, t5
# CHECK-ASM: encoding: [0x33,0xee,0xee,0x0b]
max t3, t4, t5
# CHECK-ASM-AND-OBJ: maxu a4, a5, a6
# CHECK-ASM: encoding: [0x33,0xf7,0x07,0x0b]
maxu a4, a5, a6
# CHECK-ASM-AND-OBJ: pslli.b a6, a7, 0
# CHECK-ASM: encoding: [0x1b,0xa8,0x88,0x80]
pslli.b a6, a7, 0
# CHECK-ASM-AND-OBJ: pslli.h ra, sp, 1
# CHECK-ASM: encoding: [0x9b,0x20,0x11,0x81]
pslli.h ra, sp, 1
# CHECK-ASM-AND-OBJ: pslli.w ra, sp, 2
# CHECK-ASM: encoding: [0x9b,0x20,0x21,0x82]
pslli.w ra, sp, 2
# CHECK-ASM-AND-OBJ: psslai.h t0, t1, 3
# CHECK-ASM: encoding: [0x9b,0x22,0x33,0xd1]
psslai.h t0, t1, 3
# CHECK-ASM-AND-OBJ: psslai.w a4, a5, 4
# CHECK-ASM: encoding: [0x1b,0xa7,0x47,0xd2]
psslai.w a4, a5, 4
# CHECK-ASM-AND-OBJ: pli.h a5, 5
# CHECK-ASM: encoding: [0x9b,0x27,0x05,0xb0]
pli.h a5, 5
# CHECK-ASM-AND-OBJ: pli.w a5, 5
# CHECK-ASM: encoding: [0x9b,0x27,0x05,0xb2]
pli.w a5, 5
# CHECK-ASM-AND-OBJ: pli.b a6, 6
# CHECK-ASM: encoding: [0x1b,0x28,0x06,0xb4]
pli.b a6, 6
# CHECK-ASM-AND-OBJ: psext.h.b t3, a2
# CHECK-ASM: encoding: [0x1b,0x2e,0x46,0xe0]
psext.h.b t3, a2
# CHECK-ASM-AND-OBJ: psext.w.b a2, s0
# CHECK-ASM: encoding: [0x1b,0x26,0x44,0xe2]
psext.w.b a2, s0
# CHECK-ASM-AND-OBJ: psext.w.h t1, t3
# CHECK-ASM: encoding: [0x1b,0x23,0x5e,0xe2]
psext.w.h t1, t3
# CHECK-ASM-AND-OBJ: psabs.h t1, t5
# CHECK-ASM: encoding: [0x1b,0x23,0x7f,0xe0]
psabs.h t1, t5
# CHECK-ASM-AND-OBJ: psabs.b a0, s2
# CHECK-ASM: encoding: [0x1b,0x25,0x79,0xe4]
psabs.b a0, s2
# CHECK-ASM-AND-OBJ: plui.h s2, 4
# CHECK-ASM: encoding: [0x1b,0x29,0x01,0xf0]
plui.h s2, 4
# CHECK-ASM-AND-OBJ: plui.h gp, -412
# CHECK-ASM: encoding: [0x9b,0x21,0x99,0xf0]
plui.h gp, 612
# CHECK-ASM-AND-OBJ: plui.w a2, 1
# CHECK-ASM: encoding: [0x1b,0x26,0x00,0xf3]
plui.w a2, 1
# CHECK-ASM-AND-OBJ: plui.w a2, -1
# CHECK-ASM: encoding: [0x1b,0xa6,0xff,0xf3]
plui.w a2, 1023
