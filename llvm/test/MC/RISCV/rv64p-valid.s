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
# CHECK-ASM-AND-OBJ: psll.hs s0, a2, s2
# CHECK-ASM: encoding: [0x1b,0x24,0x26,0x89]
psll.hs s0, a2, s2
# CHECK-ASM-AND-OBJ: psll.bs a0, t3, t5
# CHECK-ASM: encoding: [0x1b,0x25,0xee,0x8d]
psll.bs a0, t3, t5
# CHECK-ASM-AND-OBJ: padd.hs t1, a2, s0
# CHECK-ASM: encoding: [0x1b,0x23,0x86,0x98]
padd.hs t1, a2, s0
# CHECK-ASM-AND-OBJ: padd.bs t3, t1, t3
# CHECK-ASM: encoding: [0x1b,0x2e,0xc3,0x9d]
padd.bs t3, t1, t3
# CHECK-ASM-AND-OBJ: pssha.hs s0, t1, a2
# CHECK-ASM: encoding: [0x1b,0x24,0xc3,0xe8]
pssha.hs s0, t1, a2
# CHECK-ASM-AND-OBJ: psshar.hs s2, t5, t3
# CHECK-ASM: encoding: [0x1b,0x29,0xcf,0xf9]
psshar.hs s2, t5, t3
# CHECK-ASM-AND-OBJ: psll.ws s0, t1, a0
# CHECK-ASM: encoding: [0x1b,0x24,0xa3,0x8a]
psll.ws s0, t1, a0
# CHECK-ASM-AND-OBJ: padd.ws s2, a2, a0
# CHECK-ASM: encoding: [0x1b,0x29,0xa6,0x9a]
padd.ws s2, a2, a0
# CHECK-ASM-AND-OBJ: pssha.ws a4, a2, t1
# CHECK-ASM: encoding: [0x1b,0x27,0x66,0xea]
pssha.ws a4, a2, t1
# CHECK-ASM-AND-OBJ: psshar.ws a2, a0, a4
# CHECK-ASM: encoding: [0x1b,0x26,0xe5,0xfa]
psshar.ws a2, a0, a4
# CHECK-ASM-AND-OBJ: sha a0, t5, t5
# CHECK-ASM: encoding: [0x1b,0x25,0xef,0xef]
sha a0, t5, t5
# CHECK-ASM-AND-OBJ: shar t5, t5, t3
# CHECK-ASM: encoding: [0x1b,0x2f,0xcf,0xff]
shar t5, t5, t3
# CHECK-ASM-AND-OBJ: psrli.b a6, a7
# CHECK-ASM: encoding: [0x1b,0xc8,0x88,0x80]
psrli.b a6, a7, 0
# CHECK-ASM-AND-OBJ: psrli.h ra, sp, 1
# CHECK-ASM: encoding: [0x9b,0x40,0x11,0x81]
psrli.h ra, sp, 1
# CHECK-ASM-AND-OBJ: psrli.w ra, sp, 2
# CHECK-ASM: encoding: [0x9b,0x40,0x21,0x82]
psrli.w ra, sp, 2
# CHECK-ASM-AND-OBJ: pusati.h t2, t3, 4
# CHECK-ASM: encoding: [0x9b,0x43,0x4e,0xa1]
pusati.h t2, t3, 4
# CHECK-ASM-AND-OBJ: pusati.w t2, t3, 5
# CHECK-ASM: encoding: [0x9b,0x43,0x5e,0xa2]
pusati.w t2, t3, 5
# CHECK-ASM-AND-OBJ: usati t3, t4, 5
# CHECK-ASM: encoding: [0x1b,0xce,0x5e,0xa4]
usati t3, t4, 5
# CHECK-ASM-AND-OBJ: psrai.b a6, a7, 0
# CHECK-ASM: encoding: [0x1b,0xc8,0x88,0xc0]
psrai.b a6, a7, 0
# CHECK-ASM-AND-OBJ: psrai.h ra, sp, 1
# CHECK-ASM: encoding: [0x9b,0x40,0x11,0xc1]
psrai.h ra, sp, 1
# CHECK-ASM-AND-OBJ: psrai.w ra, sp, 2
# CHECK-ASM: encoding: [0x9b,0x40,0x21,0xc2]
psrai.w ra, sp, 2
# CHECK-ASM-AND-OBJ: psrari.h t4, t5, 6
# CHECK-ASM: encoding: [0x9b,0x4e,0x6f,0xd1]
psrari.h t4, t5, 6
# CHECK-ASM-AND-OBJ: psrari.w t5, t6, 7
# CHECK-ASM: encoding: [0x1b,0xcf,0x7f,0xd2]
psrari.w t5, t6, 7
# CHECK-ASM-AND-OBJ: srari t6, s11, 63
# CHECK-ASM: encoding: [0x9b,0xcf,0xfd,0xd7]
srari t6, s11, 63
# CHECK-ASM-AND-OBJ: psati.h s11, s10, 9
# CHECK-ASM: encoding: [0x9b,0x4d,0x9d,0xe1]
psati.h s11, s10, 9
# CHECK-ASM-AND-OBJ: psati.w s10, s9, 10
# CHECK-ASM: encoding: [0x1b,0xcd,0xac,0xe2]
psati.w s10, s9, 10
# CHECK-ASM-AND-OBJ: sati s9, s8, 32
# CHECK-ASM: encoding: [0x9b,0x4c,0x0c,0xe6]
sati s9, s8, 32
# CHECK-ASM-AND-OBJ: psrl.hs a6, a7, a1
# CHECK-ASM: encoding: [0x1b,0xc8,0xb8,0x88]
psrl.hs a6, a7, a1
# CHECK-ASM-AND-OBJ: psrl.bs a1, a2, a3
# CHECK-ASM: encoding: [0x9b,0x45,0xd6,0x8c]
psrl.bs a1, a2, a3
# CHECK-ASM-AND-OBJ: predsum.hs a4, a5, a6
# CHECK-ASM: encoding: [0x1b,0xc7,0x07,0x99]
predsum.hs a4, a5, a6
# CHECK-ASM-AND-OBJ: predsum.bs a7, a1, a1
# CHECK-ASM: encoding: [0x9b,0xc8,0xb5,0x9c]
predsum.bs a7, a1, a1
# CHECK-ASM-AND-OBJ: predsumu.hs t0, t1, t2
# CHECK-ASM: encoding: [0x9b,0x42,0x73,0xb8]
predsumu.hs t0, t1, t2
# CHECK-ASM-AND-OBJ: predsumu.bs t3, t4, t5
# CHECK-ASM: encoding: [0x1b,0xce,0xee,0xbd]
predsumu.bs t3, t4, t5
# CHECK-ASM-AND-OBJ: psra.hs ra, a1, a2
# CHECK-ASM: encoding: [0x9b,0xc0,0xc5,0xc8]
psra.hs ra, a1, a2
# CHECK-ASM-AND-OBJ: psra.bs sp, a2, a3
# CHECK-ASM: encoding: [0x1b,0x41,0xd6,0xcc]
psra.bs sp, a2, a3
