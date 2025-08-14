# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-p -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-p < %s \
# RUN:     | llvm-objdump --mattr=+experimental-p -M no-aliases -d -r --no-print-imm-hex - \
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
# CHECK-ASM-AND-OBJ: rev8 s0, s1
# CHECK-ASM: encoding: [0x13,0xd4,0x84,0x69]
rev8 s0, s1
# CHECK-ASM-AND-OBJ: rev s2, s3
# CHECK-ASM: encoding: [0x13,0xd9,0xf9,0x69]
rev s2, s3
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
# CHECK-ASM-AND-OBJ: psslai.h t0, t1, 2
# CHECK-ASM: encoding: [0x9b,0x22,0x23,0xd1]
psslai.h t0, t1, 2
# CHECK-ASM-AND-OBJ: sslai a4, a5, 3
# CHECK-ASM: encoding: [0x1b,0xa7,0x37,0xd2]
sslai a4, a5, 3
# CHECK-ASM-AND-OBJ: pli.h a5, 16
# CHECK-ASM: encoding: [0x9b,0x27,0x10,0xb0]
pli.h a5, 16
# CHECK-ASM-AND-OBJ: pli.b a6, 16
# CHECK-ASM: encoding: [0x1b,0x28,0x10,0xb4]
pli.b a6, 16
# CHECK-ASM-AND-OBJ: psext.h.b a7, a0
# CHECK-ASM: encoding: [0x9b,0x28,0x45,0xe0]
psext.h.b a7, a0
# CHECK-ASM-AND-OBJ: psabs.h a1, a2
# CHECK-ASM: encoding: [0x9b,0x25,0x76,0xe0]
psabs.h a1, a2
# CHECK-ASM-AND-OBJ: psabs.b t0, t1
# CHECK-ASM: encoding: [0x9b,0x22,0x73,0xe4]
psabs.b t0, t1
# CHECK-ASM-AND-OBJ: plui.h gp, 32
# CHECK-ASM: encoding: [0x9b,0x21,0x08,0xf0]
plui.h gp, 32
# CHECK-ASM-AND-OBJ: plui.h gp, -412
# CHECK-ASM: encoding: [0x9b,0x21,0x99,0xf0]
plui.h gp, 612
# CHECK-ASM-AND-OBJ: psll.hs a0, a1, a2
# CHECK-ASM: encoding: [0x1b,0xa5,0xc5,0x88]
psll.hs a0, a1, a2
# CHECK-ASM-AND-OBJ: psll.bs a3, a4, a5
# CHECK-ASM: encoding: [0x9b,0x26,0xf7,0x8c]
psll.bs a3, a4, a5
# CHECK-ASM-AND-OBJ: padd.hs t0, t1, t2
# CHECK-ASM: encoding: [0x9b,0x22,0x73,0x98]
padd.hs t0, t1, t2
# CHECK-ASM-AND-OBJ: padd.bs ra, a1, a2
# CHECK-ASM: encoding: [0x9b,0xa0,0xc5,0x9c]
padd.bs ra, a1, a2
# CHECK-ASM-AND-OBJ: pssha.hs a3, a4, a5
# CHECK-ASM: encoding: [0x9b,0x26,0xf7,0xe8]
pssha.hs a3, a4, a5
# CHECK-ASM-AND-OBJ: ssha gp, a4, a5
# CHECK-ASM: encoding: [0x9b,0x21,0xf7,0xea]
ssha gp, a4, a5
# CHECK-ASM-AND-OBJ: psshar.hs a6, a7, a0
# CHECK-ASM: encoding: [0x1b,0xa8,0xa8,0xf8]
psshar.hs a6, a7, a0
# CHECK-ASM-AND-OBJ: sshar t1, a7, a0
# CHECK-ASM: encoding: [0x1b,0xa3,0xa8,0xfa]
sshar t1, a7, a0
# CHECK-ASM-AND-OBJ: psrli.b a6, a7, 0
# CHECK-ASM: encoding: [0x1b,0xc8,0x88,0x80]
psrli.b a6, a7, 0
# CHECK-ASM-AND-OBJ: psrli.h ra, sp, 1
# CHECK-ASM: encoding: [0x9b,0x40,0x11,0x81]
psrli.h ra, sp, 1
# CHECK-ASM-AND-OBJ: pusati.h t2, t3, 4
# CHECK-ASM: encoding: [0x9b,0x43,0x4e,0xa1]
pusati.h t2, t3, 4
# CHECK-ASM-AND-OBJ: usati t3, t4, 5
# CHECK-ASM: encoding: [0x1b,0xce,0x5e,0xa2]
usati t3, t4, 5
# CHECK-ASM-AND-OBJ: psrai.b a6, a7, 0
# CHECK-ASM: encoding: [0x1b,0xc8,0x88,0xc0]
psrai.b a6, a7, 0
# CHECK-ASM-AND-OBJ: psrai.h ra, sp, 1
# CHECK-ASM: encoding: [0x9b,0x40,0x11,0xc1]
psrai.h ra, sp, 1
# CHECK-ASM-AND-OBJ: psrari.h t4, t5, 6
# CHECK-ASM: encoding: [0x9b,0x4e,0x6f,0xd1]
psrari.h t4, t5, 6
# CHECK-ASM-AND-OBJ: srari t5, t6, 7
# CHECK-ASM: encoding: [0x1b,0xcf,0x7f,0xd2]
srari t5, t6, 7
# CHECK-ASM-AND-OBJ: psati.h t6, s11, 8
# CHECK-ASM: encoding: [0x9b,0xcf,0x8d,0xe1]
psati.h t6, s11, 8
# CHECK-ASM-AND-OBJ: sati s11, s10, 9
# CHECK-ASM: encoding: [0x9b,0x4d,0x9d,0xe2]
sati s11, s10, 9
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
