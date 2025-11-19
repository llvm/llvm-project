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
# CHECK-ASM-AND-OBJ: pli.b a6, -1
# CHECK-ASM: encoding: [0x1b,0x28,0xff,0xb4]
pli.b a6, -1
# CHECK-ASM-AND-OBJ: pli.b a6, -1
# CHECK-ASM: encoding: [0x1b,0x28,0xff,0xb4]
pli.b a6, 255
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
# CHECK-ASM-AND-OBJ: padd.h t1, t5, s2
# CHECK-ASM: encoding: [0x3b,0x03,0x2f,0x81]
padd.h t1, t5, s2
# CHECK-ASM-AND-OBJ: padd.w t3, s0, a0
# CHECK-ASM: encoding: [0x3b,0x0e,0xa4,0x82]
padd.w t3, s0, a0
# CHECK-ASM-AND-OBJ: padd.b t5, s0, t1
# CHECK-ASM: encoding: [0x3b,0x0f,0x64,0x84]
padd.b t5, s0, t1
# CHECK-ASM-AND-OBJ: psadd.h a2, a2, s2
# CHECK-ASM: encoding: [0x3b,0x06,0x26,0x91]
psadd.h a2, a2, s2
# CHECK-ASM-AND-OBJ: psadd.w t3, t1, s2
# CHECK-ASM: encoding: [0x3b,0x0e,0x23,0x93]
psadd.w t3, t1, s2
# CHECK-ASM-AND-OBJ: psadd.b t1, a0, s0
# CHECK-ASM: encoding: [0x3b,0x03,0x85,0x94]
psadd.b t1, a0, s0
# CHECK-ASM-AND-OBJ: paadd.h t5, s0, t3
# CHECK-ASM: encoding: [0x3b,0x0f,0xc4,0x99]
paadd.h t5, s0, t3
# CHECK-ASM-AND-OBJ: paadd.w t5, t1, a4
# CHECK-ASM: encoding: [0x3b,0x0f,0xe3,0x9a]
paadd.w t5, t1, a4
# CHECK-ASM-AND-OBJ: paadd.b a4, s2, a4
# CHECK-ASM: encoding: [0x3b,0x07,0xe9,0x9c]
paadd.b a4, s2, a4
# CHECK-ASM-AND-OBJ: psaddu.h a0, t1, t1
# CHECK-ASM: encoding: [0x3b,0x05,0x63,0xb0]
psaddu.h a0, t1, t1
# CHECK-ASM-AND-OBJ: psaddu.w s0, s2, t5
# CHECK-ASM: encoding: [0x3b,0x04,0xe9,0xb3]
psaddu.w s0, s2, t5
# CHECK-ASM-AND-OBJ: psaddu.b t3, a2, a4
# CHECK-ASM: encoding: [0x3b,0x0e,0xe6,0xb4]
psaddu.b t3, a2, a4
# CHECK-ASM-AND-OBJ: paaddu.h t3, s2, a2
# CHECK-ASM: encoding: [0x3b,0x0e,0xc9,0xb8]
paaddu.h t3, s2, a2
# CHECK-ASM-AND-OBJ: paaddu.w s0, t1, s0
# CHECK-ASM: encoding: [0x3b,0x04,0x83,0xba]
paaddu.w s0, t1, s0
# CHECK-ASM-AND-OBJ: paaddu.b t3, a0, t3
# CHECK-ASM: encoding: [0x3b,0x0e,0xc5,0xbd]
paaddu.b t3, a0, t3
# CHECK-ASM-AND-OBJ: psub.h s0, s2, t3
# CHECK-ASM: encoding: [0x3b,0x04,0xc9,0xc1]
psub.h s0, s2, t3
# CHECK-ASM-AND-OBJ: psub.w t3, a0, s0
# CHECK-ASM: encoding: [0x3b,0x0e,0x85,0xc2]
psub.w t3, a0, s0
# CHECK-ASM-AND-OBJ: psub.b t5, t1, a4
# CHECK-ASM: encoding: [0x3b,0x0f,0xe3,0xc4]
psub.b t5, t1, a4
# CHECK-ASM-AND-OBJ: pdif.h t1, a4, a2
# CHECK-ASM: encoding: [0x3b,0x03,0xc7,0xc8]
pdif.h t1, a4, a2
# CHECK-ASM-AND-OBJ: pdif.b t3, t1, t5
# CHECK-ASM: encoding: [0x3b,0x0e,0xe3,0xcd]
pdif.b t3, t1, t5
# CHECK-ASM-AND-OBJ: pssub.h a0, a2, t3
# CHECK-ASM: encoding: [0x3b,0x05,0xc6,0xd1]
pssub.h a0, a2, t3
# CHECK-ASM-AND-OBJ: pssub.w t3, a4, t1
# CHECK-ASM: encoding: [0x3b,0x0e,0x67,0xd2]
pssub.w t3, a4, t1
# CHECK-ASM-AND-OBJ: pssub.b a2, t5, a4
# CHECK-ASM: encoding: [0x3b,0x06,0xef,0xd4]
pssub.b a2, t5, a4
# CHECK-ASM-AND-OBJ: pasub.h t5, t3, t3
# CHECK-ASM: encoding: [0x3b,0x0f,0xce,0xd9]
pasub.h t5, t3, t3
# CHECK-ASM-AND-OBJ: pasub.w t3, a2, a4
# CHECK-ASM: encoding: [0x3b,0x0e,0xe6,0xda]
pasub.w t3, a2, a4
# CHECK-ASM-AND-OBJ: pasub.b s0, t3, s2
# CHECK-ASM: encoding: [0x3b,0x04,0x2e,0xdd]
pasub.b s0, t3, s2
# CHECK-ASM-AND-OBJ: pdifu.h t5, s0, a4
# CHECK-ASM: encoding: [0x3b,0x0f,0xe4,0xe8]
pdifu.h t5, s0, a4
# CHECK-ASM-AND-OBJ: pdifu.b t3, a0, t5
# CHECK-ASM: encoding: [0x3b,0x0e,0xe5,0xed]
pdifu.b t3, a0, t5
# CHECK-ASM-AND-OBJ: pssubu.h t3, s2, a0
# CHECK-ASM: encoding: [0x3b,0x0e,0xa9,0xf0]
pssubu.h t3, s2, a0
# CHECK-ASM-AND-OBJ: pssubu.w a0, a4, t3
# CHECK-ASM: encoding: [0x3b,0x05,0xc7,0xf3]
pssubu.w a0, a4, t3
# CHECK-ASM-AND-OBJ: pssubu.b t3, a4, t3
# CHECK-ASM: encoding: [0x3b,0x0e,0xc7,0xf5]
pssubu.b t3, a4, t3
# CHECK-ASM-AND-OBJ: pasubu.h a2, s0, t5
# CHECK-ASM: encoding: [0x3b,0x06,0xe4,0xf9]
pasubu.h a2, s0, t5
# CHECK-ASM-AND-OBJ: pasubu.w a0, t3, a4
# CHECK-ASM: encoding: [0x3b,0x05,0xee,0xfa]
pasubu.w a0, t3, a4
# CHECK-ASM-AND-OBJ: pasubu.b s0, t5, a4
# CHECK-ASM: encoding: [0x3b,0x04,0xef,0xfc]
pasubu.b s0, t5, a4
# CHECK-ASM-AND-OBJ: slx a0, a2, s2
# CHECK-ASM: encoding: [0x3b,0x15,0x26,0x8f]
slx a0, a2, s2
# CHECK-ASM-AND-OBJ: pmul.h.b01 a2, a4, a2
# CHECK-ASM: encoding: [0x3b,0x16,0xc7,0x90]
pmul.h.b01 a2, a4, a2
# CHECK-ASM-AND-OBJ: pmul.w.h01 s2, t5, t3
# CHECK-ASM: encoding: [0x3b,0x19,0xcf,0x93]
pmul.w.h01 s2, t5, t3
# CHECK-ASM-AND-OBJ: mul.w01 t5, a4, t1
# CHECK-ASM: encoding: [0x3b,0x1f,0x67,0x96]
mul.w01 t5, a4, t1
# CHECK-ASM-AND-OBJ: pmacc.w.h01 t1, t1, a0
# CHECK-ASM: encoding: [0x3b,0x13,0xa3,0x9a]
pmacc.w.h01 t1, t1, a0
# CHECK-ASM-AND-OBJ: macc.w01 s2, a0, t3
# CHECK-ASM: encoding: [0x3b,0x19,0xc5,0x9f]
macc.w01 s2, a0, t3
# CHECK-ASM-AND-OBJ: mvm s0, t1, a2
# CHECK-ASM: encoding: [0x3b,0x14,0xc3,0xa8]
mvm s0, t1, a2
# CHECK-ASM-AND-OBJ: mvmn a2, a4, a0
# CHECK-ASM: encoding: [0x3b,0x16,0xa7,0xaa]
mvmn a2, a4, a0
# CHECK-ASM-AND-OBJ: merge a4, a2, a2
# CHECK-ASM: encoding: [0x3b,0x17,0xc6,0xac]
merge a4, a2, a2
# CHECK-ASM-AND-OBJ: srx t1, t3, a4
# CHECK-ASM: encoding: [0x3b,0x13,0xee,0xae]
srx t1, t3, a4
# CHECK-ASM-AND-OBJ: pmulu.h.b01 s2, a4, a0
# CHECK-ASM: encoding: [0x3b,0x19,0xa7,0xb0]
pmulu.h.b01 s2, a4, a0
# CHECK-ASM-AND-OBJ: pmulu.w.h01 t1, a4, a2
# CHECK-ASM: encoding: [0x3b,0x13,0xc7,0xb2]
pmulu.w.h01 t1, a4, a2
# CHECK-ASM-AND-OBJ: pdifsumu.b t3, t5, t3
# CHECK-ASM: encoding: [0x3b,0x1e,0xcf,0xb5]
pdifsumu.b t3, t5, t3
# CHECK-ASM-AND-OBJ: mulu.w01 t5, t1, t5
# CHECK-ASM: encoding: [0x3b,0x1f,0xe3,0xb7]
mulu.w01 t5, t1, t5
# CHECK-ASM-AND-OBJ: pmaccu.w.h01 t5, t5, a4
# CHECK-ASM: encoding: [0x3b,0x1f,0xef,0xba]
pmaccu.w.h01 t5, t5, a4
# CHECK-ASM-AND-OBJ: pdifsumau.b s2, a2, a0
# CHECK-ASM: encoding: [0x3b,0x19,0xa6,0xbc]
pdifsumau.b s2, a2, a0
# CHECK-ASM-AND-OBJ: maccu.w01 a0, a0, t1
# CHECK-ASM: encoding: [0x3b,0x15,0x65,0xbe]
maccu.w01 a0, a0, t1
# CHECK-ASM-AND-OBJ: psh1add.h a2, a2, t3
# CHECK-ASM: encoding: [0x3b,0x26,0xc6,0xa1]
psh1add.h a2, a2, t3
# CHECK-ASM-AND-OBJ: pssh1sadd.h a2, t3, t3
# CHECK-ASM: encoding: [0x3b,0x26,0xce,0xb1]
pssh1sadd.h a2, t3, t3
# CHECK-ASM-AND-OBJ: psh1add.w s2, t5, a2
# CHECK-ASM: encoding: [0x3b,0x29,0xcf,0xa2]
psh1add.w s2, t5, a2
# CHECK-ASM-AND-OBJ: pssh1sadd.w a4, t3, s0
# CHECK-ASM: encoding: [0x3b,0x27,0x8e,0xb2]
pssh1sadd.w a4, t3, s0
# CHECK-ASM-AND-OBJ: unzip8p a4, t3, t1
# CHECK-ASM: encoding: [0x3b,0x27,0x6e,0xe0]
unzip8p a4, t3, t1
# CHECK-ASM-AND-OBJ: unzip16p t5, a4, t3
# CHECK-ASM: encoding: [0x3b,0x2f,0xc7,0xe3]
unzip16p t5, a4, t3
# CHECK-ASM-AND-OBJ: unzip8hp s0, a0, t1
# CHECK-ASM: encoding: [0x3b,0x24,0x65,0xe4]
unzip8hp s0, a0, t1
# CHECK-ASM-AND-OBJ: unzip16hp a0, a0, a2
# CHECK-ASM: encoding: [0x3b,0x25,0xc5,0xe6]
unzip16hp a0, a0, a2
# CHECK-ASM-AND-OBJ: zip8p t5, t3, t3
# CHECK-ASM: encoding: [0x3b,0x2f,0xce,0xf1]
zip8p t5, t3, t3
# CHECK-ASM-AND-OBJ: zip16p a0, t5, a0
# CHECK-ASM: encoding: [0x3b,0x25,0xaf,0xf2]
zip16p a0, t5, a0
# CHECK-ASM-AND-OBJ: zip8hp t5, a0, a2
# CHECK-ASM: encoding: [0x3b,0x2f,0xc5,0xf4]
zip8hp t5, a0, a2
# CHECK-ASM-AND-OBJ: zip16hp t1, t5, a4
# CHECK-ASM: encoding: [0x3b,0x23,0xef,0xf6]
zip16hp t1, t5, a4
# CHECK-ASM-AND-OBJ: pmul.h.b00 a4, a4, s2
# CHECK-ASM: encoding: [0x3b,0x37,0x27,0x81]
pmul.h.b00 a4, a4, s2
# CHECK-ASM-AND-OBJ: pmul.w.h00 s2, t1, a2
# CHECK-ASM: encoding: [0x3b,0x39,0xc3,0x82]
pmul.w.h00 s2, t1, a2
# CHECK-ASM-AND-OBJ: mul.w00 a4, a0, a2
# CHECK-ASM: encoding: [0x3b,0x37,0xc5,0x86]
mul.w00 a4, a0, a2
# CHECK-ASM-AND-OBJ: pmacc.w.h00 s2, t5, a2
# CHECK-ASM: encoding: [0x3b,0x39,0xcf,0x8a]
pmacc.w.h00 s2, t5, a2
# CHECK-ASM-AND-OBJ: macc.w00 t1, a0, t5
# CHECK-ASM: encoding: [0x3b,0x33,0xe5,0x8f]
macc.w00 t1, a0, t5
# CHECK-ASM-AND-OBJ: pmul.h.b11 t1, s2, s0
# CHECK-ASM: encoding: [0x3b,0x33,0x89,0x90]
pmul.h.b11 t1, s2, s0
# CHECK-ASM-AND-OBJ: pmul.w.h11 s0, a4, s0
# CHECK-ASM: encoding: [0x3b,0x34,0x87,0x92]
pmul.w.h11 s0, a4, s0
# CHECK-ASM-AND-OBJ: mul.w11 a0, t3, a0
# CHECK-ASM: encoding: [0x3b,0x35,0xae,0x96]
mul.w11 a0, t3, a0
# CHECK-ASM-AND-OBJ: pmacc.w.h11 a4, a4, t1
# CHECK-ASM: encoding: [0x3b,0x37,0x67,0x9a]
pmacc.w.h11 a4, a4, t1
# CHECK-ASM-AND-OBJ: macc.w11 t3, s2, a2
# CHECK-ASM: encoding: [0x3b,0x3e,0xc9,0x9e]
macc.w11 t3, s2, a2
# CHECK-ASM-AND-OBJ: pmulu.h.b00 a4, a2, a0
# CHECK-ASM: encoding: [0x3b,0x37,0xa6,0xa0]
pmulu.h.b00 a4, a2, a0
# CHECK-ASM-AND-OBJ: pmulu.w.h00 a2, t3, t1
# CHECK-ASM: encoding: [0x3b,0x36,0x6e,0xa2]
pmulu.w.h00 a2, t3, t1
# CHECK-ASM-AND-OBJ: mulu.w00 a0, t5, s2
# CHECK-ASM: encoding: [0x3b,0x35,0x2f,0xa7]
mulu.w00 a0, t5, s2
# CHECK-ASM-AND-OBJ: pmaccu.w.h00 t3, t3, t5
# CHECK-ASM: encoding: [0x3b,0x3e,0xee,0xab]
pmaccu.w.h00 t3, t3, t5
# CHECK-ASM-AND-OBJ: maccu.w00 s2, t1, s2
# CHECK-ASM: encoding: [0x3b,0x39,0x23,0xaf]
maccu.w00 s2, t1, s2
# CHECK-ASM-AND-OBJ: pmulu.h.b11 a4, s0, a4
# CHECK-ASM: encoding: [0x3b,0x37,0xe4,0xb0]
pmulu.h.b11 a4, s0, a4
# CHECK-ASM-AND-OBJ: pmulu.w.h11 s0, t5, t5
# CHECK-ASM: encoding: [0x3b,0x34,0xef,0xb3]
pmulu.w.h11 s0, t5, t5
# CHECK-ASM-AND-OBJ: mulu.w11 s0, t1, a4
# CHECK-ASM: encoding: [0x3b,0x34,0xe3,0xb6]
mulu.w11 s0, t1, a4
# CHECK-ASM-AND-OBJ: pmaccu.w.h11 a0, s0, t5
# CHECK-ASM: encoding: [0x3b,0x35,0xe4,0xbb]
pmaccu.w.h11 a0, s0, t5
# CHECK-ASM-AND-OBJ: maccu.w11 s2, t3, t5
# CHECK-ASM: encoding: [0x3b,0x39,0xee,0xbf]
maccu.w11 s2, t3, t5
# CHECK-ASM-AND-OBJ: pmulsu.h.b00 a2, s0, a4
# CHECK-ASM: encoding: [0x3b,0x36,0xe4,0xe0]
pmulsu.h.b00 a2, s0, a4
# CHECK-ASM-AND-OBJ: pmulsu.w.h00 t5, t5, t5
# CHECK-ASM: encoding: [0x3b,0x3f,0xef,0xe3]
pmulsu.w.h00 t5, t5, t5
# CHECK-ASM-AND-OBJ: mulsu.w00 t1, s0, a4
# CHECK-ASM: encoding: [0x3b,0x33,0xe4,0xe6]
mulsu.w00 t1, s0, a4
# CHECK-ASM-AND-OBJ: pmulsu.h.b11 t3, t1, a4
# CHECK-ASM: encoding: [0x3b,0x3e,0xe3,0xf0]
pmulsu.h.b11 t3, t1, a4
# CHECK-ASM-AND-OBJ: pmulsu.w.h11 t1, t3, s2
# CHECK-ASM: encoding: [0x3b,0x33,0x2e,0xf3]
pmulsu.w.h11 t1, t3, s2
# CHECK-ASM-AND-OBJ: mulsu.w11 a2, s2, a2
# CHECK-ASM: encoding: [0x3b,0x36,0xc9,0xf6]
mulsu.w11 a2, s2, a2
# CHECK-ASM-AND-OBJ: pmaccsu.w.h00 a4, a0, t1
# CHECK-ASM: encoding: [0x3b,0x37,0x65,0xea]
pmaccsu.w.h00 a4, a0, t1
# CHECK-ASM-AND-OBJ: maccsu.w00 a4, s2, s0
# CHECK-ASM: encoding: [0x3b,0x37,0x89,0xee]
maccsu.w00 a4, s2, s0
# CHECK-ASM-AND-OBJ: pmaccsu.w.h11 a0, a2, t3
# CHECK-ASM: encoding: [0x3b,0x35,0xc6,0xfb]
pmaccsu.w.h11 a0, a2, t3
# CHECK-ASM-AND-OBJ: maccsu.w11 t5, a4, s2
# CHECK-ASM: encoding: [0x3b,0x3f,0x27,0xff]
maccsu.w11 t5, a4, s2
# CHECK-ASM-AND-OBJ: ppack.h s0, s0, s2
# CHECK-ASM: encoding: [0x3b,0x44,0x24,0x81]
ppack.h s0, s0, s2
# CHECK-ASM-AND-OBJ: ppack.w t5, a2, a4
# CHECK-ASM: encoding: [0x3b,0x4f,0xe6,0x82]
ppack.w t5, a2, a4
# CHECK-ASM-AND-OBJ: ppackbt.h a4, s2, t3
# CHECK-ASM: encoding: [0x3b,0x47,0xc9,0x91]
ppackbt.h a4, s2, t3
# CHECK-ASM-AND-OBJ: ppackbt.w t5, s0, t5
# CHECK-ASM: encoding: [0x3b,0x4f,0xe4,0x93]
ppackbt.w t5, s0, t5
# CHECK-ASM-AND-OBJ: packbt a0, t5, a0
# CHECK-ASM: encoding: [0x3b,0x45,0xaf,0x96]
packbt a0, t5, a0
# CHECK-ASM-AND-OBJ: ppacktb.h t3, s0, t3
# CHECK-ASM: encoding: [0x3b,0x4e,0xc4,0xa1]
ppacktb.h t3, s0, t3
# CHECK-ASM-AND-OBJ: ppacktb.w t5, t1, t1
# CHECK-ASM: encoding: [0x3b,0x4f,0x63,0xa2]
ppacktb.w t5, t1, t1
# CHECK-ASM-AND-OBJ: packtb t5, a4, a4
# CHECK-ASM: encoding: [0x3b,0x4f,0xe7,0xa6]
packtb t5, a4, a4
# CHECK-ASM-AND-OBJ: ppackt.h a0, t1, t5
# CHECK-ASM: encoding: [0x3b,0x45,0xe3,0xb1]
ppackt.h a0, t1, t5
# CHECK-ASM-AND-OBJ: ppackt.w t3, a0, s2
# CHECK-ASM: encoding: [0x3b,0x4e,0x25,0xb3]
ppackt.w t3, a0, s2
# CHECK-ASM-AND-OBJ: packt a2, t3, t1
# CHECK-ASM: encoding: [0x3b,0x46,0x6e,0xb6]
packt a2, t3, t1
# CHECK-ASM-AND-OBJ: pm2add.h a4, t3, t5
# CHECK-ASM: encoding: [0x3b,0x57,0xee,0x81]
pm2add.h a4, t3, t5
# CHECK-ASM-AND-OBJ: pm4add.b t1, s2, s2
# CHECK-ASM: encoding: [0x3b,0x53,0x29,0x85]
pm4add.b t1, s2, s2
# CHECK-ASM-AND-OBJ: pm2adda.h a0, t5, s0
# CHECK-ASM: encoding: [0x3b,0x55,0x8f,0x88]
pm2adda.h a0, t5, s0
# CHECK-ASM-AND-OBJ: pm4adda.b a0, t5, a2
# CHECK-ASM: encoding: [0x3b,0x55,0xcf,0x8c]
pm4adda.b a0, t5, a2
# CHECK-ASM-AND-OBJ: pm2add.hx a0, s0, t3
# CHECK-ASM: encoding: [0x3b,0x55,0xc4,0x91]
pm2add.hx a0, s0, t3
# CHECK-ASM-AND-OBJ: pm2adda.hx s0, a0, s2
# CHECK-ASM: encoding: [0x3b,0x54,0x25,0x99]
pm2adda.hx s0, a0, s2
# CHECK-ASM-AND-OBJ: pm2addu.h t1, a4, a0
# CHECK-ASM: encoding: [0x3b,0x53,0xa7,0xa0]
pm2addu.h t1, a4, a0
# CHECK-ASM-AND-OBJ: pm4addu.b a0, t1, t3
# CHECK-ASM: encoding: [0x3b,0x55,0xc3,0xa5]
pm4addu.b a0, t1, t3
# CHECK-ASM-AND-OBJ: pm2addau.h s2, a2, a0
# CHECK-ASM: encoding: [0x3b,0x59,0xa6,0xa8]
pm2addau.h s2, a2, a0
# CHECK-ASM-AND-OBJ: pm4addau.b s2, s2, t5
# CHECK-ASM: encoding: [0x3b,0x59,0xe9,0xad]
pm4addau.b s2, s2, t5
# CHECK-ASM-AND-OBJ: pmq2add.h s2, s2, t3
# CHECK-ASM: encoding: [0x3b,0x59,0xc9,0xb1]
pmq2add.h s2, s2, t3
# CHECK-ASM-AND-OBJ: pmqr2add.h a4, a2, a0
# CHECK-ASM: encoding: [0x3b,0x57,0xa6,0xb4]
pmqr2add.h a4, a2, a0
# CHECK-ASM-AND-OBJ: pmq2adda.h a2, s2, t1
# CHECK-ASM: encoding: [0x3b,0x56,0x69,0xb8]
pmq2adda.h a2, s2, t1
# CHECK-ASM-AND-OBJ: pmqr2adda.h a2, a0, a0
# CHECK-ASM: encoding: [0x3b,0x56,0xa5,0xbc]
pmqr2adda.h a2, a0, a0
# CHECK-ASM-AND-OBJ: pm2sub.h t5, s0, s0
# CHECK-ASM: encoding: [0x3b,0x5f,0x84,0xc0]
pm2sub.h t5, s0, s0
# CHECK-ASM-AND-OBJ: pm2sadd.h a0, t5, a0
# CHECK-ASM: encoding: [0x3b,0x55,0xaf,0xc4]
pm2sadd.h a0, t5, a0
# CHECK-ASM-AND-OBJ: pm2suba.h s0, a4, t5
# CHECK-ASM: encoding: [0x3b,0x54,0xe7,0xc9]
pm2suba.h s0, a4, t5
# CHECK-ASM-AND-OBJ: pm2sub.hx t3, t5, a0
# CHECK-ASM: encoding: [0x3b,0x5e,0xaf,0xd0]
pm2sub.hx t3, t5, a0
# CHECK-ASM-AND-OBJ: pm2sadd.hx t1, a0, t3
# CHECK-ASM: encoding: [0x3b,0x53,0xc5,0xd5]
pm2sadd.hx t1, a0, t3
# CHECK-ASM-AND-OBJ: pm2suba.hx t3, a0, a4
# CHECK-ASM: encoding: [0x3b,0x5e,0xe5,0xd8]
pm2suba.hx t3, a0, a4
# CHECK-ASM-AND-OBJ: pm2addsu.h t3, a0, t3
# CHECK-ASM: encoding: [0x3b,0x5e,0xc5,0xe1]
pm2addsu.h t3, a0, t3
# CHECK-ASM-AND-OBJ: pm4addsu.b t1, t5, s0
# CHECK-ASM: encoding: [0x3b,0x53,0x8f,0xe4]
pm4addsu.b t1, t5, s0
# CHECK-ASM-AND-OBJ: pm2addasu.h t1, t3, t3
# CHECK-ASM: encoding: [0x3b,0x53,0xce,0xe9]
pm2addasu.h t1, t3, t3
# CHECK-ASM-AND-OBJ: pm4addasu.b t3, a0, a0
# CHECK-ASM: encoding: [0x3b,0x5e,0xa5,0xec]
pm4addasu.b t3, a0, a0
# CHECK-ASM-AND-OBJ: pm2add.w t3, s0, t5
# CHECK-ASM: encoding: [0x3b,0x5e,0xe4,0x83]
pm2add.w t3, s0, t5
# CHECK-ASM-AND-OBJ: pm4add.h s2, s2, t3
# CHECK-ASM: encoding: [0x3b,0x59,0xc9,0x87]
pm4add.h s2, s2, t3
# CHECK-ASM-AND-OBJ: pm2adda.w a2, a0, s0
# CHECK-ASM: encoding: [0x3b,0x56,0x85,0x8a]
pm2adda.w a2, a0, s0
# CHECK-ASM-AND-OBJ: pm4adda.h s2, s0, t1
# CHECK-ASM: encoding: [0x3b,0x59,0x64,0x8e]
pm4adda.h s2, s0, t1
# CHECK-ASM-AND-OBJ: pm2add.wx t1, s0, a4
# CHECK-ASM: encoding: [0x3b,0x53,0xe4,0x92]
pm2add.wx t1, s0, a4
# CHECK-ASM-AND-OBJ: pm2adda.wx t5, t3, s0
# CHECK-ASM: encoding: [0x3b,0x5f,0x8e,0x9a]
pm2adda.wx t5, t3, s0
# CHECK-ASM-AND-OBJ: pm2addu.w s2, a0, s2
# CHECK-ASM: encoding: [0x3b,0x59,0x25,0xa3]
pm2addu.w s2, a0, s2
# CHECK-ASM-AND-OBJ: pm4addu.h a4, a2, a2
# CHECK-ASM: encoding: [0x3b,0x57,0xc6,0xa6]
pm4addu.h a4, a2, a2
# CHECK-ASM-AND-OBJ: pm2addau.w s0, s0, a4
# CHECK-ASM: encoding: [0x3b,0x54,0xe4,0xaa]
pm2addau.w s0, s0, a4
# CHECK-ASM-AND-OBJ: pm4addau.h a2, a4, t5
# CHECK-ASM: encoding: [0x3b,0x56,0xe7,0xaf]
pm4addau.h a2, a4, t5
# CHECK-ASM-AND-OBJ: pmq2add.w t5, t1, t1
# CHECK-ASM: encoding: [0x3b,0x5f,0x63,0xb2]
pmq2add.w t5, t1, t1
# CHECK-ASM-AND-OBJ: pmqr2add.w s0, t1, a4
# CHECK-ASM: encoding: [0x3b,0x54,0xe3,0xb6]
pmqr2add.w s0, t1, a4
# CHECK-ASM-AND-OBJ: pmq2adda.w a4, s0, t1
# CHECK-ASM: encoding: [0x3b,0x57,0x64,0xba]
pmq2adda.w a4, s0, t1
# CHECK-ASM-AND-OBJ: pmqr2adda.w a4, t5, s0
# CHECK-ASM: encoding: [0x3b,0x57,0x8f,0xbe]
pmqr2adda.w a4, t5, s0
# CHECK-ASM-AND-OBJ: pm2sub.w t3, a2, t1
# CHECK-ASM: encoding: [0x3b,0x5e,0x66,0xc2]
pm2sub.w t3, a2, t1
# CHECK-ASM-AND-OBJ: pm2suba.w t5, t5, t3
# CHECK-ASM: encoding: [0x3b,0x5f,0xcf,0xcb]
pm2suba.w t5, t5, t3
# CHECK-ASM-AND-OBJ: pm2sub.wx t5, s2, s0
# CHECK-ASM: encoding: [0x3b,0x5f,0x89,0xd2]
pm2sub.wx t5, s2, s0
# CHECK-ASM-AND-OBJ: pm2suba.wx a2, a0, a4
# CHECK-ASM: encoding: [0x3b,0x56,0xe5,0xda]
pm2suba.wx a2, a0, a4
# CHECK-ASM-AND-OBJ: pm2addsu.w s0, s0, s2
# CHECK-ASM: encoding: [0x3b,0x54,0x24,0xe3]
pm2addsu.w s0, s0, s2
# CHECK-ASM-AND-OBJ: pm4addsu.h a2, s2, s0
# CHECK-ASM: encoding: [0x3b,0x56,0x89,0xe6]
pm4addsu.h a2, s2, s0
# CHECK-ASM-AND-OBJ: pm2addasu.w a0, a2, a0
# CHECK-ASM: encoding: [0x3b,0x55,0xa6,0xea]
pm2addasu.w a0, a2, a0
# CHECK-ASM-AND-OBJ: pm4addasu.h a0, s0, t5
# CHECK-ASM: encoding: [0x3b,0x55,0xe4,0xef]
pm4addasu.h a0, s0, t5
# CHECK-ASM-AND-OBJ: pmqacc.w.h01 t5, t1, a4
# CHECK-ASM: encoding: [0x3b,0x5f,0xe3,0xf8]
pmqacc.w.h01 t5, t1, a4
# CHECK-ASM-AND-OBJ: mqacc.w01 a0, a2, t3
# CHECK-ASM: encoding: [0x3b,0x55,0xc6,0xfb]
mqacc.w01 a0, a2, t3
# CHECK-ASM-AND-OBJ: pmqracc.w.h01 a4, t1, s2
# CHECK-ASM: encoding: [0x3b,0x57,0x23,0xfd]
pmqracc.w.h01 a4, t1, s2
# CHECK-ASM-AND-OBJ: mqracc.w01 s0, t5, a4
# CHECK-ASM: encoding: [0x3b,0x54,0xef,0xfe]
mqracc.w01 s0, t5, a4
# CHECK-ASM-AND-OBJ: pas.hx a0, t5, t1
# CHECK-ASM: encoding: [0x3b,0x65,0x6f,0x80]
pas.hx a0, t5, t1
# CHECK-ASM-AND-OBJ: psa.hx a2, t3, a0
# CHECK-ASM: encoding: [0x3b,0x66,0xae,0x84]
psa.hx a2, t3, a0
# CHECK-ASM-AND-OBJ: psas.hx s0, a0, a2
# CHECK-ASM: encoding: [0x3b,0x64,0xc5,0x90]
psas.hx s0, a0, a2
# CHECK-ASM-AND-OBJ: pssa.hx a0, t1, t5
# CHECK-ASM: encoding: [0x3b,0x65,0xe3,0x95]
pssa.hx a0, t1, t5
# CHECK-ASM-AND-OBJ: paas.hx t1, a2, s0
# CHECK-ASM: encoding: [0x3b,0x63,0x86,0x98]
paas.hx t1, a2, s0
# CHECK-ASM-AND-OBJ: pasa.hx t1, s2, t3
# CHECK-ASM: encoding: [0x3b,0x63,0xc9,0x9d]
pasa.hx t1, s2, t3
# CHECK-ASM-AND-OBJ: pmseq.h t3, s0, t1
# CHECK-ASM: encoding: [0x3b,0x6e,0x64,0xc0]
pmseq.h t3, s0, t1
# CHECK-ASM-AND-OBJ: pmseq.b t5, s2, a2
# CHECK-ASM: encoding: [0x3b,0x6f,0xc9,0xc4]
pmseq.b t5, s2, a2
# CHECK-ASM-AND-OBJ: pmslt.h t1, a0, a4
# CHECK-ASM: encoding: [0x3b,0x63,0xe5,0xd0]
pmslt.h t1, a0, a4
# CHECK-ASM-AND-OBJ: pmslt.b s2, t3, t1
# CHECK-ASM: encoding: [0x3b,0x69,0x6e,0xd4]
pmslt.b s2, t3, t1
# CHECK-ASM-AND-OBJ: pmsltu.h t1, a0, t5
# CHECK-ASM: encoding: [0x3b,0x63,0xe5,0xd9]
pmsltu.h t1, a0, t5
# CHECK-ASM-AND-OBJ: pmsltu.b t3, a4, s2
# CHECK-ASM: encoding: [0x3b,0x6e,0x27,0xdd]
pmsltu.b t3, a4, s2
# CHECK-ASM-AND-OBJ: pmin.h a2, a2, s2
# CHECK-ASM: encoding: [0x3b,0x66,0x26,0xe1]
pmin.h a2, a2, s2
# CHECK-ASM-AND-OBJ: pmin.b t3, a2, a0
# CHECK-ASM: encoding: [0x3b,0x6e,0xa6,0xe4]
pmin.b t3, a2, a0
# CHECK-ASM-AND-OBJ: pminu.h a2, s2, t1
# CHECK-ASM: encoding: [0x3b,0x66,0x69,0xe8]
pminu.h a2, s2, t1
# CHECK-ASM-AND-OBJ: pminu.b a0, t1, a0
# CHECK-ASM: encoding: [0x3b,0x65,0xa3,0xec]
pminu.b a0, t1, a0
# CHECK-ASM-AND-OBJ: pmax.h a0, s0, a4
# CHECK-ASM: encoding: [0x3b,0x65,0xe4,0xf0]
pmax.h a0, s0, a4
# CHECK-ASM-AND-OBJ: pmax.b t1, t3, a2
# CHECK-ASM: encoding: [0x3b,0x63,0xce,0xf4]
pmax.b t1, t3, a2
# CHECK-ASM-AND-OBJ: pmaxu.h t1, a0, s2
# CHECK-ASM: encoding: [0x3b,0x63,0x25,0xf9]
pmaxu.h t1, a0, s2
# CHECK-ASM-AND-OBJ: pmaxu.b t1, t3, s2
# CHECK-ASM: encoding: [0x3b,0x63,0x2e,0xfd]
pmaxu.b t1, t3, s2
# CHECK-ASM-AND-OBJ: pas.wx t1, a2, s0
# CHECK-ASM: encoding: [0x3b,0x63,0x86,0x82]
pas.wx t1, a2, s0
# CHECK-ASM-AND-OBJ: psa.wx t3, t3, a2
# CHECK-ASM: encoding: [0x3b,0x6e,0xce,0x86]
psa.wx t3, t3, a2
# CHECK-ASM-AND-OBJ: psas.wx a4, a4, s2
# CHECK-ASM: encoding: [0x3b,0x67,0x27,0x93]
psas.wx a4, a4, s2
# CHECK-ASM-AND-OBJ: pssa.wx a0, a2, a0
# CHECK-ASM: encoding: [0x3b,0x65,0xa6,0x96]
pssa.wx a0, a2, a0
# CHECK-ASM-AND-OBJ: paas.wx s0, a4, t1
# CHECK-ASM: encoding: [0x3b,0x64,0x67,0x9a]
paas.wx s0, a4, t1
# CHECK-ASM-AND-OBJ: pasa.wx t1, s2, s2
# CHECK-ASM: encoding: [0x3b,0x63,0x29,0x9f]
pasa.wx t1, s2, s2
# CHECK-ASM-AND-OBJ: pmseq.w t1, a4, s2
# CHECK-ASM: encoding: [0x3b,0x63,0x27,0xc3]
pmseq.w t1, a4, s2
# CHECK-ASM-AND-OBJ: pmslt.w t5, t5, t3
# CHECK-ASM: encoding: [0x3b,0x6f,0xcf,0xd3]
pmslt.w t5, t5, t3
# CHECK-ASM-AND-OBJ: pmsltu.w s2, a4, t1
# CHECK-ASM: encoding: [0x3b,0x69,0x67,0xda]
pmsltu.w s2, a4, t1
# CHECK-ASM-AND-OBJ: pmin.w t5, a4, t5
# CHECK-ASM: encoding: [0x3b,0x6f,0xe7,0xe3]
pmin.w t5, a4, t5
# CHECK-ASM-AND-OBJ: pminu.w a0, a2, a0
# CHECK-ASM: encoding: [0x3b,0x65,0xa6,0xea]
pminu.w a0, a2, a0
# CHECK-ASM-AND-OBJ: pmax.w a0, s2, t1
# CHECK-ASM: encoding: [0x3b,0x65,0x69,0xf2]
pmax.w a0, s2, t1
# CHECK-ASM-AND-OBJ: pmaxu.w a0, a4, a4
# CHECK-ASM: encoding: [0x3b,0x65,0xe7,0xfa]
pmaxu.w a0, a4, a4
# CHECK-ASM-AND-OBJ: pmulh.h a0, t5, t1
# CHECK-ASM: encoding: [0x3b,0x75,0x6f,0x80]
pmulh.h a0, t5, t1
# CHECK-ASM-AND-OBJ: pmulhr.h s2, t5, t3
# CHECK-ASM: encoding: [0x3b,0x79,0xcf,0x85]
pmulhr.h s2, t5, t3
# CHECK-ASM-AND-OBJ: pmhacc.h t5, t3, a4
# CHECK-ASM: encoding: [0x3b,0x7f,0xee,0x88]
pmhacc.h t5, t3, a4
# CHECK-ASM-AND-OBJ: pmhracc.h s2, t5, s2
# CHECK-ASM: encoding: [0x3b,0x79,0x2f,0x8d]
pmhracc.h s2, t5, s2
# CHECK-ASM-AND-OBJ: pmulhu.h t3, t1, t1
# CHECK-ASM: encoding: [0x3b,0x7e,0x63,0x90]
pmulhu.h t3, t1, t1
# CHECK-ASM-AND-OBJ: pmulhru.h s0, t5, t1
# CHECK-ASM: encoding: [0x3b,0x74,0x6f,0x94]
pmulhru.h s0, t5, t1
# CHECK-ASM-AND-OBJ: pmhaccu.h t3, a0, t3
# CHECK-ASM: encoding: [0x3b,0x7e,0xc5,0x99]
pmhaccu.h t3, a0, t3
# CHECK-ASM-AND-OBJ: pmhraccu.h t5, t3, a2
# CHECK-ASM: encoding: [0x3b,0x7f,0xce,0x9c]
pmhraccu.h t5, t3, a2
# CHECK-ASM-AND-OBJ: pmulh.h.b0 t1, a0, a0
# CHECK-ASM: encoding: [0x3b,0x73,0xa5,0xa0]
pmulh.h.b0 t1, a0, a0
# CHECK-ASM-AND-OBJ: pmulhsu.h.b0 t3, s0, a4
# CHECK-ASM: encoding: [0x3b,0x7e,0xe4,0xa4]
pmulhsu.h.b0 t3, s0, a4
# CHECK-ASM-AND-OBJ: pmhacc.h.b0 t1, a0, a4
# CHECK-ASM: encoding: [0x3b,0x73,0xe5,0xa8]
pmhacc.h.b0 t1, a0, a4
# CHECK-ASM-AND-OBJ: pmhaccsu.h.b0 s2, t5, t3
# CHECK-ASM: encoding: [0x3b,0x79,0xcf,0xad]
pmhaccsu.h.b0 s2, t5, t3
# CHECK-ASM-AND-OBJ: pmulh.h.b1 a0, s0, a2
# CHECK-ASM: encoding: [0x3b,0x75,0xc4,0xb0]
pmulh.h.b1 a0, s0, a2
# CHECK-ASM-AND-OBJ: pmulhsu.h.b1 t1, t3, t3
# CHECK-ASM: encoding: [0x3b,0x73,0xce,0xb5]
pmulhsu.h.b1 t1, t3, t3
# CHECK-ASM-AND-OBJ: pmhacc.h.b1 t3, t5, s2
# CHECK-ASM: encoding: [0x3b,0x7e,0x2f,0xb9]
pmhacc.h.b1 t3, t5, s2
# CHECK-ASM-AND-OBJ: pmhaccsu.h.b1 t5, t5, t1
# CHECK-ASM: encoding: [0x3b,0x7f,0x6f,0xbc]
pmhaccsu.h.b1 t5, t5, t1
# CHECK-ASM-AND-OBJ: pmulhsu.h s2, t3, a4
# CHECK-ASM: encoding: [0x3b,0x79,0xee,0xc0]
pmulhsu.h s2, t3, a4
# CHECK-ASM-AND-OBJ: pmulhrsu.h a0, a0, t5
# CHECK-ASM: encoding: [0x3b,0x75,0xe5,0xc5]
pmulhrsu.h a0, a0, t5
# CHECK-ASM-AND-OBJ: pmhaccsu.h s0, t3, t1
# CHECK-ASM: encoding: [0x3b,0x74,0x6e,0xc8]
pmhaccsu.h s0, t3, t1
# CHECK-ASM-AND-OBJ: pmhraccsu.h s0, t5, a4
# CHECK-ASM: encoding: [0x3b,0x74,0xef,0xcc]
pmhraccsu.h s0, t5, a4
# CHECK-ASM-AND-OBJ: pmulq.h t3, t1, s0
# CHECK-ASM: encoding: [0x3b,0x7e,0x83,0xd0]
pmulq.h t3, t1, s0
# CHECK-ASM-AND-OBJ: pmulqr.h t1, s2, s0
# CHECK-ASM: encoding: [0x3b,0x73,0x89,0xd4]
pmulqr.h t1, s2, s0
# CHECK-ASM-AND-OBJ: pmulh.w t5, a4, a4
# CHECK-ASM: encoding: [0x3b,0x7f,0xe7,0x82]
pmulh.w t5, a4, a4
# CHECK-ASM-AND-OBJ: pmulhr.w t1, t5, t1
# CHECK-ASM: encoding: [0x3b,0x73,0x6f,0x86]
pmulhr.w t1, t5, t1
# CHECK-ASM-AND-OBJ: pmhacc.w t5, s0, t5
# CHECK-ASM: encoding: [0x3b,0x7f,0xe4,0x8b]
pmhacc.w t5, s0, t5
# CHECK-ASM-AND-OBJ: pmhracc.w s0, s2, t5
# CHECK-ASM: encoding: [0x3b,0x74,0xe9,0x8f]
pmhracc.w s0, s2, t5
# CHECK-ASM-AND-OBJ: pmulhu.w a2, a0, a4
# CHECK-ASM: encoding: [0x3b,0x76,0xe5,0x92]
pmulhu.w a2, a0, a4
# CHECK-ASM-AND-OBJ: pmulhru.w t1, t1, a4
# CHECK-ASM: encoding: [0x3b,0x73,0xe3,0x96]
pmulhru.w t1, t1, a4
# CHECK-ASM-AND-OBJ: pmhaccu.w a0, s0, a0
# CHECK-ASM: encoding: [0x3b,0x75,0xa4,0x9a]
pmhaccu.w a0, s0, a0
# CHECK-ASM-AND-OBJ: pmhraccu.w s2, s0, t3
# CHECK-ASM: encoding: [0x3b,0x79,0xc4,0x9f]
pmhraccu.w s2, s0, t3
# CHECK-ASM-AND-OBJ: pmulh.w.h0 t5, s0, t5
# CHECK-ASM: encoding: [0x3b,0x7f,0xe4,0xa3]
pmulh.w.h0 t5, s0, t5
# CHECK-ASM-AND-OBJ: pmulhsu.w.h0 a2, t3, a2
# CHECK-ASM: encoding: [0x3b,0x76,0xce,0xa6]
pmulhsu.w.h0 a2, t3, a2
# CHECK-ASM-AND-OBJ: pmhacc.w.h0 a2, a0, t1
# CHECK-ASM: encoding: [0x3b,0x76,0x65,0xaa]
pmhacc.w.h0 a2, a0, t1
# CHECK-ASM-AND-OBJ: pmhaccsu.w.h0 t1, a4, t1
# CHECK-ASM: encoding: [0x3b,0x73,0x67,0xae]
pmhaccsu.w.h0 t1, a4, t1
# CHECK-ASM-AND-OBJ: pmulh.w.h1 t1, a0, t3
# CHECK-ASM: encoding: [0x3b,0x73,0xc5,0xb3]
pmulh.w.h1 t1, a0, t3
# CHECK-ASM-AND-OBJ: pmulhsu.w.h1 s2, t3, a4
# CHECK-ASM: encoding: [0x3b,0x79,0xee,0xb6]
pmulhsu.w.h1 s2, t3, a4
# CHECK-ASM-AND-OBJ: pmhacc.w.h1 s0, t5, a2
# CHECK-ASM: encoding: [0x3b,0x74,0xcf,0xba]
pmhacc.w.h1 s0, t5, a2
# CHECK-ASM-AND-OBJ: pmhaccsu.w.h1 a0, a0, a0
# CHECK-ASM: encoding: [0x3b,0x75,0xa5,0xbe]
pmhaccsu.w.h1 a0, a0, a0
# CHECK-ASM-AND-OBJ: pmulhsu.w t3, a2, a4
# CHECK-ASM: encoding: [0x3b,0x7e,0xe6,0xc2]
pmulhsu.w t3, a2, a4
# CHECK-ASM-AND-OBJ: pmulhrsu.w t5, a0, a0
# CHECK-ASM: encoding: [0x3b,0x7f,0xa5,0xc6]
pmulhrsu.w t5, a0, a0
# CHECK-ASM-AND-OBJ: pmhaccsu.w a4, a0, a0
# CHECK-ASM: encoding: [0x3b,0x77,0xa5,0xca]
pmhaccsu.w a4, a0, a0
# CHECK-ASM-AND-OBJ: pmhraccsu.w t5, t1, t3
# CHECK-ASM: encoding: [0x3b,0x7f,0xc3,0xcf]
pmhraccsu.w t5, t1, t3
# CHECK-ASM-AND-OBJ: pmulq.w a2, a2, t5
# CHECK-ASM: encoding: [0x3b,0x76,0xe6,0xd3]
pmulq.w a2, a2, t5
# CHECK-ASM-AND-OBJ: pmulqr.w a0, t3, t5
# CHECK-ASM: encoding: [0x3b,0x75,0xee,0xd7]
pmulqr.w a0, t3, t5
# CHECK-ASM-AND-OBJ: pmqacc.w.h00 t5, a4, t1
# CHECK-ASM: encoding: [0x3b,0x7f,0x67,0xe8]
pmqacc.w.h00 t5, a4, t1
# CHECK-ASM-AND-OBJ: mqacc.w00 t1, t1, a0
# CHECK-ASM: encoding: [0x3b,0x73,0xa3,0xea]
mqacc.w00 t1, t1, a0
# CHECK-ASM-AND-OBJ: pmqracc.w.h00 t1, a2, t5
# CHECK-ASM: encoding: [0x3b,0x73,0xe6,0xed]
pmqracc.w.h00 t1, a2, t5
# CHECK-ASM-AND-OBJ: mqracc.w00 s2, s2, a2
# CHECK-ASM: encoding: [0x3b,0x79,0xc9,0xee]
mqracc.w00 s2, s2, a2
# CHECK-ASM-AND-OBJ: pmqacc.w.h11 a2, a0, a0
# CHECK-ASM: encoding: [0x3b,0x76,0xa5,0xf8]
pmqacc.w.h11 a2, a0, a0
# CHECK-ASM-AND-OBJ: mqacc.w11 a4, a2, a2
# CHECK-ASM: encoding: [0x3b,0x77,0xc6,0xfa]
mqacc.w11 a4, a2, a2
# CHECK-ASM-AND-OBJ: pmqracc.w.h11 s0, t1, t3
# CHECK-ASM: encoding: [0x3b,0x74,0xc3,0xfd]
pmqracc.w.h11 s0, t1, t3
# CHECK-ASM-AND-OBJ: mqracc.w11 s2, t1, a4
# CHECK-ASM: encoding: [0x3b,0x79,0xe3,0xfe]
mqracc.w11 s2, t1, a4
