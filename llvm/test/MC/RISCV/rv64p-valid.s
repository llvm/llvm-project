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
