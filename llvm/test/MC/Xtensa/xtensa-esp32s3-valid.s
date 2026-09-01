# RUN: llvm-mc %s -triple=xtensa  -mcpu=esp32s3 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4

LBL0:

# CHECK-INST:  ee.clr_bit_gpio_out 52
# CHECK: encoding: [0x44,0x43,0x76]
ee.clr_bit_gpio_out 52

# CHECK-INST:  ee.get_gpio_in a2
# CHECK: encoding: [0x24,0x08,0x65]
ee.get_gpio_in a2

# CHECK-INST:  ee.set_bit_gpio_out 18
# CHECK: encoding: [0x24,0x41,0x75]
ee.set_bit_gpio_out 18

# CHECK-INST:  ee.wr_mask_gpio_out a3, a2
# CHECK: encoding: [0x34,0x42,0x72]
ee.wr_mask_gpio_out a3, a2

# CHECK-INST:  ee.andq  q5, q6, q4
# CHECK: encoding: [0xc4,0xb8,0xed]
ee.andq q5, q6, q4

# CHECK-INST:  ee.bitrev  q2, a6
# CHECK: encoding: [0x64,0x7b,0xdd]
ee.bitrev q2, a6

# CHECK-INST:  ee.cmul.s16  q3, q6, q2, 3
# CHECK: encoding: [0x34,0x96,0x9e]
ee.cmul.s16 q3, q6, q2, 3

# CHECK-INST:  ee.cmul.s16.ld.incp  q2, a7, q5, q1, q4, 2
# CHECK: encoding: [0x7e,0x6e,0x05,0xe1]
ee.cmul.s16.ld.incp q2, a7, q5, q1, q4, 2

# CHECK-INST:  ee.cmul.s16.st.incp  q7, a11, q1, q5, q2, 3
# CHECK: encoding: [0xbf,0x53,0x71,0xe4]
ee.cmul.s16.st.incp q7, a11, q1, q5, q2, 3

# CHECK-INST:  ee.fft.ams.s16.ld.incp  q5, a5, q3, q1, q1, q2, q5, 1
# CHECK: encoding: [0x5e,0x5d,0xa5,0xd2]
ee.fft.ams.s16.ld.incp q5, a5, q3, q1, q1, q2, q5, 1

# CHECK-INST:  ee.fft.ams.s16.ld.incp.uaup  q7, a12, q4, q1, q5, q6, q3, 0
# CHECK: encoding: [0xcf,0x6f,0x63,0xd4]
ee.fft.ams.s16.ld.incp.uaup q7, a12, q4, q1, q5, q6, q3, 0

# CHECK-INST:  ee.fft.ams.s16.ld.r32.decp  q6, a5, q0, q2, q7, q2, q0, 0
# CHECK: encoding: [0x5f,0xc6,0x28,0xd8]
ee.fft.ams.s16.ld.r32.decp q6, a5, q0, q2, q7, q2, q0, 0

# CHECK-INST:  ee.fft.ams.s16.st.incp  q3, q6, a7, a6, q5, q5, q1, 1
# CHECK: encoding: [0x7f,0x46,0xb5,0xa7]
ee.fft.ams.s16.st.incp q3, q6, a7, a6, q5, q5, q1, 1

# CHECK-INST:  ee.fft.cmul.s16.ld.xp  q3, a12, a6, q7, q0, q7, 2
# CHECK: encoding: [0xce,0x16,0xf7,0xdd]
ee.fft.cmul.s16.ld.xp q3, a12, a6, q7, q0, q7, 2

# CHECK-INST:  ee.fft.cmul.s16.st.xp  q4, q0, q0, a2, a8, 6, 1, 1
# CHECK: encoding: [0x2f,0x38,0x08,0xaa]
ee.fft.cmul.s16.st.xp q4, q0, q0, a2, a8, 6, 1, 1

# CHECK-INST:  ee.fft.r2bf.s16  q7, q1, q3, q6, 1
# CHECK: encoding: [0x54,0x9d,0xfc]
ee.fft.r2bf.s16 q7, q1, q3, q6, 1

# CHECK-INST:  ee.fft.r2bf.s16.st.incp  q7, q3, q7, a2, 2
# CHECK: encoding: [0x2e,0xd4,0x7f,0xe8]
ee.fft.r2bf.s16.st.incp q7, q3, q7, a2, 2

# CHECK-INST:  ee.fft.vst.r32.decp  q3, a14, 0
# CHECK: encoding: [0xe4,0xb3,0xdd]
ee.fft.vst.r32.decp q3, a14, 0

# CHECK-INST:  ee.ldf.128.ip  f3, f5, f8, f0, a13, 64
# CHECK: encoding: [0xdf,0x84,0x0a,0x81]
ee.ldf.128.ip f3, f5, f8, f0, a13, 64

# CHECK-INST:  ee.ldf.128.xp  f5, f2, f4, f4, a7, a8
# CHECK: encoding: [0x7e,0x48,0x49,0x8a]
ee.ldf.128.xp f5, f2, f4, f4, a7, a8

# CHECK-INST:  ee.ldf.64.ip  f6, f5, a1, 488
# CHECK: encoding: [0x1e,0x65,0x5f,0xe0]
ee.ldf.64.ip f6, f5, a1, 488

# CHECK-INST:  ee.ldf.64.xp  f0, f6, a3, a8
# CHECK: encoding: [0x30,0x08,0x66]
ee.ldf.64.xp f0, f6, a3, a8

# CHECK-INST:  ee.ldqa.s16.128.ip  a11, 1904
# CHECK: encoding: [0xb4,0x77,0x01]
ee.ldqa.s16.128.ip a11, 1904

# CHECK-INST:  ee.ldqa.s16.128.xp  a6, a2
# CHECK: encoding: [0x64,0x42,0x7e]
ee.ldqa.s16.128.xp a6, a2

# CHECK-INST:  ee.ldqa.s8.128.ip  a8, 320
# CHECK: encoding: [0x84,0x14,0x11]
ee.ldqa.s8.128.ip a8, 320

# CHECK-INST:  ee.ldqa.s8.128.xp  a6, a11
# CHECK: encoding: [0x64,0x4b,0x71]
ee.ldqa.s8.128.xp a6, a11

# CHECK-INST:  ee.ldqa.u16.128.ip  a2, -1424
# CHECK: encoding: [0x24,0x27,0x45]
ee.ldqa.u16.128.ip a2, -1424

# CHECK-INST:  ee.ldqa.u16.128.xp  a3, a4
# CHECK: encoding: [0x34,0x44,0x7a]
ee.ldqa.u16.128.xp a3, a4

# CHECK-INST:  ee.ldqa.u8.128.ip  a4, 784
# CHECK: encoding: [0x44,0x31,0x15]
ee.ldqa.u8.128.ip a4, 784

# CHECK-INST:  ee.ldqa.u8.128.xp  a4, a9
# CHECK: encoding: [0x44,0x49,0x70]
ee.ldqa.u8.128.xp a4, a9

# CHECK-INST:  ee.ldxq.32  q2, q6, a11, 2, 1
# CHECK: encoding: [0xbf,0x8d,0xf2,0xe1]
ee.ldxq.32 q2, q6, a11, 2, 1

# CHECK-INST:  ee.ld.128.usar.ip  q4, a8, -592
# CHECK: encoding: [0x84,0x5b,0xe1]
ee.ld.128.usar.ip q4, a8, -592

# CHECK-INST:  ee.ld.128.usar.xp  q1, a9, a7
# CHECK: encoding: [0x94,0x87,0x8d]
ee.ld.128.usar.xp q1, a9, a7

# CHECK-INST:  ee.ld.accx.ip  a2, 720
# CHECK: encoding: [0x24,0x5a,0x0e]
ee.ld.accx.ip a2, 720

# CHECK-INST:  ee.ld.qacc_h.h.32.ip  a6, -292
# CHECK: encoding: [0x64,0x37,0x5e]
ee.ld.qacc_h.h.32.ip a6, -292

# CHECK-INST:  ee.ld.qacc_h.l.128.ip  a14, 96
# CHECK: encoding: [0xe4,0x06,0x06]
ee.ld.qacc_h.l.128.ip a14, 96

# CHECK-INST:  ee.ld.qacc_l.h.32.ip  a0, -184
# CHECK: encoding: [0x04,0x52,0x56]
ee.ld.qacc_l.h.32.ip a0, -184

# CHECK-INST:  ee.ld.qacc_l.l.128.ip  a5, -352
# CHECK: encoding: [0x54,0x6a,0x40]
ee.ld.qacc_l.l.128.ip a5, -352

# CHECK-INST:  ee.ld.ua_state.ip  a3, 864
# CHECK: encoding: [0x34,0x36,0x10]
ee.ld.ua_state.ip a3, 864

# CHECK-INST:  ee.movi.32.a  q7, a0, 1
# CHECK: encoding: [0x04,0xf5,0xfd]
ee.movi.32.a q7, a0, 1

# CHECK-INST:  ee.movi.32.q  q5, a5, 3
# CHECK: encoding: [0x54,0xbe,0xed]
ee.movi.32.q q5, a5, 3

# CHECK-INST:  ee.mov.s16.qacc  q1
# CHECK: encoding: [0x24,0xff,0xcd]
ee.mov.s16.qacc q1

# CHECK-INST:  ee.mov.s8.qacc  q7
# CHECK: encoding: [0x34,0xff,0xfd]
ee.mov.s8.qacc q7

# CHECK-INST:  ee.mov.u16.qacc  q2
# CHECK: encoding: [0x64,0x7f,0xdd]
ee.mov.u16.qacc q2

# CHECK-INST:  ee.mov.u8.qacc  q2
# CHECK: encoding: [0x74,0x7f,0xdd]
ee.mov.u8.qacc q2

# CHECK-INST:  ee.notq  q7, q0
# CHECK: encoding: [0x04,0xff,0xfd]
ee.notq q7, q0

# CHECK-INST:  ee.orq  q1, q5, q3
# CHECK: encoding: [0xb4,0xf4,0xcd]
ee.orq q1, q5, q3

# CHECK-INST:  ee.slci.2q  q7, q4, 2
# CHECK: encoding: [0x24,0xc6,0xfc]
ee.slci.2q q7, q4, 2

# CHECK-INST:  ee.slcxxp.2q  q6, q7, a11, a4
# CHECK: encoding: [0xb4,0x74,0xb6]
ee.slcxxp.2q q6, q7, a11, a4

# CHECK-INST:  ee.srci.2q  q6, q6, 14
# CHECK: encoding: [0xe4,0x6a,0xfc]
ee.srci.2q q6, q6, 14

# CHECK-INST:  ee.srcmb.s16.qacc  q3, a7, 0
# CHECK: encoding: [0x74,0xf2,0xdd]
ee.srcmb.s16.qacc q3, a7, 0

# CHECK-INST:  ee.srcmb.s8.qacc  q4, a1, 1
# CHECK: encoding: [0x14,0x7e,0xed]
ee.srcmb.s8.qacc q4, a1, 1

# CHECK-INST:  ee.srcq.128.st.incp  q1, q4, a6
# CHECK: encoding: [0x64,0x1e,0xec]
ee.srcq.128.st.incp q1, q4, a6

# CHECK-INST:  ee.srcxxp.2q  q6, q0, a2, a14
# CHECK: encoding: [0x24,0x0e,0xf6]
ee.srcxxp.2q q6, q0, a2, a14

# CHECK-INST:  ee.src.q  q6, q7, q5
# CHECK: encoding: [0x64,0xf3,0xec]
ee.src.q q6, q7, q5

# CHECK-INST:  ee.src.q.ld.ip  q2, a2, 1792, q6, q7
# CHECK: encoding: [0x2f,0xa0,0x7a,0xe1]
ee.src.q.ld.ip q2, a2, 1792, q6, q7

# CHECK-INST:  ee.src.q.ld.xp  q2, a4, a9, q1, q7
# CHECK: encoding: [0x4e,0x49,0x72,0xe8]
ee.src.q.ld.xp q2, a4, a9, q1, q7

# CHECK-INST:  ee.src.q.qup  q4, q3, q7
# CHECK: encoding: [0x44,0xb7,0xfc]
ee.src.q.qup q4, q3, q7

# CHECK-INST:  ee.srs.accx  a12, a1, 0
# CHECK: encoding: [0x14,0x1c,0x7e]
ee.srs.accx a12, a1, 0

# CHECK-INST:  ee.stf.128.ip  f4, f3, f8, f2, a4, -128
# CHECK: encoding: [0x4f,0x88,0x21,0x92]
ee.stf.128.ip f4, f3, f8, f2, a4, -128

# CHECK-INST:  ee.stf.128.xp  f2, f0, f5, f8, a11, a5
# CHECK: encoding: [0xbe,0x55,0x80,0x99]
ee.stf.128.xp f2, f0, f5, f8, a11, a5

# CHECK-INST:  ee.stf.64.ip  f3, f6, a10, -848
# CHECK: encoding: [0xaf,0x36,0x65,0xe2]
ee.stf.64.ip f3, f6, a10, -848

# CHECK-INST:  ee.stf.64.xp  f2, f1, a1, a14
# CHECK: encoding: [0x10,0x2e,0x17]
ee.stf.64.xp f2, f1, a1, a14

# CHECK-INST:  ee.stxq.32  q5, q2, a5, 0, 1
# CHECK: encoding: [0x5e,0x80,0xd0,0xe6]
ee.stxq.32 q5, q2, a5, 0, 1

# CHECK-INST:  ee.st.accx.ip  a10, 24
# CHECK: encoding: [0xa4,0x03,0x02]
ee.st.accx.ip a10, 24

# CHECK-INST:  ee.st.qacc_h.h.32.ip  a14, 380
# CHECK: encoding: [0xe4,0x5f,0x12]
ee.st.qacc_h.h.32.ip a14, 380

# CHECK-INST:  ee.st.qacc_h.l.128.ip  a7, -624
# CHECK: encoding: [0x74,0x59,0x4d]
ee.st.qacc_h.l.128.ip a7, -624

# CHECK-INST:  ee.st.qacc_l.h.32.ip  a10, -20
# CHECK: encoding: [0xa4,0x7b,0x5d]
ee.st.qacc_l.h.32.ip a10, -20

# CHECK-INST:  ee.st.qacc_l.l.128.ip  a4, 1936
# CHECK: encoding: [0x44,0x79,0x0c]
ee.st.qacc_l.l.128.ip a4, 1936

# CHECK-INST:  ee.st.ua_state.ip  a4, -1728
# CHECK: encoding: [0x44,0x14,0x5c]
ee.st.ua_state.ip a4, -1728

# CHECK-INST:  ee.vadds.s16  q5, q1, q4
# CHECK: encoding: [0x64,0xc1,0xae]
ee.vadds.s16 q5, q1, q4

# CHECK-INST:  ee.vadds.s16.ld.incp  q6, a6, q1, q3, q1
# CHECK: encoding: [0x6e,0xcd,0xa1,0xe3]
ee.vadds.s16.ld.incp q6, a6, q1, q3, q1

# CHECK-INST:  ee.vadds.s16.st.incp  q4, a0, q1, q3, q1
# CHECK: encoding: [0x0e,0xc0,0xc9,0xe4]
ee.vadds.s16.st.incp q4, a0, q1, q3, q1

# CHECK-INST:  ee.vadds.s32  q3, q5, q2
# CHECK: encoding: [0x74,0x95,0x9e]
ee.vadds.s32 q3, q5, q2

# CHECK-INST:  ee.vadds.s32.ld.incp  q4, a4, q1, q6, q5
# CHECK: encoding: [0x4f,0xad,0xb1,0xe2]
ee.vadds.s32.ld.incp q4, a4, q1, q6, q5

# CHECK-INST:  ee.vadds.s32.st.incp  q5, a1, q0, q6, q0
# CHECK: encoding: [0x1f,0x81,0x58,0xe4]
ee.vadds.s32.st.incp q5, a1, q0, q6, q0

# CHECK-INST:  ee.vadds.s8  q4, q4, q5
# CHECK: encoding: [0x84,0x4c,0xae]
ee.vadds.s8 q4, q4, q5

# CHECK-INST:  ee.vadds.s8.ld.incp  q2, a14, q0, q3, q3
# CHECK: encoding: [0xee,0xdc,0x90,0xe1]
ee.vadds.s8.ld.incp q2, a14, q0, q3, q3

# CHECK-INST:  ee.vadds.s8.st.incp  q0, a9, q4, q7, q0
# CHECK: encoding: [0x9f,0xc2,0x0c,0xe4]
ee.vadds.s8.st.incp q0, a9, q4, q7, q0

# CHECK-INST:  ee.vcmp.eq.s16  q5, q3, q0
# CHECK: encoding: [0x94,0x83,0xae]
ee.vcmp.eq.s16 q5, q3, q0

# CHECK-INST:  ee.vcmp.eq.s32  q5, q5, q4
# CHECK: encoding: [0xa4,0xc5,0xae]
ee.vcmp.eq.s32 q5, q5, q4

# CHECK-INST:  ee.vcmp.eq.s8  q0, q4, q2
# CHECK: encoding: [0xb4,0x14,0x8e]
ee.vcmp.eq.s8 q0, q4, q2

# CHECK-INST:  ee.vcmp.gt.s16  q1, q5, q2
# CHECK: encoding: [0xc4,0x95,0x8e]
ee.vcmp.gt.s16 q1, q5, q2

# CHECK-INST:  ee.vcmp.gt.s32  q4, q1, q5
# CHECK: encoding: [0xd4,0x49,0xae]
ee.vcmp.gt.s32 q4, q1, q5

# CHECK-INST:  ee.vcmp.gt.s8  q3, q6, q3
# CHECK: encoding: [0xe4,0x9e,0x9e]
ee.vcmp.gt.s8 q3, q6, q3

# CHECK-INST:  ee.vcmp.lt.s16  q3, q7, q0
# CHECK: encoding: [0xf4,0x87,0x9e]
ee.vcmp.lt.s16 q3, q7, q0

# CHECK-INST:  ee.vcmp.lt.s32  q2, q2, q1
# CHECK: encoding: [0x04,0x2a,0x9e]
ee.vcmp.lt.s32 q2, q2, q1

# CHECK-INST:  ee.vcmp.lt.s8  q7, q1, q6
# CHECK: encoding: [0x14,0xf1,0xbe]
ee.vcmp.lt.s8 q7, q1, q6

# CHECK-INST:  ee.vldbc.16  q6, a11
# CHECK: encoding: [0xb4,0x73,0xfd]
ee.vldbc.16 q6, a11

# CHECK-INST:  ee.vldbc.16.ip  q6, a4, 124
# CHECK: encoding: [0x44,0x3e,0xb5]
ee.vldbc.16.ip q6, a4, 124

# CHECK-INST:  ee.vldbc.16.xp  q2, a0, a7
# CHECK: encoding: [0x04,0x47,0x9d]
ee.vldbc.16.xp q2, a0, a7

# CHECK-INST:  ee.vldbc.32  q4, a0
# CHECK: encoding: [0x04,0x77,0xed]
ee.vldbc.32 q4, a0

# CHECK-INST:  ee.vldbc.32.ip  q6, a12, 308
# CHECK: encoding: [0xc4,0x4d,0xb2]
ee.vldbc.32.ip q6, a12, 308

# CHECK-INST:  ee.vldbc.32.xp  q1, a11, a0
# CHECK: encoding: [0xb4,0x90,0x8d]
ee.vldbc.32.xp q1, a11, a0

# CHECK-INST:  ee.vldbc.8  q2, a3
# CHECK: encoding: [0x34,0x3b,0xdd]
ee.vldbc.8 q2, a3

# CHECK-INST:  ee.vldbc.8.ip  q3, a3, 103
# CHECK: encoding: [0x34,0xe7,0xd5]
ee.vldbc.8.ip q3, a3, 103

# CHECK-INST:  ee.vldbc.8.xp  q2, a0, a13
# CHECK: encoding: [0x04,0x5d,0x9d]
ee.vldbc.8.xp q2, a0, a13

# CHECK-INST:  ee.vldhbc.16.incp  q5, q5, a14
# CHECK: encoding: [0xe4,0xd2,0xec]
ee.vldhbc.16.incp q5, q5, a14

# CHECK-INST:  ee.vld.128.ip  q3, a14, 1248
# CHECK: encoding: [0xe4,0xce,0x93]
ee.vld.128.ip q3, a14, 1248

# CHECK-INST:  ee.vld.128.xp  q5, a10, a12
# CHECK: encoding: [0xa4,0xac,0xad]
ee.vld.128.xp q5, a10, a12

# CHECK-INST:  ee.vld.h.64.ip  q4, a14, 240
# CHECK: encoding: [0xe4,0x1e,0xa8]
ee.vld.h.64.ip q4, a14, 240

# CHECK-INST:  ee.vld.h.64.xp  q7, a4, a8
# CHECK: encoding: [0x44,0xe8,0xbd]
ee.vld.h.64.xp q7, a4, a8

# CHECK-INST:  ee.vld.l.64.ip  q1, a8, 8
# CHECK: encoding: [0x84,0x81,0x89]
ee.vld.l.64.ip q1, a8, 8

# CHECK-INST:  ee.vld.l.64.xp  q1, a2, a9
# CHECK: encoding: [0x24,0xb9,0x8d]
ee.vld.l.64.xp q1, a2, a9

# CHECK-INST:  ee.vmax.s16  q2, q5, q6
# CHECK: encoding: [0x24,0x75,0x9e]
ee.vmax.s16 q2, q5, q6

# CHECK-INST:  ee.vmax.s16.ld.incp  q0, a0, q6, q1, q2
# CHECK: encoding: [0x0e,0x5d,0x16,0xe0]
ee.vmax.s16.ld.incp q0, a0, q6, q1, q2

# CHECK-INST:  ee.vmax.s16.st.incp  q5, a10, q6, q6, q7
# CHECK: encoding: [0xaf,0xb3,0xde,0xe4]
ee.vmax.s16.st.incp q5, a10, q6, q6, q7

# CHECK-INST:  ee.vmax.s32  q3, q2, q7
# CHECK: encoding: [0x34,0xfa,0x9e]
ee.vmax.s32 q3, q2, q7

# CHECK-INST:  ee.vmax.s32.ld.incp  q1, a3, q1, q1, q0
# CHECK: encoding: [0x3e,0x4e,0x19,0xe0]
ee.vmax.s32.ld.incp q1, a3, q1, q1, q0

# CHECK-INST:  ee.vmax.s32.st.incp  q3, a12, q4, q6, q3
# CHECK: encoding: [0xcf,0x90,0xb4,0xe5]
ee.vmax.s32.st.incp q3, a12, q4, q6, q3

# CHECK-INST:  ee.vmax.s8  q4, q1, q6
# CHECK: encoding: [0x44,0x71,0xae]
ee.vmax.s8 q4, q1, q6

# CHECK-INST:  ee.vmax.s8.ld.incp  q3, a10, q5, q1, q5
# CHECK: encoding: [0xae,0x6f,0x9d,0xe1]
ee.vmax.s8.ld.incp q3, a10, q5, q1, q5

# CHECK-INST:  ee.vmax.s8.st.incp  q3, a9, q3, q6, q7
# CHECK: encoding: [0x9f,0xb0,0xbb,0xe5]
ee.vmax.s8.st.incp q3, a9, q3, q6, q7

# CHECK-INST:  ee.vmin.s16  q6, q2, q5
# CHECK: encoding: [0x54,0x6a,0xbe]
ee.vmin.s16 q6, q2, q5

# CHECK-INST:  ee.vmin.s16.ld.incp  q5, a3, q2, q4, q0
# CHECK: encoding: [0x3f,0x0e,0x2a,0xe2]
ee.vmin.s16.ld.incp q5, a3, q2, q4, q0

# CHECK-INST:  ee.vmin.s16.st.incp  q4, a9, q4, q6, q0
# CHECK: encoding: [0x9f,0x81,0x44,0xe5]
ee.vmin.s16.st.incp q4, a9, q4, q6, q0

# CHECK-INST:  ee.vmin.s32  q1, q1, q6
# CHECK: encoding: [0x64,0xf1,0x8e]
ee.vmin.s32 q1, q1, q6

# CHECK-INST:  ee.vmin.s32.ld.incp  q0, a1, q3, q2, q0
# CHECK: encoding: [0x1e,0x8e,0x33,0xe0]
ee.vmin.s32.ld.incp q0, a1, q3, q2, q0

# CHECK-INST:  ee.vmin.s32.st.incp  q0, a12, q4, q4, q3
# CHECK: encoding: [0xcf,0x11,0x8c,0xe5]
ee.vmin.s32.st.incp q0, a12, q4, q4, q3

# CHECK-INST:  ee.vmin.s8  q7, q6, q0
# CHECK: encoding: [0x74,0xa6,0xbe]
ee.vmin.s8 q7, q6, q0

# CHECK-INST:  ee.vmin.s8.ld.incp  q2, a13, q7, q7, q3
# CHECK: encoding: [0xdf,0xdf,0xa7,0xe1]
ee.vmin.s8.ld.incp q2, a13, q7, q7, q3

# CHECK-INST:  ee.vmin.s8.st.incp  q2, a4, q4, q7, q1
# CHECK: encoding: [0x4f,0xc2,0xa4,0xe5]
ee.vmin.s8.st.incp q2, a4, q4, q7, q1

# CHECK-INST:  ee.vmulas.s16.accx  q0, q7
# CHECK: encoding: [0x84,0x58,0x1a]
ee.vmulas.s16.accx q0, q7

# CHECK-INST:  ee.vmulas.s16.accx.ld.ip  q7, a7, -16, q2, q0
# CHECK: encoding: [0x7e,0x8f,0x08,0xff]
ee.vmulas.s16.accx.ld.ip q7, a7, -16, q2, q0

# CHECK-INST:  ee.vmulas.s16.accx.ld.ip.qup  q5, a14, 32, q0, q2, q0, q2
# CHECK: encoding: [0xee,0x12,0x0a,0x02]
ee.vmulas.s16.accx.ld.ip.qup q5, a14, 32, q0, q2, q0, q2

# CHECK-INST:  ee.vmulas.s16.accx.ld.xp  q1, a0, a1, q2, q6
# CHECK: encoding: [0x0e,0xb1,0x18,0xf0]
ee.vmulas.s16.accx.ld.xp q1, a0, a1, q2, q6

# CHECK-INST:  ee.vmulas.s16.accx.ld.xp.qup  q4, a8, a10, q4, q0, q0, q3
# CHECK: encoding: [0x8f,0x0a,0x03,0xb2]
ee.vmulas.s16.accx.ld.xp.qup q4, a8, a10, q4, q0, q0, q3

# CHECK-INST:  ee.vmulas.s16.qacc  q0, q6
# CHECK: encoding: [0x84,0x70,0x1a]
ee.vmulas.s16.qacc q0, q6

# CHECK-INST:  ee.vmulas.s16.qacc.ldbc.incp  q2, a6, q3, q4
# CHECK: encoding: [0x64,0xc3,0x87]
ee.vmulas.s16.qacc.ldbc.incp q2, a6, q3, q4

# CHECK-INST:  ee.vmulas.s16.qacc.ldbc.incp.qup  q0, a4, q1, q6, q4, q5
# CHECK: encoding: [0x4e,0x78,0x45,0xe0]
ee.vmulas.s16.qacc.ldbc.incp.qup q0, a4, q1, q6, q4, q5

# CHECK-INST:  ee.vmulas.s16.qacc.ld.ip  q7, a7, -64, q7, q7
# CHECK: encoding: [0x7f,0xfc,0x89,0xff]
ee.vmulas.s16.qacc.ld.ip q7, a7, -64, q7, q7

# CHECK-INST:  ee.vmulas.s16.qacc.ld.ip.qup  q0, a10, 48, q3, q6, q3, q6
# CHECK: encoding: [0xae,0xf3,0x36,0x10]
ee.vmulas.s16.qacc.ld.ip.qup q0, a10, 48, q3, q6, q3, q6

# CHECK-INST:  ee.vmulas.s16.qacc.ld.xp  q3, a11, a4, q4, q5
# CHECK: encoding: [0xbf,0x24,0x99,0xf1]
ee.vmulas.s16.qacc.ld.xp q3, a11, a4, q4, q5

# CHECK-INST:  ee.vmulas.s16.qacc.ld.xp.qup  q2, a9, a1, q3, q2, q1, q7
# CHECK: encoding: [0x9e,0xd1,0x17,0xb5]
ee.vmulas.s16.qacc.ld.xp.qup q2, a9, a1, q3, q2, q1, q7

# CHECK-INST:  ee.vmulas.s8.accx  q1, q0
# CHECK: encoding: [0xc4,0x01,0x1a]
ee.vmulas.s8.accx q1, q0

# CHECK-INST:  ee.vmulas.s8.accx.ld.ip  q2, a8, 80, q3, q0
# CHECK: encoding: [0x8e,0xc5,0x02,0xf1]
ee.vmulas.s8.accx.ld.ip q2, a8, 80, q3, q0

# CHECK-INST:  ee.vmulas.s8.accx.ld.ip.qup  q2, a9, -80, q1, q2, q6, q3
# CHECK: encoding: [0x9e,0x5b,0x63,0x2d]
ee.vmulas.s8.accx.ld.ip.qup q2, a9, -80, q1, q2, q6, q3

# CHECK-INST:  ee.vmulas.s8.accx.ld.xp  q3, a3, a4, q4, q7
# CHECK: encoding: [0x3f,0x34,0x9a,0xf1]
ee.vmulas.s8.accx.ld.xp q3, a3, a4, q4, q7

# CHECK-INST:  ee.vmulas.s8.accx.ld.xp.qup  q0, a3, a1, q4, q5, q3, q3
# CHECK: encoding: [0x3f,0x21,0xb3,0xb8]
ee.vmulas.s8.accx.ld.xp.qup q0, a3, a1, q4, q5, q3, q3

# CHECK-INST:  ee.vmulas.s8.qacc  q5, q7
# CHECK: encoding: [0xc4,0x7d,0x1a]
ee.vmulas.s8.qacc q5, q7

# CHECK-INST:  ee.vmulas.s8.qacc.ldbc.incp  q7, a1, q6, q1
# CHECK: encoding: [0x14,0xae,0xb7]
ee.vmulas.s8.qacc.ldbc.incp q7, a1, q6, q1

# CHECK-INST:  ee.vmulas.s8.qacc.ldbc.incp.qup  q3, a11, q4, q6, q5, q6
# CHECK: encoding: [0xbf,0x39,0x5e,0xe1]
ee.vmulas.s8.qacc.ldbc.incp.qup q3, a11, q4, q6, q5, q6

# CHECK-INST:  ee.vmulas.s8.qacc.ld.ip  q5, a10, -16, q0, q0
# CHECK: encoding: [0xae,0x0f,0x0b,0xfe]
ee.vmulas.s8.qacc.ld.ip q5, a10, -16, q0, q0

# CHECK-INST:  ee.vmulas.s8.qacc.ld.ip.qup  q7, a9, -48, q6, q2, q1, q2
# CHECK: encoding: [0x9f,0x9d,0x1a,0x3f]
ee.vmulas.s8.qacc.ld.ip.qup q7, a9, -48, q6, q2, q1, q2

# CHECK-INST:  ee.vmulas.s8.qacc.ld.xp  q1, a1, a12, q5, q0
# CHECK: encoding: [0x1f,0x4c,0x1b,0xf0]
ee.vmulas.s8.qacc.ld.xp q1, a1, a12, q5, q0

# CHECK-INST:  ee.vmulas.s8.qacc.ld.xp.qup  q0, a1, a14, q1, q6, q2, q4
# CHECK: encoding: [0x1e,0x7e,0x24,0xbc]
ee.vmulas.s8.qacc.ld.xp.qup q0, a1, a14, q1, q6, q2, q4

# CHECK-INST:  ee.vmulas.u16.accx  q7, q1
# CHECK: encoding: [0x84,0x0f,0x0a]
ee.vmulas.u16.accx q7, q1

# CHECK-INST:  ee.vmulas.u16.accx.ld.ip  q5, a8, -32, q1, q4
# CHECK: encoding: [0x8e,0x6e,0x0c,0xfe]
ee.vmulas.u16.accx.ld.ip q5, a8, -32, q1, q4

# CHECK-INST:  ee.vmulas.u16.accx.ld.ip.qup  q1, a0, 48, q7, q4, q4, q0
# CHECK: encoding: [0x0f,0xe3,0x48,0x40]
ee.vmulas.u16.accx.ld.ip.qup q1, a0, 48, q7, q4, q4, q0

# CHECK-INST:  ee.vmulas.u16.accx.ld.xp  q3, a14, a4, q5, q4
# CHECK: encoding: [0xef,0x64,0x1c,0xf1]
ee.vmulas.u16.accx.ld.xp q3, a14, a4, q5, q4

# CHECK-INST:  ee.vmulas.u16.accx.ld.xp.qup  q4, a3, a7, q6, q2, q4, q4
# CHECK: encoding: [0x3f,0x97,0x44,0xc2]
ee.vmulas.u16.accx.ld.xp.qup q4, a3, a7, q6, q2, q4, q4

# CHECK-INST:  ee.vmulas.u16.qacc  q5, q5
# CHECK: encoding: [0x84,0x6d,0x0a]
ee.vmulas.u16.qacc q5, q5

# CHECK-INST:  ee.vmulas.u16.qacc.ldbc.incp  q6, a7, q0, q3
# CHECK: encoding: [0x74,0x98,0xd7]
ee.vmulas.u16.qacc.ldbc.incp q6, a7, q0, q3

# CHECK-INST:  ee.vmulas.u16.qacc.ldbc.incp.qup  q0, a12, q6, q3, q2, q0
# CHECK: encoding: [0xcf,0x9a,0xa0,0xe0]
ee.vmulas.u16.qacc.ldbc.incp.qup q0, a12, q6, q3, q2, q0

# CHECK-INST:  ee.vmulas.u16.qacc.ld.ip  q4, a10, 16, q3, q2
# CHECK: encoding: [0xae,0xd1,0x05,0xf2]
ee.vmulas.u16.qacc.ld.ip q4, a10, 16, q3, q2

# CHECK-INST:  ee.vmulas.u16.qacc.ld.ip.qup  q2, a4, 0, q5, q4, q2, q6
# CHECK: encoding: [0x4f,0x60,0x26,0x51]
ee.vmulas.u16.qacc.ld.ip.qup q2, a4, 0, q5, q4, q2, q6

# CHECK-INST:  ee.vmulas.u16.qacc.ld.xp  q6, a14, a2, q4, q0
# CHECK: encoding: [0xef,0x02,0x15,0xf3]
ee.vmulas.u16.qacc.ld.xp q6, a14, a2, q4, q0

# CHECK-INST:  ee.vmulas.u16.qacc.ld.xp.qup  q6, a12, a11, q6, q7, q4, q1
# CHECK: encoding: [0xcf,0xbb,0xc1,0xc7]
ee.vmulas.u16.qacc.ld.xp.qup q6, a12, a11, q6, q7, q4, q1

# CHECK-INST:  ee.vmulas.u8.accx  q2, q1
# CHECK: encoding: [0xc4,0x0a,0x0a]
ee.vmulas.u8.accx q2, q1

# CHECK-INST:  ee.vmulas.u8.accx.ld.ip  q6, a3, -112, q2, q7
# CHECK: encoding: [0x3e,0xb9,0x86,0xff]
ee.vmulas.u8.accx.ld.ip q6, a3, -112, q2, q7

# CHECK-INST:  ee.vmulas.u8.accx.ld.ip.qup  q7, a3, -32, q3, q3, q7, q5
# CHECK: encoding: [0x3e,0xde,0xfd,0x6f]
ee.vmulas.u8.accx.ld.ip.qup q7, a3, -32, q3, q3, q7, q5

# CHECK-INST:  ee.vmulas.u8.accx.ld.xp  q4, a4, a9, q4, q0
# CHECK: encoding: [0x4f,0x09,0x16,0xf2]
ee.vmulas.u8.accx.ld.xp q4, a4, a9, q4, q0

# CHECK-INST:  ee.vmulas.u8.accx.ld.xp.qup  q5, a7, a13, q4, q7, q2, q6
# CHECK: encoding: [0x7f,0x3d,0xae,0xca]
ee.vmulas.u8.accx.ld.xp.qup q5, a7, a13, q4, q7, q2, q6

# CHECK-INST:  ee.vmulas.u8.qacc  q3, q6
# CHECK: encoding: [0xc4,0x73,0x0a]
ee.vmulas.u8.qacc q3, q6

# CHECK-INST:  ee.vmulas.u8.qacc.ldbc.incp  q4, a1, q0, q5
# CHECK: encoding: [0x14,0x48,0xf7]
ee.vmulas.u8.qacc.ldbc.incp q4, a1, q0, q5

# CHECK-INST:  ee.vmulas.u8.qacc.ldbc.incp.qup  q2, a1, q5, q7, q6, q4
# CHECK: encoding: [0x1f,0x7b,0xe4,0xe1]
ee.vmulas.u8.qacc.ldbc.incp.qup q2, a1, q5, q7, q6, q4

# CHECK-INST:  ee.vmulas.u8.qacc.ld.ip  q2, a12, 32, q1, q4
# CHECK: encoding: [0xce,0x62,0x07,0xf1]
ee.vmulas.u8.qacc.ld.ip q2, a12, 32, q1, q4

# CHECK-INST:  ee.vmulas.u8.qacc.ld.ip.qup  q0, a6, 48, q0, q0, q6, q0
# CHECK: encoding: [0x6e,0x03,0x60,0x70]
ee.vmulas.u8.qacc.ld.ip.qup q0, a6, 48, q0, q0, q6, q0

# CHECK-INST:  ee.vmulas.u8.qacc.ld.xp  q6, a1, a1, q2, q5
# CHECK: encoding: [0x1e,0xa1,0x97,0xf3]
ee.vmulas.u8.qacc.ld.xp q6, a1, a1, q2, q5

# CHECK-INST:  ee.vmulas.u8.qacc.ld.xp.qup  q1, a8, a10, q3, q7, q1, q3
# CHECK: encoding: [0x8e,0xfa,0x9b,0xcc]
ee.vmulas.u8.qacc.ld.xp.qup q1, a8, a10, q3, q7, q1, q3

# CHECK-INST:  ee.vmul.s16  q0, q4, q1
# CHECK: encoding: [0x84,0x2c,0x8e]
ee.vmul.s16 q0, q4, q1

# CHECK-INST:  ee.vmul.s16.ld.incp  q4, a5, q1, q5, q5
# CHECK: encoding: [0x5f,0x6f,0xb1,0xe2]
ee.vmul.s16.ld.incp q4, a5, q1, q5, q5

# CHECK-INST:  ee.vmul.s16.st.incp  q4, a4, q2, q5, q0
# CHECK: encoding: [0x4f,0x42,0x4a,0xe5]
ee.vmul.s16.st.incp q4, a4, q2, q5, q0

# CHECK-INST:  ee.vmul.s8  q5, q3, q2
# CHECK: encoding: [0x94,0xb3,0xae]
ee.vmul.s8 q5, q3, q2

# CHECK-INST:  ee.vmul.s8.ld.incp  q6, a11, q3, q6, q4
# CHECK: encoding: [0xbf,0xac,0x43,0xe3]
ee.vmul.s8.ld.incp q6, a11, q3, q6, q4

# CHECK-INST:  ee.vmul.s8.st.incp  q5, a5, q5, q2, q4
# CHECK: encoding: [0x5e,0xa3,0x55,0xe5]
ee.vmul.s8.st.incp q5, a5, q5, q2, q4

# CHECK-INST:  ee.vmul.u16  q0, q0, q5
# CHECK: encoding: [0xa4,0x68,0x8e]
ee.vmul.u16 q0, q0, q5

# CHECK-INST:  ee.vmul.u16.ld.incp  q4, a2, q0, q1, q1
# CHECK: encoding: [0x2e,0x4c,0xd0,0xe2]
ee.vmul.u16.ld.incp q4, a2, q0, q1, q1

# CHECK-INST:  ee.vmul.u16.st.incp  q6, a5, q1, q2, q7
# CHECK: encoding: [0x5e,0xb3,0xe9,0xe5]
ee.vmul.u16.st.incp q6, a5, q1, q2, q7

# CHECK-INST:  ee.vmul.u8  q6, q4, q5
# CHECK: encoding: [0xb4,0x6c,0xbe]
ee.vmul.u8 q6, q4, q5

# CHECK-INST:  ee.vmul.u8.ld.incp  q1, a5, q4, q1, q1
# CHECK: encoding: [0x5e,0x4c,0xec,0xe0]
ee.vmul.u8.ld.incp q1, a5, q4, q1, q1

# CHECK-INST:  ee.vmul.u8.st.incp  q4, a12, q5, q0, q4
# CHECK: encoding: [0xce,0x20,0x4d,0xe8]
ee.vmul.u8.st.incp q4, a12, q5, q0, q4

# CHECK-INST:  ee.vprelu.s16  q2, q7, q0, a1
# CHECK: encoding: [0x14,0x07,0x9c]
ee.vprelu.s16 q2, q7, q0, a1

# CHECK-INST:  ee.vprelu.s8  q5, q6, q5, a13
# CHECK: encoding: [0xd4,0xee,0xac]
ee.vprelu.s8 q5, q6, q5, a13

# CHECK-INST:  ee.vrelu.s16  q2, a14, a5
# CHECK: encoding: [0x54,0x1e,0xdd]
ee.vrelu.s16 q2, a14, a5

# CHECK-INST:  ee.vrelu.s8  q4, a14, a1
# CHECK: encoding: [0x14,0x5e,0xed]
ee.vrelu.s8 q4, a14, a1

# CHECK-INST:  ee.vsl.32  q0, q1
# CHECK: encoding: [0x04,0xbf,0xcd]
ee.vsl.32 q0, q1

# CHECK-INST:  ee.vsmulas.s16.qacc  q2, q7, 2
# CHECK: encoding: [0xc4,0x7a,0x9e]
ee.vsmulas.s16.qacc q2, q7, 2

# CHECK-INST:  ee.vsmulas.s16.qacc.ld.incp  q7, a3, q3, q4, 3
# CHECK: encoding: [0x3e,0xec,0x7b,0xe3]
ee.vsmulas.s16.qacc.ld.incp q7, a3, q3, q4, 3

# CHECK-INST:  ee.vsmulas.s8.qacc  q3, q6, 3
# CHECK: encoding: [0x54,0xd3,0x8e]
ee.vsmulas.s8.qacc q3, q6, 3

# CHECK-INST:  ee.vsmulas.s8.qacc.ld.incp  q1, a8, q1, q1, 4
# CHECK: encoding: [0x8e,0x4c,0xaa,0xe0]
ee.vsmulas.s8.qacc.ld.incp q1, a8, q1, q1, 4

# CHECK-INST:  ee.vsr.32  q4, q3
# CHECK: encoding: [0xc4,0xbf,0xdd]
ee.vsr.32 q4, q3

# CHECK-INST:  ee.vst.128.ip  q3, a6, -816
# CHECK: encoding: [0x64,0xcd,0xda]
ee.vst.128.ip q3, a6, -816

# CHECK-INST:  ee.vst.128.xp  q6, a12, a14
# CHECK: encoding: [0xc4,0x7e,0xbd]
ee.vst.128.xp q6, a12, a14

# CHECK-INST:  ee.vst.h.64.ip  q2, a5, 40
# CHECK: encoding: [0x54,0x05,0x9b]
ee.vst.h.64.ip q2, a5, 40

# CHECK-INST:  ee.vst.h.64.xp  q2, a13, a6
# CHECK: encoding: [0xd4,0x06,0xdd]
ee.vst.h.64.xp q2, a13, a6

# CHECK-INST:  ee.vst.l.64.ip  q5, a8, 16
# CHECK: encoding: [0x84,0x82,0xa4]
ee.vst.l.64.ip q5, a8, 16

# CHECK-INST:  ee.vst.l.64.xp  q0, a13, a6
# CHECK: encoding: [0xd4,0x46,0xcd]
ee.vst.l.64.xp q0, a13, a6

# CHECK-INST:  ee.vsubs.s16  q5, q1, q4
# CHECK: encoding: [0xd4,0xe1,0xae]
ee.vsubs.s16 q5, q1, q4

# CHECK-INST:  ee.vsubs.s16.ld.incp  q1, a4, q6, q0, q1
# CHECK: encoding: [0x4e,0x0d,0xce,0xe0]
ee.vsubs.s16.ld.incp q1, a4, q6, q0, q1

# CHECK-INST:  ee.vsubs.s16.st.incp  q7, a13, q7, q5, q2
# CHECK: encoding: [0xdf,0x51,0x7f,0xe8]
ee.vsubs.s16.st.incp q7, a13, q7, q5, q2

# CHECK-INST:  ee.vsubs.s32  q2, q7, q6
# CHECK: encoding: [0xe4,0x77,0x9e]
ee.vsubs.s32 q2, q7, q6

# CHECK-INST:  ee.vsubs.s32.ld.incp  q1, a8, q1, q4, q0
# CHECK: encoding: [0x8f,0x0d,0x59,0xe0]
ee.vsubs.s32.ld.incp q1, a8, q1, q4, q0

# CHECK-INST:  ee.vsubs.s32.st.incp  q1, a5, q7, q4, q0
# CHECK: encoding: [0x5f,0x02,0x1f,0xe8]
ee.vsubs.s32.st.incp q1, a5, q7, q4, q0

# CHECK-INST:  ee.vsubs.s8  q7, q1, q5
# CHECK: encoding: [0xf4,0xe9,0xbe]
ee.vsubs.s8 q7, q1, q5

# CHECK-INST:  ee.vsubs.s8.ld.incp  q4, a2, q6, q1, q6
# CHECK: encoding: [0x2e,0x7d,0x66,0xe2]
ee.vsubs.s8.ld.incp q4, a2, q6, q1, q6

# CHECK-INST:  ee.vsubs.s8.st.incp  q6, a1, q6, q2, q3
# CHECK: encoding: [0x1e,0x93,0xee,0xe8]
ee.vsubs.s8.st.incp q6, a1, q6, q2, q3

# CHECK-INST:  ee.vunzip.16  q6, q5
# CHECK: encoding: [0x84,0xe3,0xec]
ee.vunzip.16 q6, q5

# CHECK-INST:  ee.vunzip.32  q0, q6
# CHECK: encoding: [0x94,0x03,0xfc]
ee.vunzip.32 q0, q6

# CHECK-INST:  ee.vunzip.8  q5, q1
# CHECK: encoding: [0xa4,0xd3,0xcc]
ee.vunzip.8 q5, q1

# CHECK-INST:  ee.vzip.16  q2, q0
# CHECK: encoding: [0xb4,0x23,0xcc]
ee.vzip.16 q2, q0

# CHECK-INST:  ee.vzip.32  q0, q3
# CHECK: encoding: [0xc4,0x83,0xdc]
ee.vzip.32 q0, q3

# CHECK-INST:  ee.vzip.8  q4, q5
# CHECK: encoding: [0xd4,0xc3,0xec]
ee.vzip.8 q4, q5

# CHECK-INST:  ee.xorq  q1, q3, q4
# CHECK: encoding: [0x54,0xb9,0xcd]
ee.xorq q1, q3, q4

# CHECK-INST:  ee.zero.accx
# CHECK: encoding: [0x04,0x08,0x25]
ee.zero.accx

# CHECK-INST:  ee.zero.q  q0
# CHECK: encoding: [0xa4,0x7f,0xcd]
ee.zero.q q0

# CHECK-INST:  ee.zero.qacc
# CHECK: encoding: [0x44,0x08,0x25]
ee.zero.qacc

# CHECK-INST:  rur a11, accx_0
# CHECK: encoding: [0x00,0xb0,0xe3]
rur.accx_0 a11

# CHECK-INST:  rur a11, accx_1
# CHECK: encoding: [0x10,0xb0,0xe3]
rur.accx_1 a11

# CHECK-INST:  rur a11, fft_bit_width
# CHECK: encoding: [0xe0,0xb0,0xe3]
rur.fft_bit_width a11

# CHECK-INST:  rur a3, gpio_out
# CHECK: encoding: [0xc0,0x30,0xe3]
rur.gpio_out a3

# CHECK-INST:  rur a1, qacc_h_0
# CHECK: encoding: [0x20,0x10,0xe3]
rur.qacc_h_0 a1

# CHECK-INST:  rur a10, qacc_h_1
# CHECK: encoding: [0x30,0xa0,0xe3]
rur.qacc_h_1 a10

# CHECK-INST:  rur a2, qacc_h_2
# CHECK: encoding: [0x40,0x20,0xe3]
rur.qacc_h_2 a2

# CHECK-INST:  rur a11, qacc_h_3
# CHECK: encoding: [0x50,0xb0,0xe3]
rur.qacc_h_3 a11

# CHECK-INST:  rur a13, qacc_h_4
# CHECK: encoding: [0x60,0xd0,0xe3]
rur.qacc_h_4 a13

# CHECK-INST:  rur a8, qacc_l_0
# CHECK: encoding: [0x70,0x80,0xe3]
rur.qacc_l_0 a8

# CHECK-INST:  rur a7, qacc_l_1
# CHECK: encoding: [0x80,0x70,0xe3]
rur.qacc_l_1 a7

# CHECK-INST:  rur a2, qacc_l_2
# CHECK: encoding: [0x90,0x20,0xe3]
rur.qacc_l_2 a2

# CHECK-INST:  rur a13, qacc_l_3
# CHECK: encoding: [0xa0,0xd0,0xe3]
rur.qacc_l_3 a13

# CHECK-INST:  rur a7, qacc_l_4
# CHECK: encoding: [0xb0,0x70,0xe3]
rur.qacc_l_4 a7

# CHECK-INST:  rur a9, sar_byte
# CHECK: encoding: [0xd0,0x90,0xe3]
rur.sar_byte a9

# CHECK-INST:  rur a12, ua_state_0
# CHECK: encoding: [0xf0,0xc0,0xe3]
rur.ua_state_0 a12

# CHECK-INST:  rur a2, ua_state_1
# CHECK: encoding: [0x00,0x21,0xe3]
rur.ua_state_1 a2

# CHECK-INST:  rur a5, ua_state_2
# CHECK: encoding: [0x10,0x51,0xe3]
rur.ua_state_2 a5

# CHECK-INST:  rur a3, ua_state_3
# CHECK: encoding: [0x20,0x31,0xe3]
rur.ua_state_3 a3

# CHECK-INST:  wur a6, accx_0
# CHECK: encoding: [0x60,0x00,0xf3]
wur.accx_0 a6

# CHECK-INST:  wur a6, accx_1
# CHECK: encoding: [0x60,0x01,0xf3]
wur.accx_1 a6

# CHECK-INST:  wur a13, fft_bit_width
# CHECK: encoding: [0xd0,0x0e,0xf3]
wur.fft_bit_width a13

# CHECK-INST:  wur a0, gpio_out
# CHECK: encoding: [0x00,0x0c,0xf3]
wur.gpio_out a0

# CHECK-INST:  wur a12, qacc_h_0
# CHECK: encoding: [0xc0,0x02,0xf3]
wur.qacc_h_0 a12

# CHECK-INST:  wur a1, qacc_h_1
# CHECK: encoding: [0x10,0x03,0xf3]
wur.qacc_h_1 a1

# CHECK-INST:  wur a2, qacc_h_2
# CHECK: encoding: [0x20,0x04,0xf3]
wur.qacc_h_2 a2

# CHECK-INST:  wur a12, qacc_h_3
# CHECK: encoding: [0xc0,0x05,0xf3]
wur.qacc_h_3 a12

# CHECK-INST:  wur a14, qacc_h_4
# CHECK: encoding: [0xe0,0x06,0xf3]
wur.qacc_h_4 a14

# CHECK-INST:  wur a6, qacc_l_0
# CHECK: encoding: [0x60,0x07,0xf3]
wur.qacc_l_0 a6

# CHECK-INST:  wur a5, qacc_l_1
# CHECK: encoding: [0x50,0x08,0xf3]
wur.qacc_l_1 a5

# CHECK-INST:  wur a6, qacc_l_2
# CHECK: encoding: [0x60,0x09,0xf3]
wur.qacc_l_2 a6

# CHECK-INST:  wur a6, qacc_l_3
# CHECK: encoding: [0x60,0x0a,0xf3]
wur.qacc_l_3 a6

# CHECK-INST:  wur a7, qacc_l_4
# CHECK: encoding: [0x70,0x0b,0xf3]
wur.qacc_l_4 a7

# CHECK-INST:  wur a9, sar_byte
# CHECK: encoding: [0x90,0x0d,0xf3]
wur.sar_byte a9

# CHECK-INST:  wur a8, ua_state_0
# CHECK: encoding: [0x80,0x0f,0xf3]
wur.ua_state_0 a8

# CHECK-INST:  wur a14, ua_state_1
# CHECK: encoding: [0xe0,0x10,0xf3]
wur.ua_state_1 a14

# CHECK-INST:  wur a9, ua_state_2
# CHECK: encoding: [0x90,0x11,0xf3]
wur.ua_state_2 a9

# CHECK-INST:  wur a10, ua_state_3
# CHECK: encoding: [0xa0,0x12,0xf3]
wur.ua_state_3 a10

# CHECK-INST:  mv.qr  q4, q7
# CHECK: encoding: [0x24,0x0c,0xaf]
mv.qr q4, q7
