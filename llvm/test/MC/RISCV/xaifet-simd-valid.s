# XAIFET - AI Foundry ET SIMD instructions

# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xaifet %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xaifet %s \
# RUN:        | llvm-objdump -d --mattr=+xaifet - | FileCheck %s --check-prefix=CHECK-DISASM

aif.bitmixb	s7, gp, gp
// CHECK-ENCODING: aif.bitmixb	s7, gp, gp              # encoding: [0xbb,0xfb,0x31,0x80]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 8031fbbb     	aif.bitmixb	s7, gp, gp

aif.cubefaceidx.ps	fa0, fs9, ft9
// CHECK-ENCODING: aif.cubefaceidx.ps	fa0, fs9, ft9   # encoding: [0x7b,0x95,0xdc,0x89]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 89dc957b     	aif.cubefaceidx.ps	fa0, fs9, ft9

aif.cubeface.ps	ft9, fa3, ft10
// CHECK-ENCODING: aif.cubeface.ps	ft9, fa3, ft10          # encoding: [0xfb,0x8e,0xe6,0x89]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 89e68efb     	aif.cubeface.ps	ft9, fa3, ft10

aif.cubesgnsc.ps	ft9, ft8, fs9
// CHECK-ENCODING: aif.cubesgnsc.ps	ft9, ft8, fs9           # encoding: [0xfb,0x2e,0x9e,0x89]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 899e2efb     	aif.cubesgnsc.ps	ft9, ft8, fs9

aif.cubesgntc.ps	fs2, ft6, fa6
// CHECK-ENCODING: aif.cubesgntc.ps	fs2, ft6, fa6           # encoding: [0x7b,0x39,0x03,0x89]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 8903397b     	aif.cubesgntc.ps	fs2, ft6, fa6

#====----------------------------------------------------------------------===//
# SIMD FP|INT instructions
#====----------------------------------------------------------------------===//

aif.faddi.pi	fa4, ft1, -96
// CHECK-ENCODING: aif.faddi.pi	fa4, ft1, -96           # encoding: [0x3f,0x87,0x00,0xec]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: ec00873f     	aif.faddi.pi	fa4, ft1, -0x60

aif.fadd.pi	fs10, fa5, fs8
// CHECK-ENCODING: aif.fadd.pi	fs10, fa5, fs8          # encoding: [0x7b,0x8d,0x87,0x07]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 07878d7b     	aif.fadd.pi	fs10, fa5, fs8

aif.fadd.ps	fs0, ft0, fs4, rtz
// CHECK-ENCODING: aif.fadd.ps	fs0, ft0, fs4, rtz      # encoding: [0x7b,0x14,0x40,0x01]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 0140147b     	aif.fadd.ps	fs0, ft0, fs4, rtz

aif.fandi.pi	fa7, fs1, 16
// CHECK-ENCODING: aif.fandi.pi	fa7, fs1, 16            # encoding: [0xbf,0x98,0x04,0x05]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 050498bf     	aif.fandi.pi	fa7, fs1, 0x10

aif.fand.pi	ft2, fs8, fa1
// CHECK-ENCODING: aif.fand.pi	ft2, fs8, fa1           # encoding: [0x7b,0x71,0xbc,0x06]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 06bc717b     	aif.fand.pi	ft2, fs8, fa1

aif.fbci.pi	fs9, 1015523
// CHECK-ENCODING: aif.fbci.pi	fs9, 1015523            # encoding: [0xdf,0x3c,0xee,0xf7]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: f7ee3cdf     	aif.fbci.pi	fs9, 0xf7ee3

aif.fbci.ps	fs9, 946477
// CHECK-ENCODING: aif.fbci.ps	fs9, 946477             # encoding: [0x9f,0xdc,0x12,0xe7]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: e712dc9f     	aif.fbci.ps	fs9, 0xe712d

aif.fbcx.ps	fa3, s3
// CHECK-ENCODING: aif.fbcx.ps	fa3, s3                 # encoding: [0x8b,0xb6,0x09,0x00]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 0009b68b     	aif.fbcx.ps	fa3, s3

aif.fbc.ps	fs0, 1529(s9)
// CHECK-ENCODING: aif.fbc.ps	fs0, 1529(s9)           # encoding: [0x0b,0x84,0x9c,0x5f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 5f9c840b     	aif.fbc.ps	fs0, 0x5f9(s9)

aif.fclass.ps	fs0, fs8
// CHECK-ENCODING: aif.fclass.ps	fs0, fs8                # encoding: [0x7b,0x14,0x0c,0xe0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: e00c147b     	aif.fclass.ps	fs0, fs8

aif.fcmovm.ps	ft5, fa7, fs6
// CHECK-ENCODING: aif.fcmovm.ps	ft5, fa7, fs6           # encoding: [0xf7,0x82,0x68,0x01]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 016882f7     	aif.fcmovm.ps	ft5, fa7, fs6

aif.fcmov.ps	fs11, ft1, ft8, fa1
// CHECK-ENCODING: aif.fcmov.ps	fs11, ft1, ft8, fa1     # encoding: [0xbf,0xad,0xc0,0x5d]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 5dc0adbf     	aif.fcmov.ps	fs11, ft1, ft8, fa1

aif.fcvt.f10.ps	fa2, ft1
// CHECK-ENCODING: aif.fcvt.f10.ps	fa2, ft1                # encoding: [0x7b,0x86,0xb0,0xd8]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d8b0867b     	aif.fcvt.f10.ps	fa2, ft1

aif.fcvt.f11.ps	fa0, ft4
// CHECK-ENCODING: aif.fcvt.f11.ps	fa0, ft4                # encoding: [0x7b,0x05,0x82,0xd8]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d882057b     	aif.fcvt.f11.ps	fa0, ft4

aif.fcvt.f16.ps	ft1, fa2
// CHECK-ENCODING: aif.fcvt.f16.ps	ft1, fa2                # encoding: [0xfb,0x00,0x96,0xd8]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d89600fb     	aif.fcvt.f16.ps	ft1, fa2

aif.fcvt.ps.f10	ft5, fa0
// CHECK-ENCODING: aif.fcvt.ps.f10	ft5, fa0                # encoding: [0xfb,0x02,0x85,0xd0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d08502fb     	aif.fcvt.ps.f10	ft5, fa0

aif.fcvt.ps.f11	fa0, fa6
// CHECK-ENCODING: aif.fcvt.ps.f11	fa0, fa6                # encoding: [0x7b,0x05,0x98,0xd0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d098057b     	aif.fcvt.ps.f11	fa0, fa6

aif.fcvt.ps.f16	fa4, fs0
// CHECK-ENCODING: aif.fcvt.ps.f16	fa4, fs0                # encoding: [0x7b,0x07,0xa4,0xd0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d0a4077b     	aif.fcvt.ps.f16	fa4, fs0

aif.fcvt.ps.pw	fs0, ft1, rtz
// CHECK-ENCODING: aif.fcvt.ps.pw	fs0, ft1, rtz           # encoding: [0x7b,0x94,0x00,0xd0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d000947b     	aif.fcvt.ps.pw	fs0, ft1, rtz

aif.fcvt.ps.pwu	ft8, ft0, rtz
// CHECK-ENCODING: aif.fcvt.ps.pwu	ft8, ft0, rtz           # encoding: [0x7b,0x1e,0x10,0xd0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d0101e7b     	aif.fcvt.ps.pwu	ft8, ft0, rtz

aif.fcvt.ps.rast	ft7, fa0
// CHECK-ENCODING: aif.fcvt.ps.rast	ft7, fa0                # encoding: [0xfb,0x03,0x25,0xd0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d02503fb     	aif.fcvt.ps.rast	ft7, fa0

aif.fcvt.ps.sn16	ft11, fa2
// CHECK-ENCODING: aif.fcvt.ps.sn16	ft11, fa2               # encoding: [0xfb,0x0f,0x96,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d1960ffb     	aif.fcvt.ps.sn16	ft11, fa2

aif.fcvt.ps.sn8	fs1, ft2
// CHECK-ENCODING: aif.fcvt.ps.sn8	fs1, ft2                # encoding: [0xfb,0x04,0xb1,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d1b104fb     	aif.fcvt.ps.sn8	fs1, ft2

aif.fcvt.ps.un10	fs0, fs11
// CHECK-ENCODING: aif.fcvt.ps.un10	fs0, fs11               # encoding: [0x7b,0x84,0x2d,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d12d847b     	aif.fcvt.ps.un10	fs0, fs11

aif.fcvt.ps.un16	fa7, ft5
// CHECK-ENCODING: aif.fcvt.ps.un16	fa7, ft5                # encoding: [0xfb,0x88,0x12,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d11288fb     	aif.fcvt.ps.un16	fa7, ft5

aif.fcvt.ps.un2	fs0, fa7
// CHECK-ENCODING: aif.fcvt.ps.un2	fs0, fa7                # encoding: [0x7b,0x84,0x78,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d178847b     	aif.fcvt.ps.un2	fs0, fa7

aif.fcvt.ps.un24	ft6, ft1
// CHECK-ENCODING: aif.fcvt.ps.un24	ft6, ft1                # encoding: [0x7b,0x83,0x00,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d100837b     	aif.fcvt.ps.un24	ft6, ft1

aif.fcvt.ps.un8	fa3, fa4
// CHECK-ENCODING: aif.fcvt.ps.un8	fa3, fa4                # encoding: [0xfb,0x06,0x37,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d13706fb     	aif.fcvt.ps.un8	fa3, fa4

aif.fcvt.pwu.ps	ft10, ft1, rup
// CHECK-ENCODING: aif.fcvt.pwu.ps	ft10, ft1, rup          # encoding: [0x7b,0xbf,0x10,0xc0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: c010bf7b     	aif.fcvt.pwu.ps	ft10, ft1, rup

aif.fcvt.pw.ps	fs6, ft5, rup
// CHECK-ENCODING: aif.fcvt.pw.ps	fs6, ft5, rup           # encoding: [0x7b,0xbb,0x02,0xc0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: c002bb7b     	aif.fcvt.pw.ps	fs6, ft5, rup

aif.fcvt.rast.ps	ft11, fs7
// CHECK-ENCODING: aif.fcvt.rast.ps	ft11, fs7               # encoding: [0xfb,0x8f,0x2b,0xc0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: c02b8ffb     	aif.fcvt.rast.ps	ft11, fs7

aif.fcvt.sn16.ps	ft1, fs7
// CHECK-ENCODING: aif.fcvt.sn16.ps	ft1, fs7                # encoding: [0xfb,0x80,0x9b,0xd9]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d99b80fb     	aif.fcvt.sn16.ps	ft1, fs7

aif.fcvt.sn8.ps	fs3, ft5
// CHECK-ENCODING: aif.fcvt.sn8.ps	fs3, ft5                # encoding: [0xfb,0x89,0xb2,0xd9]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d9b289fb     	aif.fcvt.sn8.ps	fs3, ft5

aif.fcvt.un10.ps	ft5, fa6
// CHECK-ENCODING: aif.fcvt.un10.ps	ft5, fa6                # encoding: [0xfb,0x02,0x28,0xd9]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d92802fb     	aif.fcvt.un10.ps	ft5, fa6

aif.fcvt.un16.ps	fs9, ft6
// CHECK-ENCODING: aif.fcvt.un16.ps	fs9, ft6                # encoding: [0xfb,0x0c,0x13,0xd9]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d9130cfb     	aif.fcvt.un16.ps	fs9, ft6

aif.fcvt.un24.ps	fs8, ft8
// CHECK-ENCODING: aif.fcvt.un24.ps	fs8, ft8                # encoding: [0x7b,0x0c,0x0e,0xd9]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d90e0c7b     	aif.fcvt.un24.ps	fs8, ft8

aif.fcvt.un2.ps	fa0, fa3
// CHECK-ENCODING: aif.fcvt.un2.ps	fa0, fa3                # encoding: [0x7b,0x85,0x76,0xd9]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d976857b     	aif.fcvt.un2.ps	fa0, fa3

aif.fcvt.un8.ps	fa0, ft2
// CHECK-ENCODING: aif.fcvt.un8.ps	fa0, ft2                # encoding: [0x7b,0x05,0x31,0xd9]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d931057b     	aif.fcvt.un8.ps	fa0, ft2

aif.fdivu.pi	fa1, fa5, fa2
// CHECK-ENCODING: aif.fdivu.pi	fa1, fa5, fa2           # encoding: [0xfb,0x95,0xc7,0x1e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 1ec795fb     	aif.fdivu.pi	fa1, fa5, fa2

aif.fdiv.pi	ft3, fa0, fs4
// CHECK-ENCODING: aif.fdiv.pi	ft3, fa0, fs4           # encoding: [0xfb,0x01,0x45,0x1f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 1f4501fb     	aif.fdiv.pi	ft3, fa0, fs4

aif.fdiv.ps	fs10, ft5, fs11, rdn
// CHECK-ENCODING: aif.fdiv.ps	fs10, ft5, fs11, rdn    # encoding: [0x7b,0xad,0xb2,0x19]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 19b2ad7b     	aif.fdiv.ps	fs10, ft5, fs11, rdn

aif.feqm.ps	m4, ft5, ft8
// CHECK-ENCODING: aif.feqm.ps	m4, ft5, ft8            # encoding: [0x7b,0xe2,0xc2,0xa1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a1c2e27b     	aif.feqm.ps	m4, ft5, ft8

aif.feq.pi	ft10, fa1, fs0
// CHECK-ENCODING: aif.feq.pi	ft10, fa1, fs0          # encoding: [0x7b,0xaf,0x85,0xa6]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a685af7b     	aif.feq.pi	ft10, fa1, fs0

aif.feq.ps	fa5, ft4, fs2
// CHECK-ENCODING: aif.feq.ps	fa5, ft4, fs2           # encoding: [0xfb,0x27,0x22,0xa1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a12227fb     	aif.feq.ps	fa5, ft4, fs2

aif.fexp.ps	fs10, ft3
// CHECK-ENCODING: aif.fexp.ps	fs10, ft3               # encoding: [0x7b,0x8d,0x41,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 58418d7b     	aif.fexp.ps	fs10, ft3

aif.flog.ps	fa0, fs10
// CHECK-ENCODING: aif.flog.ps	fa0, fs10               # encoding: [0x7b,0x05,0x3d,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 583d057b     	aif.flog.ps	fa0, fs10

aif.ffrc.ps	fs9, ft8
// CHECK-ENCODING: aif.ffrc.ps	fs9, ft8                # encoding: [0xfb,0x0c,0x2e,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 582e0cfb     	aif.ffrc.ps	fs9, ft8

aif.fg32b.ps	fa2, a0, (s0)
// CHECK-ENCODING: aif.fg32b.ps	fa2, a0, (s0)           # encoding: [0x0b,0x16,0x85,0x08]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 0885160b     	aif.fg32b.ps	fa2, a0, (s0)

aif.fg32h.ps	ft10, s6, (s9)
// CHECK-ENCODING: aif.fg32h.ps	ft10, s6, (s9)          # encoding: [0x0b,0x1f,0x9b,0x11]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 119b1f0b     	aif.fg32h.ps	ft10, s6, (s9)

aif.fg32w.ps	fs5, ra, (a0)
// CHECK-ENCODING: aif.fg32w.ps	fs5, ra, (a0)           # encoding: [0x8b,0x9a,0xa0,0x20]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 20a09a8b     	aif.fg32w.ps	fs5, ra, (a0)

aif.fgb.ps	fs11, ft7, (s4)
// CHECK-ENCODING: aif.fgb.ps	fs11, ft7, (s4)         # encoding: [0x8b,0x9d,0x43,0x49]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 49439d8b     	aif.fgb.ps	fs11, ft7, (s4)

aif.fgh.ps	ft3, ft7, (t5)
// CHECK-ENCODING: aif.fgh.ps	ft3, ft7, (t5)          # encoding: [0x8b,0x91,0xe3,0x51]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 51e3918b     	aif.fgh.ps	ft3, ft7, (t5)

aif.fgw.ps	fs10, fa7, (s3)
// CHECK-ENCODING: aif.fgw.ps	fs10, fa7, (s3)         # encoding: [0x0b,0x9d,0x38,0x61]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 61389d0b     	aif.fgw.ps	fs10, fa7, (s3)

aif.flem.ps	m6, ft4, ft3
// CHECK-ENCODING: aif.flem.ps	m6, ft4, ft3            # encoding: [0x7b,0x43,0x32,0xa0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a032437b     	aif.flem.ps	m6, ft4, ft3

aif.fle.pi	ft2, fs3, fa7
// CHECK-ENCODING: aif.fle.pi	ft2, fs3, fa7           # encoding: [0x7b,0x81,0x19,0xa7]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a719817b     	aif.fle.pi	ft2, fs3, fa7

aif.fle.ps	ft9, fs11, fa1
// CHECK-ENCODING: aif.fle.ps	ft9, fs11, fa1          # encoding: [0xfb,0x8e,0xbd,0xa0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a0bd8efb     	aif.fle.ps	ft9, fs11, fa1

aif.flq2	fa5, -2020(sp)
// CHECK-ENCODING: aif.flq2	fa5, -2020(sp)                  # encoding: [0x87,0x57,0xc1,0x81]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 81c15787     	aif.flq2	fa5, -0x7e4(sp)

aif.fltm.pi	m0, ft3, fs1
// CHECK-ENCODING: aif.fltm.pi	m0, ft3, fs1            # encoding: [0x7b,0x80,0x91,0x3e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 3e91807b     	aif.fltm.pi	m0, ft3, fs1

aif.fltm.ps	m6, ft2, fa3
// CHECK-ENCODING: aif.fltm.ps	m6, ft2, fa3            # encoding: [0x7b,0x53,0xd1,0xa0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a0d1537b     	aif.fltm.ps	m6, ft2, fa3

aif.fltu.pi	ft5, fa1, fs10
// CHECK-ENCODING: aif.fltu.pi	ft5, fa1, fs10          # encoding: [0xfb,0xb2,0xa5,0xa7]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a7a5b2fb     	aif.fltu.pi	ft5, fa1, fs10

aif.flt.pi	ft11, fa4, fs8
// CHECK-ENCODING: aif.flt.pi	ft11, fa4, fs8          # encoding: [0xfb,0x1f,0x87,0xa7]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a7871ffb     	aif.flt.pi	ft11, fa4, fs8

aif.flt.ps	fs11, fs3, fs5
// CHECK-ENCODING: aif.flt.ps	fs11, fs3, fs5          # encoding: [0xfb,0x9d,0x59,0xa1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a1599dfb     	aif.flt.ps	fs11, fs3, fs5

aif.flw.ps	ft9, 1224(s5)
// CHECK-ENCODING: aif.flw.ps	ft9, 1224(s5)           # encoding: [0x8b,0xae,0x8a,0x4c]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 4c8aae8b     	aif.flw.ps	ft9, 0x4c8(s5)

aif.fmadd.ps	fs0, fs2, ft8, ft8, rmm
// CHECK-ENCODING: aif.fmadd.ps	fs0, fs2, ft8, ft8, rmm # encoding: [0x5b,0x44,0xc9,0xe1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: e1c9445b     	aif.fmadd.ps	fs0, fs2, ft8, ft8, rmm

aif.fmaxu.pi	ft10, fs3, fs11
// CHECK-ENCODING: aif.fmaxu.pi	ft10, fs3, fs11         # encoding: [0x7b,0xbf,0xb9,0x2f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 2fb9bf7b     	aif.fmaxu.pi	ft10, fs3, fs11

aif.fmax.pi	ft3, fs1, fs6
// CHECK-ENCODING: aif.fmax.pi	ft3, fs1, fs6           # encoding: [0xfb,0x91,0x64,0x2f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 2f6491fb     	aif.fmax.pi	ft3, fs1, fs6

aif.fmax.ps	fa0, fs2, ft3
// CHECK-ENCODING: aif.fmax.ps	fa0, fs2, ft3           # encoding: [0x7b,0x15,0x39,0x28]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 2839157b     	aif.fmax.ps	fa0, fs2, ft3

aif.fminu.pi	ft5, ft2, ft2
// CHECK-ENCODING: aif.fminu.pi	ft5, ft2, ft2           # encoding: [0xfb,0x22,0x21,0x2e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 2e2122fb     	aif.fminu.pi	ft5, ft2, ft2

aif.fmin.pi	fs10, fs10, fs3
// CHECK-ENCODING: aif.fmin.pi	fs10, fs10, fs3         # encoding: [0x7b,0x0d,0x3d,0x2f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 2f3d0d7b     	aif.fmin.pi	fs10, fs10, fs3

aif.fmin.ps	fa1, ft5, ft5
// CHECK-ENCODING: aif.fmin.ps	fa1, ft5, ft5           # encoding: [0xfb,0x85,0x52,0x28]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 285285fb     	aif.fmin.ps	fa1, ft5, ft5

aif.fmsub.ps	fs0, fs7, fs6, ft3, rmm
// CHECK-ENCODING: aif.fmsub.ps	fs0, fs7, fs6, ft3, rmm # encoding: [0x5b,0xc4,0x6b,0x1b]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 1b6bc45b     	aif.fmsub.ps	fs0, fs7, fs6, ft3, rmm

aif.fmulhu.pi	ft6, fs7, fs4
// CHECK-ENCODING: aif.fmulhu.pi	ft6, fs7, fs4           # encoding: [0x7b,0xa3,0x4b,0x17]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 174ba37b     	aif.fmulhu.pi	ft6, fs7, fs4

aif.fmulh.pi	fa3, fa3, fs1
// CHECK-ENCODING: aif.fmulh.pi	fa3, fa3, fs1           # encoding: [0xfb,0x96,0x96,0x16]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 169696fb     	aif.fmulh.pi	fa3, fa3, fs1

aif.fmul.pi	ft4, ft7, ft4
// CHECK-ENCODING: aif.fmul.pi	ft4, ft7, ft4           # encoding: [0x7b,0x82,0x43,0x16]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 1643827b     	aif.fmul.pi	ft4, ft7, ft4

aif.fmul.ps	fa5, fa2, fs10, rtz
// CHECK-ENCODING: aif.fmul.ps	fa5, fa2, fs10, rtz     # encoding: [0xfb,0x17,0xa6,0x11]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 11a617fb     	aif.fmul.ps	fa5, fa2, fs10, rtz

aif.fmvs.x.ps	fp, fs0, 6
// CHECK-ENCODING: aif.fmvs.x.ps	s0, fs0, 6              # encoding: [0x7b,0x24,0x64,0xe0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: e064247b     	aif.fmvs.x.ps	s0, fs0, 0x6

aif.fmvz.x.ps	a5, ft6, 6
// CHECK-ENCODING: aif.fmvz.x.ps	a5, ft6, 6              # encoding: [0xfb,0x07,0x63,0xe0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: e06307fb     	aif.fmvz.x.ps	a5, ft6, 0x6

aif.fnmadd.ps	ft10, ft8, fs5, fs3, rdn
// CHECK-ENCODING: aif.fnmadd.ps	ft10, ft8, fs5, fs3, rdn # encoding: [0x5b,0x2f,0x5e,0x9f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 9f5e2f5b     	aif.fnmadd.ps	ft10, ft8, fs5, fs3, rdn

aif.fnmsub.ps	ft0, ft11, fs7, fa5, rtz
// CHECK-ENCODING: aif.fnmsub.ps	ft0, ft11, fs7, fa5, rtz # encoding: [0x5b,0x90,0x7f,0x7d]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 7d7f905b     	aif.fnmsub.ps	ft0, ft11, fs7, fa5, rtz

aif.fnot.pi	fs10, fs6
// CHECK-ENCODING: aif.fnot.pi	fs10, fs6               # encoding: [0x7b,0x2d,0x0b,0x06]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 060b2d7b     	aif.fnot.pi	fs10, fs6

aif.for.pi	ft10, fa0, fa4
// CHECK-ENCODING: aif.for.pi	ft10, fa0, fa4          # encoding: [0x7b,0x6f,0xe5,0x06]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 06e56f7b     	aif.for.pi	ft10, fa0, fa4

aif.fpackrepb.pi	ft8, fs11
// CHECK-ENCODING: aif.fpackrepb.pi	ft8, fs11               # encoding: [0x7b,0x8e,0x0d,0x26]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 260d8e7b     	aif.fpackrepb.pi	ft8, fs11

aif.fpackreph.pi	ft2, ft10
// CHECK-ENCODING: aif.fpackreph.pi	ft2, ft10               # encoding: [0x7b,0x11,0x0f,0x26]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 260f117b     	aif.fpackreph.pi	ft2, ft10

aif.frcp_fix.rast	ft3, fs0, fa1
// CHECK-ENCODING: aif.frcp_fix.rast	ft3, fs0, fa1   # encoding: [0xfb,0x01,0xb4,0x30]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 30b401fb     	aif.frcp_fix.rast	ft3, fs0, fa1

aif.frcp.ps	ft11, fs2
// CHECK-ENCODING: aif.frcp.ps	ft11, fs2               # encoding: [0xfb,0x0f,0x79,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 58790ffb     	aif.frcp.ps	ft11, fs2

aif.fremu.pi	ft4, ft4, ft1
// CHECK-ENCODING: aif.fremu.pi	ft4, ft4, ft1           # encoding: [0x7b,0x32,0x12,0x1e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 1e12327b     	aif.fremu.pi	ft4, ft4, ft1

aif.frem.pi	fa2, fs10, ft1
// CHECK-ENCODING: aif.frem.pi	fa2, fs10, ft1          # encoding: [0x7b,0x26,0x1d,0x1e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 1e1d267b     	aif.frem.pi	fa2, fs10, ft1

aif.fround.ps	fa3, fs7, rtz
// CHECK-ENCODING: aif.fround.ps	fa3, fs7, rtz           # encoding: [0xfb,0x96,0x1b,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 581b96fb     	aif.fround.ps	fa3, fs7, rtz

aif.frsq.ps	fs1, fs1
// CHECK-ENCODING: aif.frsq.ps	fs1, fs1                # encoding: [0xfb,0x84,0x84,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 588484fb     	aif.frsq.ps	fs1, fs1

aif.fsat8.pi	ft1, fs7
// CHECK-ENCODING: aif.fsat8.pi	ft1, fs7                # encoding: [0xfb,0xb0,0x0b,0x06]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 060bb0fb     	aif.fsat8.pi	ft1, fs7

aif.fsatu8.pi	fa1, fa2
// CHECK-ENCODING: aif.fsatu8.pi	fa1, fa2                # encoding: [0xfb,0x35,0x16,0x06]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 061635fb     	aif.fsatu8.pi	fa1, fa2

aif.fsc32b.ps	ft7, gp, (s3)
// CHECK-ENCODING: aif.fsc32b.ps	ft7, gp, (s3)           # encoding: [0x8b,0x93,0x31,0x89]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 8931938b     	aif.fsc32b.ps	ft7, gp, (s3)

aif.fsc32h.ps	fa3, s6, (tp)
// CHECK-ENCODING: aif.fsc32h.ps	fa3, s6, (tp)           # encoding: [0x8b,0x16,0x4b,0x90]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 904b168b     	aif.fsc32h.ps	fa3, s6, (tp)

aif.fsc32w.ps	ft0, t5, (a0)
// CHECK-ENCODING: aif.fsc32w.ps	ft0, t5, (a0)           # encoding: [0x0b,0x10,0xaf,0xa0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a0af100b     	aif.fsc32w.ps	ft0, t5, (a0)

aif.fscb.ps	ft9, ft1, (zero)
// CHECK-ENCODING: aif.fscb.ps	ft9, ft1, (zero)        # encoding: [0x8b,0x9e,0x00,0xc8]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: c8009e8b     	aif.fscb.ps	ft9, ft1, (zero)

aif.fsch.ps	fs2, fa0, (t5)
// CHECK-ENCODING: aif.fsch.ps	fs2, fa0, (t5)          # encoding: [0x0b,0x19,0xe5,0xd1]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d1e5190b     	aif.fsch.ps	fs2, fa0, (t5)

aif.fscw.ps	ft1, ft0, (a0)
// CHECK-ENCODING: aif.fscw.ps	ft1, ft0, (a0)          # encoding: [0x8b,0x10,0xa0,0xe0]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: e0a0108b     	aif.fscw.ps	ft1, ft0, (a0)

aif.fsetm.pi	m0, fa5
// CHECK-ENCODING: aif.fsetm.pi	m0, fa5                 # encoding: [0x7b,0xc0,0x07,0xa6]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: a607c07b     	aif.fsetm.pi	m0, fa5

aif.fsgnjn.ps	ft6, fs10, ft2
// CHECK-ENCODING: aif.fsgnjn.ps	ft6, fs10, ft2          # encoding: [0x7b,0x13,0x2d,0x20]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 202d137b     	aif.fsgnjn.ps	ft6, fs10, ft2

aif.fsgnjx.ps	ft9, fs2, fs0
// CHECK-ENCODING: aif.fsgnjx.ps	ft9, fs2, fs0           # encoding: [0xfb,0x2e,0x89,0x20]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 20892efb     	aif.fsgnjx.ps	ft9, fs2, fs0

aif.fsgnj.ps	fs3, fa7, ft0
// CHECK-ENCODING: aif.fsgnj.ps	fs3, fa7, ft0           # encoding: [0xfb,0x89,0x08,0x20]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 200889fb     	aif.fsgnj.ps	fs3, fa7, ft0

aif.fsin.ps	ft2, ft4
// CHECK-ENCODING: aif.fsin.ps	ft2, ft4                # encoding: [0x7b,0x01,0x62,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 5862017b     	aif.fsin.ps	ft2, ft4

aif.fslli.pi	ft1, fs10, 25
// CHECK-ENCODING: aif.fslli.pi	ft1, fs10, 25           # encoding: [0xfb,0x10,0x9d,0x4f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 4f9d10fb     	aif.fslli.pi	ft1, fs10, 0x19

aif.fsll.pi	ft3, ft9, fa7
// CHECK-ENCODING: aif.fsll.pi	ft3, ft9, fa7           # encoding: [0xfb,0x91,0x1e,0x07]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 071e91fb     	aif.fsll.pi	ft3, ft9, fa7

aif.fsq2	fa4, -1100(s5)
// CHECK-ENCODING: aif.fsq2	fa4, -1100(s5)                  # encoding: [0x27,0xda,0xea,0xba]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: baeada27     	aif.fsq2	fa4, -0x44c(s5)

aif.fsqrt.ps	ft2, fs11
// CHECK-ENCODING: aif.fsqrt.ps	ft2, fs11               # encoding: [0x7b,0x81,0x0d,0x58]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 580d817b     	aif.fsqrt.ps	ft2, fs11

aif.fsrai.pi	ft1, fs5, 5
// CHECK-ENCODING: aif.fsrai.pi	ft1, fs5, 5             # encoding: [0xfb,0xf0,0x5a,0x4e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 4e5af0fb     	aif.fsrai.pi	ft1, fs5, 0x5

aif.fsra.pi	fs1, ft11, ft2
// CHECK-ENCODING: aif.fsra.pi	fs1, ft11, ft2          # encoding: [0xfb,0xd4,0x2f,0x0e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 0e2fd4fb     	aif.fsra.pi	fs1, ft11, ft2

aif.fsrli.pi	fs0, fa4, 5
// CHECK-ENCODING: aif.fsrli.pi	fs0, fa4, 5             # encoding: [0x7b,0x54,0x57,0x4e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 4e57547b     	aif.fsrli.pi	fs0, fa4, 0x5

aif.fsrl.pi	fa2, fs0, fa4
// CHECK-ENCODING: aif.fsrl.pi	fa2, fs0, fa4           # encoding: [0x7b,0x56,0xe4,0x06]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 06e4567b     	aif.fsrl.pi	fa2, fs0, fa4

aif.fsub.pi	ft0, ft10, ft9
// CHECK-ENCODING: aif.fsub.pi	ft0, ft10, ft9          # encoding: [0x7b,0x00,0xdf,0x0f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 0fdf007b     	aif.fsub.pi	ft0, ft10, ft9

aif.fsub.ps	ft2, ft2, fs8, dyn
// CHECK-ENCODING: aif.fsub.ps	ft2, ft2, fs8           # encoding: [0x7b,0x71,0x81,0x09]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 0981717b     	aif.fsub.ps	ft2, ft2, fs8

aif.fswizz.ps	ft3, fa4, 188
// CHECK-ENCODING: aif.fswizz.ps	ft3, fa4, 188           # encoding: [0xfb,0x41,0x77,0xe7]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: e77741fb     	aif.fswizz.ps	ft3, fa4, 0xbc

aif.fsw.ps	fs4, 1772(a1)
// CHECK-ENCODING: aif.fsw.ps	fs4, 1772(a1)           # encoding: [0x0b,0xe6,0x45,0x6f]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 6f45e60b     	aif.fsw.ps	fs4, 0x6ec(a1)

aif.fxor.pi	ft10, fa6, fs6
// CHECK-ENCODING: aif.fxor.pi	ft10, fa6, fs6          # encoding: [0x7b,0x4f,0x68,0x07]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 07684f7b     	aif.fxor.pi	ft10, fa6, fs6

#====----------------------------------------------------------------------===//
# Masking instructions
#====----------------------------------------------------------------------===//

aif.maskand	m3, m5, m0
// CHECK-ENCODING: aif.maskand	m3, m5, m0              # encoding: [0xfb,0xf1,0x02,0x66]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 6602f1fb     	aif.maskand	m3, m5, m0

aif.masknot	m6, m6
// CHECK-ENCODING: aif.masknot	m6, m6                  # encoding: [0x7b,0x23,0x03,0x66]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 6603237b     	aif.masknot	m6, m6

aif.maskor	m0, m7, m5
// CHECK-ENCODING: aif.maskor	m0, m7, m5              # encoding: [0x7b,0xe0,0x53,0x66]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 6653e07b     	aif.maskor	m0, m7, m5

aif.maskpopc	s9, m7
// CHECK-ENCODING: aif.maskpopc	s9, m7                  # encoding: [0xfb,0x8c,0x03,0x52]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 52038cfb     	aif.maskpopc	s9, m7

aif.maskpopcz	s10, m4
// CHECK-ENCODING: aif.maskpopcz	s10, m4                 # encoding: [0x7b,0x0d,0x02,0x54]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 54020d7b     	aif.maskpopcz	s10, m4

aif.maskpopc.rast	m1, m3, m2, 2
// CHECK-ENCODING: aif.maskpopc.rast	m1, m3, m2, 2   # encoding: [0xfb,0x80,0x29,0x5e]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 5e2980fb     	aif.maskpopc.rast	m1, m3, m2, 0x2

aif.maskxor	m6, m6, m2
// CHECK-ENCODING: aif.maskxor	m6, m6, m2              # encoding: [0x7b,0x43,0x23,0x66]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 6623437b     	aif.maskxor	m6, m6, m2

aif.mova.m.x	t1
// CHECK-ENCODING: aif.mova.m.x	t1                      # encoding: [0x7b,0x10,0x03,0xd6]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d603107b     	aif.mova.m.x	t1

aif.mova.x.m	s1
// CHECK-ENCODING: aif.mova.x.m	s1                      # encoding: [0xfb,0x04,0x00,0xd6]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: d60004fb     	aif.mova.x.m	s1

aif.mov.m.x	m3, t0, 201
// CHECK-ENCODING: aif.mov.m.x	m3, t0, 201             # encoding: [0xfb,0x91,0x92,0x57]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 579291fb     	aif.mov.m.x	m3, t0, 0xc9

aif.packb	a1, a2, a4
// CHECK-ENCODING: aif.packb	a1, a2, a4              # encoding: [0xbb,0x65,0xe6,0x80]
// CHECK-ERROR: :[[@LINE-2]]:1: error: instruction requires the following: 'XAIFET' (AI Foundry ET Extension)
// CHECK-DISASM: 80e665bb     	aif.packb	a1, a2, a4
