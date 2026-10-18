# RUN: llvm-mc -triple=agx -mcpu=agx-g13g -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=agx -mcpu=agx-g14c -show-encoding %s | FileCheck %s

# Encodings cross-checked against the applegpu reference assembler and
# disassembler (dougallj/applegpu): each case is a random canonical encoding
# that applegpu disassembles to the same text.

ceil.sat r74h, u79h
# CHECK: ceil.sat r74h, u79h{{ +}}// encoding: [0x4a,0xaa,0x1f,0x01,0x01,0x28]
ceil r90l, r118h.neg
# CHECK: ceil r90l, r118h.neg{{ +}}// encoding: [0x0a,0xe8,0x6d,0x08,0x01,0x2c]
ceil.sat r10.cache, r59h.abs.neg
# CHECK: ceil.sat r10.cache, r59h.abs.neg{{ +}}// encoding: [0xca,0xa9,0x77,0x0c,0x01,0x04]
ceil r94l, u165.neg
# CHECK: ceil r94l, u165.neg{{ +}}// encoding: [0x0a,0xf8,0xca,0x09,0x01,0x24]



exp2.sat r54.cache, r0.cache.abs
# CHECK: exp2.sat r54.cache, r0.cache.abs{{ +}}// encoding: [0xca,0xd9,0x80,0xd6,0x00,0x10]
exp2.sat r103l.cache, u118
# CHECK: exp2.sat r103l.cache, u118{{ +}}// encoding: [0xca,0x9c,0xac,0xd1,0x00,0x3c]
exp2 r112h.cache, r95.abs
# CHECK: exp2 r112h.cache, r95.abs{{ +}}// encoding: [0x8a,0xc2,0x7e,0xd6,0x00,0x38]
exp2.sat r109.cache, u106h
# CHECK: exp2.sat r109.cache, u106h{{ +}}// encoding: [0xca,0xb5,0x15,0xd1,0x00,0x3c]

fadd16.sat r16h.cache, 7.5, u143h.neg
# CHECK: fadd16.sat r16h.cache, 7.5, u143h.neg{{ +}}// encoding: [0xe6,0xc2,0x1e,0xf0,0x55,0x04]
fadd16.sat r52l.cache, u24l.abs, r117h.discard
# CHECK: fadd16.sat r52l.cache, u24l.abs, r117h.discard{{ +}}// encoding: [0xe6,0xd0,0x30,0xb3,0x0e,0x13]
fadd16 r80h.cache, r12h, u105l.neg
# CHECK: fadd16 r80h.cache, r12h, u105l.neg{{ +}}// encoding: [0xa6,0xc2,0x59,0x20,0x51,0x23]
fadd16.sat r127l.cache, r120h.cache.abs.neg, u155l
# CHECK: fadd16.sat r127l.cache, r120h.cache.abs.neg, u155l{{ +}}// encoding: [0xe6,0xfc,0xb1,0x66,0x17,0x3c]

fadd32 r8l.cache, -7.5, r42.discard.abs.neg
# CHECK: fadd32 r8l.cache, -7.5, r42.discard.abs.neg{{ +}}// encoding: [0xaa,0xa0,0x1e,0x40,0xed,0x0d]
fadd32.sat r102h.cache, r71h.discard.neg, r10h.discard.abs
# CHECK: fadd32.sat r102h.cache, r71h.discard.neg, r10h.discard.abs{{ +}}// encoding: [0xea,0x9a,0xcf,0x58,0x4d,0x38]
fadd32 r124h, r54.discard.abs.neg, r54.discard
# CHECK: fadd32 r124h, r54.discard.abs.neg, r54.discard{{ +}}// encoding: [0x2a,0xf2,0xec,0xce,0x2e,0x35]
fadd32 r16h.cache, u89h.abs, u90h.abs
# CHECK: fadd32 r16h.cache, u89h.abs, u90h.abs{{ +}}// encoding: [0xaa,0xc2,0x33,0x55,0x53,0x0a]

floor.sat r98l, r36h.cache.abs
# CHECK: floor.sat r98l, r36h.cache.abs{{ +}}// encoding: [0x4a,0x88,0x89,0x04,0x00,0x34]
floor.sat r44h.cache, r31h.cache
# CHECK: floor.sat r44h.cache, r31h.cache{{ +}}// encoding: [0xca,0xb2,0xbf,0x00,0x00,0x10]
floor r121, r45l.cache.neg
# CHECK: floor r121, r45l.cache.neg{{ +}}// encoding: [0x0a,0xe5,0x9a,0x08,0x00,0x34]
floor.sat r18l, r118l.cache.neg
# CHECK: floor.sat r18l, r118l.cache.neg{{ +}}// encoding: [0x4a,0xc8,0xac,0x08,0x00,0x0c]

fmadd16 r56l.cache, r31l.neg, r86h.discard.neg, u253l.neg
# CHECK: fmadd16 r56l.cache, r31l.neg, r86h.discard.neg, u253l.neg{{ +}}// encoding: [0xb6,0xe0,0x7e,0xd4,0x4e,0x7a,0xc5,0x12]
fmadd16.sat r101l, r14l.abs, u154l, u150.abs
# CHECK: fmadd16.sat r101l, r14l.abs, u154l, u150.abs{{ +}}// encoding: [0x76,0x94,0x5c,0x42,0x17,0xec,0x03,0x30]
fmadd16 r5h, u163.neg, r27h.abs.neg, u217.abs.neg
# CHECK: fmadd16 r5h, u163.neg, r27h.abs.neg, u217.abs.neg{{ +}}// encoding: [0x36,0x96,0xc6,0x75,0x67,0xf2,0x87,0x04]
fmadd16.sat r50l.cache, u180l.neg, u99.neg, r94h.cache
# CHECK: fmadd16.sat r50l.cache, u180l.neg, u99.neg, r94h.cache{{ +}}// encoding: [0xf6,0xc8,0x68,0x65,0x58,0xbd,0x80,0x17]

fmadd32 r4.cache, u178l.abs.neg, u219l.abs.neg, r106h.abs.neg
# CHECK: fmadd32 r4.cache, u178l.abs.neg, u219l.abs.neg, r106h.abs.neg{{ +}}// encoding: [0xba,0x91,0x64,0x6d,0xd7,0x55,0xcc,0x06]
fmadd32 r34.cache, r45h.neg, r35h.cache.neg, u192l.abs.neg
# CHECK: fmadd32 r34.cache, r45h.neg, r35h.cache.neg, u192l.abs.neg{{ +}}// encoding: [0xba,0x89,0x5b,0x78,0x88,0x40,0x8d,0x15]
fmadd32.sat r101.cache, r44.cache.abs.neg, u16h.abs.neg, r104.discard
# CHECK: fmadd32.sat r101.cache, r44.cache.abs.neg, u16h.abs.neg, r104.discard{{ +}}// encoding: [0xfa,0x95,0x98,0x1e,0xd2,0xd0,0xc2,0x34]
fmadd32 r38h, u205l.abs.neg, r70l.cache.neg, r121.cache
# CHECK: fmadd32 r38h, u205l.abs.neg, r70l.cache.neg, r121.cache{{ +}}// encoding: [0x3a,0x9a,0x5a,0xcd,0x88,0xb2,0xc2,0x1a]

fmul16 r73l.cache, r77l, u193
# CHECK: fmul16 r73l.cache, r77l, u193{{ +}}// encoding: [0x96,0xa4,0x5a,0x20,0x1c,0x2a]
fmul16.sat r91h.cache, u46h.abs.neg, u167h.abs.neg
# CHECK: fmul16.sat r91h.cache, u46h.abs.neg, u167h.abs.neg{{ +}}// encoding: [0xd6,0xee,0x1d,0xf7,0x74,0x25]
fmul16 r16l, r17l.discard.neg, r42l.abs.neg
# CHECK: fmul16 r16l, r17l.discard.neg, r42l.abs.neg{{ +}}// encoding: [0x16,0xc0,0xe2,0x44,0x65,0x01]
fmul16.sat r107l.cache, u156l, r79l.abs.neg
# CHECK: fmul16.sat r107l.cache, u156l, r79l.abs.neg{{ +}}// encoding: [0xd6,0xac,0x78,0xe1,0x65,0x32]

fmul32 r28l, u6.abs, r7.neg
# CHECK: fmul32 r28l, u6.abs, r7.neg{{ +}}// encoding: [0x1a,0xf0,0x8c,0xe5,0xa4,0x00]
fmul32.sat r16h.cache, u208.neg, u196l.neg
# CHECK: fmul32.sat r16h.cache, u208.neg, u196l.neg{{ +}}// encoding: [0xda,0xc2,0xe0,0x89,0x94,0x0a]
fmul32 r65.cache, u42h.neg, u208h.neg
# CHECK: fmul32 r65.cache, u42h.neg, u208h.neg{{ +}}// encoding: [0x9a,0x85,0x15,0x19,0x96,0x26]
fmul32.sat r15l.cache, r38h.discard.neg, u124
# CHECK: fmul32.sat r15l.cache, r38h.discard.neg, u124{{ +}}// encoding: [0xda,0xbc,0xcd,0x88,0x1b,0x07]

log2 r121l.cache, r7.abs
# CHECK: log2 r121l.cache, r7.abs{{ +}}// encoding: [0x8a,0xe4,0x4e,0xc6,0x00,0x30]
log2.sat r83.cache, r91.abs
# CHECK: log2.sat r83.cache, r91.abs{{ +}}// encoding: [0xca,0xcd,0x76,0xc6,0x00,0x28]
log2.sat r45.cache, u69
# CHECK: log2.sat r45.cache, u69{{ +}}// encoding: [0xca,0xb5,0x8a,0xc1,0x00,0x18]
log2.sat r71l, r122h.neg
# CHECK: log2.sat r71l, r122h.neg{{ +}}// encoding: [0x4a,0x9c,0x75,0xc8,0x00,0x2c]

rcp.sat r91.cache, u245h
# CHECK: rcp.sat r91.cache, u245h{{ +}}// encoding: [0xca,0xed,0x6b,0x81,0x00,0x2c]
rcp.sat r39.cache, r107.cache.abs
# CHECK: rcp.sat r39.cache, r107.cache.abs{{ +}}// encoding: [0xca,0x9d,0x96,0x86,0x00,0x1c]
rcp r112h.cache, u107.abs.neg
# CHECK: rcp r112h.cache, u107.abs.neg{{ +}}// encoding: [0x8a,0xc2,0x96,0x8d,0x00,0x3c]
rcp.sat r80l, r55l.discard.abs.neg
# CHECK: rcp.sat r80l, r55l.discard.abs.neg{{ +}}// encoding: [0x4a,0xc0,0xee,0x8c,0x00,0x24]

rint r11l.cache, u97l.abs.neg
# CHECK: rint r11l.cache, u97l.abs.neg{{ +}}// encoding: [0x8a,0xac,0x02,0x0d,0x03,0x0c]
rint.sat r90.cache, u85l.abs
# CHECK: rint.sat r90.cache, u85l.abs{{ +}}// encoding: [0xca,0xe9,0x2a,0x05,0x03,0x28]
rint r104l.cache, r13l.abs.neg
# CHECK: rint r104l.cache, r13l.abs.neg{{ +}}// encoding: [0x8a,0xa0,0x5a,0x0c,0x03,0x30]
rint r85h.cache, u152
# CHECK: rint r85h.cache, u152{{ +}}// encoding: [0x8a,0xd6,0xf0,0x01,0x03,0x20]

rsqrt.sat r97, r102h.cache.neg
# CHECK: rsqrt.sat r97, r102h.cache.neg{{ +}}// encoding: [0x4a,0x85,0x8d,0x98,0x00,0x3c]
rsqrt.sat r114l.cache, u62h.abs
# CHECK: rsqrt.sat r114l.cache, u62h.abs{{ +}}// encoding: [0xca,0xc8,0x3d,0x95,0x00,0x34]
rsqrt.sat r84.cache, u91l
# CHECK: rsqrt.sat r84.cache, u91l{{ +}}// encoding: [0xca,0xd1,0x36,0x91,0x00,0x28]
rsqrt r73, r10l.neg
# CHECK: rsqrt r73, r10l.neg{{ +}}// encoding: [0x0a,0xa5,0x54,0x98,0x00,0x20]

rsqrt_special r46h.cache, r26l
# CHECK: rsqrt_special r46h.cache, r26l{{ +}}// encoding: [0x8a,0xba,0x74,0x10,0x00,0x10]
rsqrt_special r93, u120
# CHECK: rsqrt_special r93, u120{{ +}}// encoding: [0x0a,0xf5,0xb0,0x11,0x00,0x2c]
rsqrt_special.sat r1.cache, u148l.abs.neg
# CHECK: rsqrt_special.sat r1.cache, u148l.abs.neg{{ +}}// encoding: [0xca,0x05,0x68,0x1d]
rsqrt_special.sat r53h, u24h.neg
# CHECK: rsqrt_special.sat r53h, u24h.neg{{ +}}// encoding: [0x4a,0xd6,0x31,0x19,0x00,0x10]

sin_pt_1.sat r62l.cache, r94.cache
# CHECK: sin_pt_1.sat r62l.cache, r94.cache{{ +}}// encoding: [0xca,0xf8,0xbc,0xa2,0x00,0x18]
sin_pt_1.sat r14.cache, u170h.neg
# CHECK: sin_pt_1.sat r14.cache, u170h.neg{{ +}}// encoding: [0xca,0xb9,0x55,0xa9,0x00,0x04]
sin_pt_1.sat r79l, u6
# CHECK: sin_pt_1.sat r79l, u6{{ +}}// encoding: [0x4a,0xbc,0x8c,0xa1,0x00,0x20]
sin_pt_1 r115l.cache, u44
# CHECK: sin_pt_1 r115l.cache, u44{{ +}}// encoding: [0x8a,0xcc,0x98,0xa1,0x00,0x34]

sin_pt_2.sat r31h, r21h.discard.abs
# CHECK: sin_pt_2.sat r31h, r21h.discard.abs{{ +}}// encoding: [0x4a,0x7e,0xeb,0xe4]
sin_pt_2.sat r48l, r104h.discard.abs.neg
# CHECK: sin_pt_2.sat r48l, r104h.discard.abs.neg{{ +}}// encoding: [0x4a,0xc0,0xd1,0xec,0x00,0x1c]
sin_pt_2 r118, r68l.cache.abs.neg
# CHECK: sin_pt_2 r118, r68l.cache.abs.neg{{ +}}// encoding: [0x0a,0xd9,0x88,0xec,0x00,0x38]
sin_pt_2 r82l, r12h.cache.abs.neg
# CHECK: sin_pt_2 r82l, r12h.cache.abs.neg{{ +}}// encoding: [0x0a,0xc8,0x99,0xec,0x00,0x20]

trunc.sat r82, u57l.abs
# CHECK: trunc.sat r82, u57l.abs{{ +}}// encoding: [0x4a,0xc9,0x32,0x05,0x02,0x24]
trunc.sat r30, u117l.neg
# CHECK: trunc.sat r30, u117l.neg{{ +}}// encoding: [0x4a,0xf9,0x2a,0x09,0x02,0x0c]
trunc.sat r124l.cache, r55l.discard.abs
# CHECK: trunc.sat r124l.cache, r55l.discard.abs{{ +}}// encoding: [0xca,0xf0,0xee,0x04,0x02,0x34]
trunc r126h.cache, u57.abs
# CHECK: trunc r126h.cache, u57.abs{{ +}}// encoding: [0x8a,0xfa,0xb2,0x05,0x02,0x34]

