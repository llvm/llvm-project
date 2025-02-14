# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

## MRMDestMem

# CHECK: vextractf32x4	xmmword ptr [r16 + r17], zmm0, 1
# CHECK: encoding: [0x62,0xfb,0x79,0x48,0x19,0x04,0x08,0x01]
         vextractf32x4	xmmword ptr [r16 + r17], zmm0, 1

# CHECK: add	r18, qword ptr [r17 + 123], r16
# CHECK: encoding: [0x62,0xec,0xec,0x10,0x01,0x41,0x7b]
         add	r18, qword ptr [r17 + 123], r16

## MRMDestMemCC

# CHECK: cfcmovb	qword ptr [r17 + 4*r18 + 123], r16
# CHECK: encoding: [0x62,0xec,0xf8,0x0c,0x42,0x44,0x91,0x7b]
         cfcmovb	qword ptr [r17 + 4*r18 + 123], r16

## MRMSrcMem

# CHECK: vbroadcasti32x4	zmm0, xmmword ptr [r16 + r17]
# CHECK: encoding: [0x62,0xfa,0x79,0x48,0x5a,0x04,0x08]
         vbroadcasti32x4	zmm0, xmmword ptr [r16 + r17]

# CHECK: sub	r18, r17, qword ptr [r16 + 123]
# CHECK: encoding: [0x62,0xec,0xec,0x10,0x2b,0x48,0x7b]
         sub	r18, r17, qword ptr [r16 + 123]

## MRMSrcMemCC

# CHECK: cfcmovb	r18, qword ptr [r16 + 4*r17 + 123]
# CHECK: encoding: [0x62,0xec,0xf8,0x08,0x42,0x54,0x88,0x7b]
         cfcmovb	r18, qword ptr [r16 + 4*r17 + 123]

# CHECK: cfcmovb	r19, r18, qword ptr [r16 + 4*r17 + 123]
# CHECK: encoding: [0x62,0xec,0xe0,0x14,0x42,0x54,0x88,0x7b]
         cfcmovb	r19, r18, qword ptr [r16 + 4*r17 + 123]

## MRM0m

# CHECK: vprorq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x04,0x08,0x00]
         vprorq	zmm0, zmmword ptr [r16 + r17], 0

# CHECK: add	r17, qword ptr [r16 + 123], 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0x40,0x7b,0x7f]
         add	r17, qword ptr [r16 + 123], 127

## MRM1m

# CHECK: vprolq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x0c,0x08,0x00]
         vprolq	zmm0, zmmword ptr [r16 + r17], 0

# CHECK: or	r17, qword ptr [r16 + 123], 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0x48,0x7b,0x7f]
         or	r17, qword ptr [r16 + 123], 127

## MRM2m

# CHECK: vpsrlq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x73,0x14,0x08,0x00]
         vpsrlq	zmm0, zmmword ptr [r16 + r17], 0

# CHECK: adc	r17, qword ptr [r16 + 123], 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0x50,0x7b,0x7f]
         adc	r17, qword ptr [r16 + 123], 127

## MRM3m

# CHECK: vpsrldq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0x79,0x48,0x73,0x1c,0x08,0x00]
         vpsrldq	zmm0, zmmword ptr [r16 + r17], 0

# CHECK: sbb	r17, qword ptr [r16 + 123], 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0x58,0x7b,0x7f]
         sbb	r17, qword ptr [r16 + 123], 127

## MRM4m

# CHECK: vpsraq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x24,0x08,0x00]
         vpsraq	zmm0, zmmword ptr [r16 + r17], 0

# CHECK: and	r17, qword ptr [r16 + 123], 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0x60,0x7b,0x7f]
         and	r17, qword ptr [r16 + 123], 127

## MRM5m
## AsmParser is buggy for this KNC instruction
# COM: CHECK: vscatterpf0dps	{k1}, zmmword ptr [r16 + zmm0]
# COM: CHECK: encoding: [0x62,0xfa,0x7d,0x49,0xc6,0x2c,0x00]
#         vscatterpf0dps	{k1}, zmmword ptr [r16 + zmm0]

# CHECK: sub	r17, qword ptr [r16 + 123], 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0x68,0x7b,0x7f]
         sub	r17, qword ptr [r16 + 123], 127

## MRM6m

# CHECK: vpsllq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x73,0x34,0x08,0x00]
         vpsllq	zmm0, zmmword ptr [r16 + r17], 0

# CHECK: xor	r17, qword ptr [r16 + 123], 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0x70,0x7b,0x7f]
         xor	r17, qword ptr [r16 + 123], 127

## MRM7m

# CHECK: vpslldq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0x79,0x48,0x73,0x3c,0x08,0x00]
         vpslldq	zmm0, zmmword ptr [r16 + r17], 0

# CHECK: sar	r18, qword ptr [r16 + r17 + 291], 123
# CHECK: encoding: [0x62,0xfc,0xe8,0x10,0xc1,0xbc,0x08,0x23,0x01,0x00,0x00,0x7b]
         sar	r18, qword ptr [r16 + r17 + 291], 123

## MRMDestMem4VOp3CC

# CHECK: cmpbexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d
# CHECK: encoding: [0x62,0x8a,0x69,0x00,0xe6,0xb4,0xac,0x23,0x01,0x00,0x00]
         cmpbexadd	dword ptr [r28 + 4*r29 + 291], r22d, r18d

## MRMSrcMem4VOp3

# CHECK: bzhi	r23, qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         bzhi	r23, qword ptr [r28 + 4*r29 + 291], r19

## MRMDestReg

# CHECK: vextractps	r16d, xmm16, 1
# CHECK: encoding: [0x62,0xeb,0x7d,0x08,0x17,0xc0,0x01]
         vextractps	r16d, xmm16, 1

# CHECK: {nf}	add	r17, r16
# CHECK: encoding: [0x62,0xec,0xfc,0x0c,0x01,0xc1]
         {nf}	add	r17, r16

## MRMDestRegCC

# CHECK: cfcmovb	r17, r16
# CHECK: encoding: [0x62,0xec,0xfc,0x0c,0x42,0xc1]
         cfcmovb	r17, r16

## MRMSrcReg

# CHECK: mulx	r18, r17, r16
# CHECK: encoding: [0x62,0xea,0xf7,0x00,0xf6,0xd0]
         mulx	r18, r17, r16

## MRMSrcRegCC

# CHECK: cfcmovb	r18, r17, r16
# CHECK: encoding: [0x62,0xec,0xec,0x14,0x42,0xc8]
         cfcmovb	r18, r17, r16

## MRMSrcReg4VOp3

# CHECK: bzhi	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xe4,0x00,0xf5,0xdf]
         bzhi	r27, r23, r19

## MRM0r

# CHECK: add	r17, r16, 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0xc0,0x7f]
         add	r17, r16, 127

## MRM1r

# CHECK: or	r17, r16, 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0xc8,0x7f]
         or	r17, r16, 127

## MRM2r

# CHECK: adc	r17, r16, 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0xd0,0x7f]
         adc	r17, r16, 127

## MRM3r

# CHECK: sbb	r17, r16, 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0xd8,0x7f]
         sbb	r17, r16, 127

## MRM4r

# CHECK: and	r17, r16, 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0xe0,0x7f]
         and	r17, r16, 127

## MRM5r

# CHECK: sub	r17, r16, 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0xe8,0x7f]
         sub	r17, r16, 127

## MRM6r

# CHECK: xor	r17, r16, 127
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0x83,0xf0,0x7f]
         xor	r17, r16, 127

## MRM7r

# CHECK: sar	r17, r16, 123
# CHECK: encoding: [0x62,0xfc,0xf4,0x10,0xc1,0xf8,0x7b]
         sar	r17, r16, 123

## MRMXrCC
# CHECK: setzuo	r16b
# CHECK: encoding: [0x62,0xfc,0x7f,0x18,0x40,0xc0]
         setzuo r16b

## MRMXmCC
# CHECK: setzuo byte ptr [r16 + r17]
# CHECK: encoding: [0x62,0xfc,0x7b,0x18,0x40,0x04,0x08]
         setzuo byte ptr [r16 + r17]

## NoCD8

# CHECK: {nf}	neg	qword ptr [r16 + 123]
# CHECK: encoding: [0x62,0xfc,0xfc,0x0c,0xf7,0x58,0x7b]
         {nf}	neg	qword ptr [r16 + 123]

# CHECK: {evex}	not	qword ptr [r16 + 123]
# CHECK: encoding: [0x62,0xfc,0xfc,0x08,0xf7,0x50,0x7b]
         {evex}	not	qword ptr [r16 + 123]
