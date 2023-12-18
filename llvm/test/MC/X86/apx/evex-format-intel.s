## NOTE: This file needs to be updated after promoted instruction is supported
# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

## MRMDestMem

# CHECK: vextractf32x4	xmmword ptr [r16 + r17], zmm0, 1
# CHECK: encoding: [0x62,0xfb,0x79,0x48,0x19,0x04,0x08,0x01]
         vextractf32x4	xmmword ptr [r16 + r17], zmm0, 1

## MRMSrcMem

# CHECK: vbroadcasti32x4	zmm0, xmmword ptr [r16 + r17]
# CHECK: encoding: [0x62,0xfa,0x79,0x48,0x5a,0x04,0x08]
         vbroadcasti32x4	zmm0, xmmword ptr [r16 + r17]

## MRM0m

# CHECK: vprorq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x04,0x08,0x00]
         vprorq	zmm0, zmmword ptr [r16 + r17], 0

## MRM1m

# CHECK: vprolq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x0c,0x08,0x00]
         vprolq	zmm0, zmmword ptr [r16 + r17], 0

## MRM2m

# CHECK: vpsrlq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x73,0x14,0x08,0x00]
         vpsrlq	zmm0, zmmword ptr [r16 + r17], 0

## MRM3m

# CHECK: vpsrldq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0x79,0x48,0x73,0x1c,0x08,0x00]
         vpsrldq	zmm0, zmmword ptr [r16 + r17], 0
## MRM4m

# CHECK: vpsraq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x24,0x08,0x00]
         vpsraq	zmm0, zmmword ptr [r16 + r17], 0

## MRM5m
## AsmParser is buggy for this KNC instruction
# C;HECK: vscatterpf0dps	{k1}, zmmword ptr [r16 + zmm0]
# C;HECK: encoding: [0x62,0xfa,0x7d,0x49,0xc6,0x2c,0x00]
#         vscatterpf0dps	{k1}, zmmword ptr [r16 + zmm0]

## MRM6m

# CHECK: vpsllq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x73,0x34,0x08,0x00]
         vpsllq	zmm0, zmmword ptr [r16 + r17], 0

## MRM7m

# CHECK: vpslldq	zmm0, zmmword ptr [r16 + r17], 0
# CHECK: encoding: [0x62,0xf9,0x79,0x48,0x73,0x3c,0x08,0x00]
         vpslldq	zmm0, zmmword ptr [r16 + r17], 0

## MRMSrcMem4VOp3

# CHECK: bzhi	r23, qword ptr [r28 + 4*r29 + 291], r19
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         bzhi	r23, qword ptr [r28 + 4*r29 + 291], r19

## MRMDestReg

# CHECK: vextractps	r16d, xmm16, 1
# CHECK: encoding: [0x62,0xeb,0x7d,0x08,0x17,0xc0,0x01]
         vextractps	r16d, xmm16, 1

## MRMSrcReg4VOp3

# CHECK: bzhi	r27, r23, r19
# CHECK: encoding: [0x62,0x6a,0xe4,0x00,0xf5,0xdf]
         bzhi	r27, r23, r19
