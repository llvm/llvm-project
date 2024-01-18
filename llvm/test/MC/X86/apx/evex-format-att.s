## NOTE: This file needs to be updated after promoted instruction is supported
# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s

## MRMDestMem

# CHECK: vextractf32x4	$1, %zmm0, (%r16,%r17)
# CHECK: encoding: [0x62,0xfb,0x79,0x48,0x19,0x04,0x08,0x01]
         vextractf32x4	$1, %zmm0, (%r16,%r17)

## MRMSrcMem

# CHECK: vbroadcasti32x4	(%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xfa,0x79,0x48,0x5a,0x04,0x08]
         vbroadcasti32x4	(%r16,%r17), %zmm0

## MRM0m

# CHECK: vprorq	$0, (%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x04,0x08,0x00]
         vprorq	$0, (%r16,%r17), %zmm0

## MRM1m

# CHECK: vprolq	$0, (%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x0c,0x08,0x00]
         vprolq	$0, (%r16,%r17), %zmm0
## MRM2m

# CHECK: vpsrlq	$0, (%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x73,0x14,0x08,0x00]
         vpsrlq	$0, (%r16,%r17), %zmm0

## MRM3m

# CHECK: vpsrldq	$0, (%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xf9,0x79,0x48,0x73,0x1c,0x08,0x00]
         vpsrldq	$0, (%r16,%r17), %zmm0

## MRM4m

# CHECK: vpsraq	$0, (%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x72,0x24,0x08,0x00]
         vpsraq	$0, (%r16,%r17), %zmm0

## MRM5m

# CHECK: vscatterpf0dps	(%r16,%zmm0) {%k1}
# CHECK: encoding: [0x62,0xfa,0x7d,0x49,0xc6,0x2c,0x00]
         vscatterpf0dps	(%r16,%zmm0) {%k1}

## MRM6m

# CHECK: vpsllq	$0, (%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xf9,0xf9,0x48,0x73,0x34,0x08,0x00]
         vpsllq	$0, (%r16,%r17), %zmm0

## MRM7m

# CHECK: vpslldq	$0, (%r16,%r17), %zmm0
# CHECK: encoding: [0x62,0xf9,0x79,0x48,0x73,0x3c,0x08,0x00]
         vpslldq	$0, (%r16,%r17), %zmm0

## MRMSrcMem4VOp3

# CHECK: bzhiq	%r19, 291(%r28,%r29,4), %r23
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         bzhiq	%r19, 291(%r28,%r29,4), %r23

## MRMDestReg

# CHECK: vextractps	$1, %xmm16, %r16d
# CHECK: encoding: [0x62,0xeb,0x7d,0x08,0x17,0xc0,0x01]
         vextractps	$1, %xmm16, %r16d

## MRMSrcReg4VOp3

# CHECK: bzhiq	%r19, %r23, %r27
# CHECK: encoding: [0x62,0x6a,0xe4,0x00,0xf5,0xdf]
         bzhiq	%r19, %r23, %r27
