# RUN: llvm-mc -triple i386 -show-encoding %s | FileCheck %s

# CHECK: vp2intersectq	%zmm2, %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
         vp2intersectq	%zmm2, %zmm1, %k0

# CHECK: vp2intersectq	(%edi), %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
         vp2intersectq	(%edi), %zmm1, %k0

# CHECK: vp2intersectq	(%edi){1to8}, %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
         vp2intersectq	(%edi){1to8}, %zmm1, %k0

# CHECK: vp2intersectq	%zmm2, %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
         vp2intersectq	%zmm2, %zmm1, %k0

# CHECK: vp2intersectq	(%edi), %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
         vp2intersectq	(%edi), %zmm1, %k0

# CHECK: vp2intersectq	(%edi){1to8}, %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
         vp2intersectq	(%edi){1to8}, %zmm1, %k0

# CHECK: vp2intersectq	%zmm7, %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0xf7]
         vp2intersectq	%zmm7, %zmm4, %k6

# CHECK: vp2intersectq	(%esi), %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0x36]
         vp2intersectq	(%esi), %zmm4, %k6

# CHECK: vp2intersectq	(%esi){1to8}, %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x58,0x68,0x36]
         vp2intersectq	(%esi){1to8}, %zmm4, %k6

# CHECK: vp2intersectq	%zmm7, %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0xf7]
         vp2intersectq	%zmm7, %zmm4, %k6

# CHECK: vp2intersectq	(%esi), %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0x36]
         vp2intersectq	(%esi), %zmm4, %k6

# CHECK: vp2intersectq	(%esi){1to8}, %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x58,0x68,0x36]
         vp2intersectq	(%esi){1to8}, %zmm4, %k6

# CHECK: vp2intersectq	%ymm2, %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
         vp2intersectq	%ymm2, %ymm1, %k0

# CHECK: vp2intersectq	(%edi), %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
         vp2intersectq	(%edi), %ymm1, %k0

# CHECK: vp2intersectq	(%edi){1to4}, %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
         vp2intersectq	(%edi){1to4}, %ymm1, %k0

# CHECK: vp2intersectq	%ymm2, %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
         vp2intersectq	%ymm2, %ymm1, %k0

# CHECK: vp2intersectq	(%edi), %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
         vp2intersectq	(%edi), %ymm1, %k0

# CHECK: vp2intersectq	(%edi){1to4}, %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
         vp2intersectq	(%edi){1to4}, %ymm1, %k0

# CHECK: vp2intersectq	%ymm7, %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0xf7]
         vp2intersectq	%ymm7, %ymm4, %k6

# CHECK: vp2intersectq	(%esi), %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0x36]
         vp2intersectq	(%esi), %ymm4, %k6

# CHECK: vp2intersectq	(%esi){1to4}, %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x38,0x68,0x36]
         vp2intersectq	(%esi){1to4}, %ymm4, %k6

# CHECK: vp2intersectq	%ymm7, %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0xf7]
         vp2intersectq	%ymm7, %ymm4, %k6

# CHECK: vp2intersectq	(%esi), %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0x36]
         vp2intersectq	(%esi), %ymm4, %k6

# CHECK: vp2intersectq	%xmm2, %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
         vp2intersectq	%xmm2, %xmm1, %k0

# CHECK: vp2intersectq	(%edi), %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
         vp2intersectq	(%edi), %xmm1, %k0

# CHECK: vp2intersectq	(%edi){1to2}, %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x18,0x68,0x07]
         vp2intersectq	(%edi){1to2}, %xmm1, %k0

# CHECK: vp2intersectq	%xmm2, %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
         vp2intersectq	%xmm2, %xmm1, %k0

# CHECK: vp2intersectq	(%edi), %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
         vp2intersectq	(%edi), %xmm1, %k0

# CHECK: vp2intersectq	%xmm7, %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0xf7]
         vp2intersectq	%xmm7, %xmm4, %k6

# CHECK: vp2intersectq	(%esi), %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0x36]
         vp2intersectq	(%esi), %xmm4, %k6

# CHECK: vp2intersectq	%xmm7, %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0xf7]
         vp2intersectq	%xmm7, %xmm4, %k6

# CHECK: vp2intersectq	(%esi), %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0x36]
         vp2intersectq	(%esi), %xmm4, %k6

# CHECK: vp2intersectd	%zmm2, %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
         vp2intersectd	%zmm2, %zmm1, %k0

# CHECK: vp2intersectd	(%edi), %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
         vp2intersectd	(%edi), %zmm1, %k0

# CHECK: vp2intersectd	%zmm2, %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
         vp2intersectd	%zmm2, %zmm1, %k0

# CHECK: vp2intersectd	(%edi), %zmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
         vp2intersectd	(%edi), %zmm1, %k0

# CHECK: vp2intersectd	%zmm7, %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0xf7]
         vp2intersectd	%zmm7, %zmm4, %k6

# CHECK: vp2intersectd	(%esi), %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0x36]
         vp2intersectd	(%esi), %zmm4, %k6

# CHECK: vp2intersectd	%zmm7, %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0xf7]
         vp2intersectd	%zmm7, %zmm4, %k6

# CHECK: vp2intersectd	(%esi), %zmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0x36]
         vp2intersectd	(%esi), %zmm4, %k6

# CHECK: vp2intersectd	%ymm2, %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
         vp2intersectd	%ymm2, %ymm1, %k0

# CHECK: vp2intersectd	(%edi), %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
         vp2intersectd	(%edi), %ymm1, %k0

# CHECK: vp2intersectd	%ymm2, %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
         vp2intersectd	%ymm2, %ymm1, %k0

# CHECK: vp2intersectd	(%edi), %ymm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
         vp2intersectd	(%edi), %ymm1, %k0

# CHECK: vp2intersectd	%ymm7, %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0xf7]
         vp2intersectd	%ymm7, %ymm4, %k6

# CHECK: vp2intersectd	(%esi), %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0x36]
         vp2intersectd	(%esi), %ymm4, %k6

# CHECK: vp2intersectd	%ymm7, %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0xf7]
         vp2intersectd	%ymm7, %ymm4, %k6

# CHECK: vp2intersectd	(%esi), %ymm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0x36]
         vp2intersectd	(%esi), %ymm4, %k6

# CHECK: vp2intersectd	%xmm2, %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
         vp2intersectd	%xmm2, %xmm1, %k0

# CHECK: vp2intersectd	(%edi), %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
         vp2intersectd	(%edi), %xmm1, %k0

# CHECK: vp2intersectd	%xmm2, %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
         vp2intersectd	%xmm2, %xmm1, %k0

# CHECK: vp2intersectd	(%edi), %xmm1, %k0
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
         vp2intersectd	(%edi), %xmm1, %k0

# CHECK: vp2intersectd	%xmm7, %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0xf7]
         vp2intersectd	%xmm7, %xmm4, %k6

# CHECK: vp2intersectd	(%esi), %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0x36]
         vp2intersectd	(%esi), %xmm4, %k6

# CHECK: vp2intersectd	%xmm7, %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0xf7]
         vp2intersectd	%xmm7, %xmm4, %k6

# CHECK: vp2intersectd	(%esi), %xmm4, %k6
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0x36]
         vp2intersectd	(%esi), %xmm4, %k6
