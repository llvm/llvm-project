# RUN: llvm-mc -triple i386 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: vp2intersectq	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
         vp2intersectq	k0, zmm1, zmm2

# CHECK: vp2intersectq	k0, zmm1, zmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
         vp2intersectq	k0, zmm1, zmmword ptr [edi]

# CHECK: vp2intersectq	k0, zmm1, qword ptr [edi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
         vp2intersectq	k0, zmm1, qword ptr [edi]{1to8}

# CHECK: vp2intersectq	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
         vp2intersectq	k0, zmm1, zmm2

# CHECK: vp2intersectq	k0, zmm1, zmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
         vp2intersectq	k0, zmm1, zmmword ptr [edi]

# CHECK: vp2intersectq	k0, zmm1, qword ptr [edi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
         vp2intersectq	k0, zmm1, qword ptr [edi]{1to8}

# CHECK: vp2intersectq	k6, zmm4, zmm7
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0xf7]
         vp2intersectq	k6, zmm4, zmm7

# CHECK: vp2intersectq	k6, zmm4, zmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0x36]
         vp2intersectq	k6, zmm4, zmmword ptr [esi]

# CHECK: vp2intersectq	k6, zmm4, qword ptr [esi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xdf,0x58,0x68,0x36]
         vp2intersectq	k6, zmm4, qword ptr [esi]{1to8}

# CHECK: vp2intersectq	k6, zmm4, zmm7
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0xf7]
         vp2intersectq	k6, zmm4, zmm7

# CHECK: vp2intersectq	k6, zmm4, zmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0xdf,0x48,0x68,0x36]
         vp2intersectq	k6, zmm4, zmmword ptr [esi]

# CHECK: vp2intersectq	k6, zmm4, qword ptr [esi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xdf,0x58,0x68,0x36]
         vp2intersectq	k6, zmm4, qword ptr [esi]{1to8}

# CHECK: vp2intersectq	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
         vp2intersectq	k0, ymm1, ymm2

# CHECK: vp2intersectq	k0, ymm1, ymmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
         vp2intersectq	k0, ymm1, ymmword ptr [edi]

# CHECK: vp2intersectq	k0, ymm1, qword ptr [edi]{1to4}
# CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
         vp2intersectq	k0, ymm1, qword ptr [edi]{1to4}

# CHECK: vp2intersectq	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
         vp2intersectq	k0, ymm1, ymm2

# CHECK: vp2intersectq	k0, ymm1, ymmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
         vp2intersectq	k0, ymm1, ymmword ptr [edi]

# CHECK: vp2intersectq	k0, ymm1, qword ptr [edi]{1to4}
# CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
         vp2intersectq	k0, ymm1, qword ptr [edi]{1to4}

# CHECK: vp2intersectq	k6, ymm4, ymm7
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0xf7]
         vp2intersectq	k6, ymm4, ymm7

# CHECK: vp2intersectq	k6, ymm4, ymmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0x36]
         vp2intersectq	k6, ymm4, ymmword ptr [esi]

# CHECK: vp2intersectq	k6, ymm4, qword ptr [esi]{1to4}
# CHECK: encoding: [0x62,0xf2,0xdf,0x38,0x68,0x36]
         vp2intersectq	k6, ymm4, qword ptr [esi]{1to4}

# CHECK: vp2intersectq	k6, ymm4, ymm7
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0xf7]
         vp2intersectq	k6, ymm4, ymm7

# CHECK: vp2intersectq	k6, ymm4, ymmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0xdf,0x28,0x68,0x36]
         vp2intersectq	k6, ymm4, ymmword ptr [esi]

# CHECK: vp2intersectq	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
         vp2intersectq	k0, xmm1, xmm2

# CHECK: vp2intersectq	k0, xmm1, xmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
         vp2intersectq	k0, xmm1, xmmword ptr [edi]

# CHECK: vp2intersectq	k0, xmm1, qword ptr [edi]{1to2}
# CHECK: encoding: [0x62,0xf2,0xf7,0x18,0x68,0x07]
         vp2intersectq	k0, xmm1, qword ptr [edi]{1to2}

# CHECK: vp2intersectq	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
         vp2intersectq	k0, xmm1, xmm2

# CHECK: vp2intersectq	k0, xmm1, xmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
         vp2intersectq	k0, xmm1, xmmword ptr [edi]

# CHECK: vp2intersectq	k6, xmm4, xmm7
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0xf7]
         vp2intersectq	k6, xmm4, xmm7

# CHECK: vp2intersectq	k6, xmm4, xmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0x36]
         vp2intersectq	k6, xmm4, xmmword ptr [esi]

# CHECK: vp2intersectq	k6, xmm4, xmm7
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0xf7]
         vp2intersectq	k6, xmm4, xmm7

# CHECK: vp2intersectq	k6, xmm4, xmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0xdf,0x08,0x68,0x36]
         vp2intersectq	k6, xmm4, xmmword ptr [esi]

# CHECK: vp2intersectd	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
         vp2intersectd	k0, zmm1, zmm2

# CHECK: vp2intersectd	k0, zmm1, zmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
         vp2intersectd	k0, zmm1, zmmword ptr [edi]

# CHECK: vp2intersectd	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
         vp2intersectd	k0, zmm1, zmm2

# CHECK: vp2intersectd	k0, zmm1, zmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
         vp2intersectd	k0, zmm1, zmmword ptr [edi]

# CHECK: vp2intersectd	k6, zmm4, zmm7
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0xf7]
         vp2intersectd	k6, zmm4, zmm7

# CHECK: vp2intersectd	k6, zmm4, zmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0x36]
         vp2intersectd	k6, zmm4, zmmword ptr [esi]

# CHECK: vp2intersectd	k6, zmm4, zmm7
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0xf7]
         vp2intersectd	k6, zmm4, zmm7

# CHECK: vp2intersectd	k6, zmm4, zmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0x5f,0x48,0x68,0x36]
         vp2intersectd	k6, zmm4, zmmword ptr [esi]

# CHECK: vp2intersectd	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
         vp2intersectd	k0, ymm1, ymm2

# CHECK: vp2intersectd	k0, ymm1, ymmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
         vp2intersectd	k0, ymm1, ymmword ptr [edi]

# CHECK: vp2intersectd	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
         vp2intersectd	k0, ymm1, ymm2

# CHECK: vp2intersectd	k0, ymm1, ymmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
         vp2intersectd	k0, ymm1, ymmword ptr [edi]

# CHECK: vp2intersectd	k6, ymm4, ymm7
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0xf7]
         vp2intersectd	k6, ymm4, ymm7

# CHECK: vp2intersectd	k6, ymm4, ymmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0x36]
         vp2intersectd	k6, ymm4, ymmword ptr [esi]

# CHECK: vp2intersectd	k6, ymm4, ymm7
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0xf7]
         vp2intersectd	k6, ymm4, ymm7

# CHECK: vp2intersectd	k6, ymm4, ymmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0x5f,0x28,0x68,0x36]
         vp2intersectd	k6, ymm4, ymmword ptr [esi]

# CHECK: vp2intersectd	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
         vp2intersectd	k0, xmm1, xmm2

# CHECK: vp2intersectd	k0, xmm1, xmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
         vp2intersectd	k0, xmm1, xmmword ptr [edi]

# CHECK: vp2intersectd	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
         vp2intersectd	k0, xmm1, xmm2

# CHECK: vp2intersectd	k0, xmm1, xmmword ptr [edi]
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
         vp2intersectd	k0, xmm1, xmmword ptr [edi]

# CHECK: vp2intersectd	k6, xmm4, xmm7
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0xf7]
         vp2intersectd	k6, xmm4, xmm7

# CHECK: vp2intersectd	k6, xmm4, xmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0x36]
         vp2intersectd	k6, xmm4, xmmword ptr [esi]

# CHECK: vp2intersectd	k6, xmm4, xmm7
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0xf7]
         vp2intersectd	k6, xmm4, xmm7

# CHECK: vp2intersectd	k6, xmm4, xmmword ptr [esi]
# CHECK: encoding: [0x62,0xf2,0x5f,0x08,0x68,0x36]
         vp2intersectd	k6, xmm4, xmmword ptr [esi]
