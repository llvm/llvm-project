# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s


# CHECK: vp2intersectq	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
         vp2intersectq	k0, zmm1, zmm2

# CHECK: vp2intersectq	k0, zmm1, zmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
         vp2intersectq	k0, zmm1, zmmword ptr [rdi]

# CHECK: vp2intersectq	k0, zmm1, qword ptr [rdi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
         vp2intersectq	k0, zmm1, qword ptr [rdi]{1to8}

# CHECK: vp2intersectq	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0xc2]
         vp2intersectq	k0, zmm1, zmm2

# CHECK: vp2intersectq	k0, zmm1, zmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x48,0x68,0x07]
         vp2intersectq	k0, zmm1, zmmword ptr [rdi]

# CHECK: vp2intersectq	k0, zmm1, qword ptr [rdi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xf7,0x58,0x68,0x07]
         vp2intersectq	k0, zmm1, qword ptr [rdi]{1to8}

# CHECK: vp2intersectq	k6, zmm9, zmm7
# CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0xf7]
         vp2intersectq	k6, zmm9, zmm7

# CHECK: vp2intersectq	k6, zmm9, zmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0x36]
         vp2intersectq	k6, zmm9, zmmword ptr [rsi]

# CHECK: vp2intersectq	k6, zmm9, qword ptr [rsi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xb7,0x58,0x68,0x36]
         vp2intersectq	k6, zmm9, qword ptr [rsi]{1to8}

# CHECK: vp2intersectq	k6, zmm9, zmm7
# CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0xf7]
         vp2intersectq	k6, zmm9, zmm7

# CHECK: vp2intersectq	k6, zmm9, zmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0xb7,0x48,0x68,0x36]
         vp2intersectq	k6, zmm9, zmmword ptr [rsi]

# CHECK: vp2intersectq	k6, zmm9, qword ptr [rsi]{1to8}
# CHECK: encoding: [0x62,0xf2,0xb7,0x58,0x68,0x36]
         vp2intersectq	k6, zmm9, qword ptr [rsi]{1to8}

# CHECK: vp2intersectq	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
         vp2intersectq	k0, ymm1, ymm2

# CHECK: vp2intersectq	k0, ymm1, ymmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
         vp2intersectq	k0, ymm1, ymmword ptr [rdi]

# CHECK: vp2intersectq	k0, ymm1, qword ptr [rdi]{1to4}
# CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
         vp2intersectq	k0, ymm1, qword ptr [rdi]{1to4}

# CHECK: vp2intersectq	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0xc2]
         vp2intersectq	k0, ymm1, ymm2

# CHECK: vp2intersectq	k0, ymm1, ymmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x28,0x68,0x07]
         vp2intersectq	k0, ymm1, ymmword ptr [rdi]

# CHECK: vp2intersectq	k0, ymm1, qword ptr [rdi]{1to4}
# CHECK: encoding: [0x62,0xf2,0xf7,0x38,0x68,0x07]
         vp2intersectq	k0, ymm1, qword ptr [rdi]{1to4}

# CHECK: vp2intersectq	k6, ymm9, ymm7
# CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0xf7]
         vp2intersectq	k6, ymm9, ymm7

# CHECK: vp2intersectq	k6, ymm9, ymmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0x36]
         vp2intersectq	k6, ymm9, ymmword ptr [rsi]

# CHECK: vp2intersectq	k6, ymm9, qword ptr [rsi]{1to4}
# CHECK: encoding: [0x62,0xf2,0xb7,0x38,0x68,0x36]
         vp2intersectq	k6, ymm9, qword ptr [rsi]{1to4}

# CHECK: vp2intersectq	k6, ymm9, ymm7
# CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0xf7]
         vp2intersectq	k6, ymm9, ymm7

# CHECK: vp2intersectq	k6, ymm9, ymmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0xb7,0x28,0x68,0x36]
         vp2intersectq	k6, ymm9, ymmword ptr [rsi]

# CHECK: vp2intersectq	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
         vp2intersectq	k0, xmm1, xmm2

# CHECK: vp2intersectq	k0, xmm1, xmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
         vp2intersectq	k0, xmm1, xmmword ptr [rdi]

# CHECK: vp2intersectq	k0, xmm1, qword ptr [rdi]{1to2}
# CHECK: encoding: [0x62,0xf2,0xf7,0x18,0x68,0x07]
         vp2intersectq	k0, xmm1, qword ptr [rdi]{1to2}

# CHECK: vp2intersectq	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0xc2]
         vp2intersectq	k0, xmm1, xmm2

# CHECK: vp2intersectq	k0, xmm1, xmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0xf7,0x08,0x68,0x07]
         vp2intersectq	k0, xmm1, xmmword ptr [rdi]

# CHECK: vp2intersectq	k6, xmm9, xmm7
# CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0xf7]
         vp2intersectq	k6, xmm9, xmm7

# CHECK: vp2intersectq	k6, xmm9, xmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0x36]
         vp2intersectq	k6, xmm9, xmmword ptr [rsi]

# CHECK: vp2intersectq	k6, xmm9, xmm7
# CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0xf7]
         vp2intersectq	k6, xmm9, xmm7

# CHECK: vp2intersectq	k6, xmm9, xmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0xb7,0x08,0x68,0x36]
         vp2intersectq	k6, xmm9, xmmword ptr [rsi]

# CHECK: vp2intersectd	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
         vp2intersectd	k0, zmm1, zmm2

# CHECK: vp2intersectd	k0, zmm1, zmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
         vp2intersectd	k0, zmm1, zmmword ptr [rdi]

# CHECK: vp2intersectd	k0, zmm1, zmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0xc2]
         vp2intersectd	k0, zmm1, zmm2

# CHECK: vp2intersectd	k0, zmm1, zmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0x77,0x48,0x68,0x07]
         vp2intersectd	k0, zmm1, zmmword ptr [rdi]

# CHECK: vp2intersectd	k6, zmm9, zmm7
# CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0xf7]
         vp2intersectd	k6, zmm9, zmm7

# CHECK: vp2intersectd	k6, zmm9, zmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0x36]
         vp2intersectd	k6, zmm9, zmmword ptr [rsi]

# CHECK: vp2intersectd	k6, zmm9, zmm7
# CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0xf7]
         vp2intersectd	k6, zmm9, zmm7

# CHECK: vp2intersectd	k6, zmm9, zmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0x37,0x48,0x68,0x36]
         vp2intersectd	k6, zmm9, zmmword ptr [rsi]

# CHECK: vp2intersectd	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
         vp2intersectd	k0, ymm1, ymm2

# CHECK: vp2intersectd	k0, ymm1, ymmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
         vp2intersectd	k0, ymm1, ymmword ptr [rdi]

# CHECK: vp2intersectd	k0, ymm1, ymm2
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0xc2]
         vp2intersectd	k0, ymm1, ymm2

# CHECK: vp2intersectd	k0, ymm1, ymmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0x77,0x28,0x68,0x07]
         vp2intersectd	k0, ymm1, ymmword ptr [rdi]

# CHECK: vp2intersectd	k6, ymm9, ymm7
# CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0xf7]
         vp2intersectd	k6, ymm9, ymm7

# CHECK: vp2intersectd	k6, ymm9, ymmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0x36]
         vp2intersectd	k6, ymm9, ymmword ptr [rsi]

# CHECK: vp2intersectd	k6, ymm9, ymm7
# CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0xf7]
         vp2intersectd	k6, ymm9, ymm7

# CHECK: vp2intersectd	k6, ymm9, ymmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0x37,0x28,0x68,0x36]
         vp2intersectd	k6, ymm9, ymmword ptr [rsi]

# CHECK: vp2intersectd	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
         vp2intersectd	k0, xmm1, xmm2

# CHECK: vp2intersectd	k0, xmm1, xmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
         vp2intersectd	k0, xmm1, xmmword ptr [rdi]

# CHECK: vp2intersectd	k0, xmm1, xmm2
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0xc2]
         vp2intersectd	k0, xmm1, xmm2

# CHECK: vp2intersectd	k0, xmm1, xmmword ptr [rdi]
# CHECK: encoding: [0x62,0xf2,0x77,0x08,0x68,0x07]
         vp2intersectd	k0, xmm1, xmmword ptr [rdi]

# CHECK: vp2intersectd	k6, xmm9, xmm7
# CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0xf7]
         vp2intersectd	k6, xmm9, xmm7

# CHECK: vp2intersectd	k6, xmm9, xmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0x36]
         vp2intersectd	k6, xmm9, xmmword ptr [rsi]

# CHECK: vp2intersectd	k6, xmm9, xmm7
# CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0xf7]
         vp2intersectd	k6, xmm9, xmm7

# CHECK: vp2intersectd	k6, xmm9, xmmword ptr [rsi]
# CHECK: encoding: [0x62,0xf2,0x37,0x08,0x68,0x36]
         vp2intersectd	k6, xmm9, xmmword ptr [rsi]
