# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: vcvtps2qq	xmm2 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7b,0x51,0x10]
         vcvtps2qq	xmm2 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvtps2qq	xmm2 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7b,0x51,0x10]
         vcvtps2qq	xmm2 {k2}, qword ptr [rcx + 128]
# CHECK: vcvtps2qq	xmm2, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7b,0x51,0x10]
         vcvtps2qq	xmm2, qword ptr [rcx + 128]
# CHECK: vcvttps2qq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7a,0x49,0x10]
         vcvttps2qq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvttps2qq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7a,0x49,0x10]
         vcvttps2qq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: vcvttps2qq	xmm1, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7a,0x49,0x10]
         vcvttps2qq	xmm1, qword ptr [rcx + 128]
# CHECK: vcvtps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x79,0x49,0x10]
         vcvtps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvtps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x79,0x49,0x10]
         vcvtps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: vcvtps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x79,0x49,0x10]
         vcvtps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: vcvttps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x78,0x49,0x10]
         vcvttps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvttps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
         vcvttps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: vcvttps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x78,0x49,0x10]
         vcvttps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: vcvtps2qq	xmm2 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7b,0x51,0x10]
         vcvtps2qq	xmm2 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvtps2qq	xmm2 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7b,0x51,0x10]
         vcvtps2qq	xmm2 {k2}, qword ptr [rcx + 128]
# CHECK: vcvtps2qq	xmm2, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7b,0x51,0x10]
         vcvtps2qq	xmm2, qword ptr [rcx + 128]
# CHECK: vcvttps2qq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x7a,0x49,0x10]
         vcvttps2qq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvttps2qq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x7a,0x49,0x10]
         vcvttps2qq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: vcvttps2qq	xmm1, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x7a,0x49,0x10]
         vcvttps2qq	xmm1, qword ptr [rcx + 128]
# CHECK: vcvtps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x79,0x49,0x10]
         vcvtps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvtps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x79,0x49,0x10]
         vcvtps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: vcvtps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x79,0x49,0x10]
         vcvtps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: vcvttps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x8a,0x78,0x49,0x10]
         vcvttps2uqq	xmm1 {k2} {z}, qword ptr [rcx + 128]
# CHECK: vcvttps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
         vcvttps2uqq	xmm1 {k2}, qword ptr [rcx + 128]
# CHECK: vcvttps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: encoding: [0x62,0xf1,0x7d,0x08,0x78,0x49,0x10]
         vcvttps2uqq	xmm1, qword ptr [rcx + 128]
# CHECK: vfpclasspd	k2, xmm18, 171
# CHECK: encoding: [0x62,0xb3,0xfd,0x08,0x66,0xd2,0xab]
         vfpclasspd	k2, xmm18, 171
# CHECK: vfpclasspd	k2 {k7}, xmm18, 171
# CHECK: encoding: [0x62,0xb3,0xfd,0x0f,0x66,0xd2,0xab]
         vfpclasspd	k2 {k7}, xmm18, 171
# CHECK: vfpclasspd	k2, xmmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x08,0x66,0x11,0x7b]
         vfpclasspd	k2, xmmword ptr [rcx], 123
# CHECK: vfpclasspd	k2 {k7}, xmmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x0f,0x66,0x11,0x7b]
         vfpclasspd	k2 {k7}, xmmword ptr [rcx], 123
# CHECK: vfpclasspd	k2, qword ptr [rcx]{1to2}, 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x18,0x66,0x11,0x7b]
         vfpclasspd	k2, qword ptr [rcx]{1to2}, 123
# CHECK: vfpclasspd	k2 {k7}, qword ptr [rcx]{1to2}, 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x1f,0x66,0x11,0x7b]
         vfpclasspd	k2 {k7}, qword ptr [rcx]{1to2}, 123
# CHECK: vfpclassps	k2, xmm18, 171
# CHECK: encoding: [0x62,0xb3,0x7d,0x08,0x66,0xd2,0xab]
         vfpclassps	k2, xmm18, 171
# CHECK: vfpclassps	k2 {k7}, xmm18, 171
# CHECK: encoding: [0x62,0xb3,0x7d,0x0f,0x66,0xd2,0xab]
         vfpclassps	k2 {k7}, xmm18, 171
# CHECK: vfpclassps	k2, xmmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x08,0x66,0x11,0x7b]
         vfpclassps	k2, xmmword ptr [rcx], 123
# CHECK: vfpclassps	k2 {k7}, xmmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x0f,0x66,0x11,0x7b]
         vfpclassps	k2 {k7}, xmmword ptr [rcx], 123
# CHECK: vfpclassps	k2, dword ptr [rcx]{1to4}, 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x18,0x66,0x11,0x7b]
         vfpclassps	k2, dword ptr [rcx]{1to4}, 123
# CHECK: vfpclassps	k2 {k7}, dword ptr [rcx]{1to4}, 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x1f,0x66,0x11,0x7b]
         vfpclassps	k2 {k7}, dword ptr [rcx]{1to4}, 123
# CHECK: vfpclasspd	k2, ymm18, 171
# CHECK: encoding: [0x62,0xb3,0xfd,0x28,0x66,0xd2,0xab]
         vfpclasspd	k2, ymm18, 171
# CHECK: vfpclasspd	k2 {k7}, ymm18, 171
# CHECK: encoding: [0x62,0xb3,0xfd,0x2f,0x66,0xd2,0xab]
         vfpclasspd	k2 {k7}, ymm18, 171
# CHECK: vfpclasspd	k2, ymmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x28,0x66,0x11,0x7b]
         vfpclasspd	k2, ymmword ptr [rcx], 123
# CHECK: vfpclasspd	k2 {k7}, ymmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x2f,0x66,0x11,0x7b]
         vfpclasspd	k2 {k7}, ymmword ptr [rcx], 123
# CHECK: vfpclasspd	k2, qword ptr [rcx]{1to4}, 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x38,0x66,0x11,0x7b]
         vfpclasspd	k2, qword ptr [rcx]{1to4}, 123
# CHECK: vfpclasspd	k2 {k7}, qword ptr [rcx]{1to4}, 123
# CHECK: encoding: [0x62,0xf3,0xfd,0x3f,0x66,0x11,0x7b]
         vfpclasspd	k2 {k7}, qword ptr [rcx]{1to4}, 123
# CHECK: vfpclassps	k2, ymm18, 171
# CHECK: encoding: [0x62,0xb3,0x7d,0x28,0x66,0xd2,0xab]
         vfpclassps	k2, ymm18, 171
# CHECK: vfpclassps	k2 {k7}, ymm18, 171
# CHECK: encoding: [0x62,0xb3,0x7d,0x2f,0x66,0xd2,0xab]
         vfpclassps	k2 {k7}, ymm18, 171
# CHECK: vfpclassps	k2, ymmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x28,0x66,0x11,0x7b]
         vfpclassps	k2, ymmword ptr [rcx], 123
# CHECK: vfpclassps	k2 {k7}, ymmword ptr [rcx], 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x2f,0x66,0x11,0x7b]
         vfpclassps	k2 {k7}, ymmword ptr [rcx], 123
# CHECK: vfpclassps	k2, dword ptr [rcx]{1to8}, 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x38,0x66,0x11,0x7b]
         vfpclassps	k2, dword ptr [rcx]{1to8}, 123
# CHECK: vfpclassps	k2 {k7}, dword ptr [rcx]{1to8}, 123
# CHECK: encoding: [0x62,0xf3,0x7d,0x3f,0x66,0x11,0x7b]
         vfpclassps	k2 {k7}, dword ptr [rcx]{1to8}, 123
# CHECK: vcvttps2uqq     xmm1 {k2}, qword ptr [ecx + 128]
# CHECK: encoding: [0x67,0x62,0xf1,0x7d,0x0a,0x78,0x49,0x10]
         vcvttps2uqq     xmm1 {k2}, qword ptr [ecx + 128]
