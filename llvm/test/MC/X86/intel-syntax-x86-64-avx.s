// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: vgatherdpd    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 
// CHECK: encoding: [0xc4,0xe2,0xf9,0x92,0x14,0x4f]
          vgatherdpd    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 

// CHECK: vgatherqpd    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 
// CHECK: encoding: [0xc4,0xe2,0xf9,0x93,0x14,0x4f]
          vgatherqpd    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 

// CHECK: vgatherdpd    ymm2, ymmword ptr [rdi + 2*xmm1], ymm0 
// CHECK: encoding: [0xc4,0xe2,0xfd,0x92,0x14,0x4f]
          vgatherdpd    ymm2, ymmword ptr [rdi + 2*xmm1], ymm0 

// CHECK: vgatherqpd    ymm2, ymmword ptr [rdi + 2*ymm1], ymm0 
// CHECK: encoding: [0xc4,0xe2,0xfd,0x93,0x14,0x4f]
          vgatherqpd    ymm2, ymmword ptr [rdi + 2*ymm1], ymm0 

// CHECK: vgatherdps    xmm10, xmmword ptr [r15 + 2*xmm9], xmm8 
// CHECK: encoding: [0xc4,0x02,0x39,0x92,0x14,0x4f]
          vgatherdps    xmm10, xmmword ptr [r15 + 2*xmm9], xmm8 

// CHECK: vgatherqps    xmm10, qword ptr [r15 + 2*xmm9], xmm8 
// CHECK: encoding: [0xc4,0x02,0x39,0x93,0x14,0x4f]
          vgatherqps    xmm10, qword ptr [r15 + 2*xmm9], xmm8 

// CHECK: vgatherdps    ymm10, ymmword ptr [r15 + 2*ymm9], ymm8 
// CHECK: encoding: [0xc4,0x02,0x3d,0x92,0x14,0x4f]
          vgatherdps    ymm10, ymmword ptr [r15 + 2*ymm9], ymm8 

// CHECK: vgatherqps    xmm10, xmmword ptr [r15 + 2*ymm9], xmm8 
// CHECK: encoding: [0xc4,0x02,0x3d,0x93,0x14,0x4f]
          vgatherqps    xmm10, xmmword ptr [r15 + 2*ymm9], xmm8 

// CHECK: vpgatherdq    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 
// CHECK: encoding: [0xc4,0xe2,0xf9,0x90,0x14,0x4f]
          vpgatherdq    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 

// CHECK: vpgatherqq    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 
// CHECK: encoding: [0xc4,0xe2,0xf9,0x91,0x14,0x4f]
          vpgatherqq    xmm2, xmmword ptr [rdi + 2*xmm1], xmm0 

// CHECK: vpgatherdq    ymm2, ymmword ptr [rdi + 2*xmm1], ymm0 
// CHECK: encoding: [0xc4,0xe2,0xfd,0x90,0x14,0x4f]
          vpgatherdq    ymm2, ymmword ptr [rdi + 2*xmm1], ymm0 

// CHECK: vpgatherqq    ymm2, ymmword ptr [rdi + 2*ymm1], ymm0 
// CHECK: encoding: [0xc4,0xe2,0xfd,0x91,0x14,0x4f]
          vpgatherqq    ymm2, ymmword ptr [rdi + 2*ymm1], ymm0 

// CHECK: vpgatherdd    xmm10, xmmword ptr [r15 + 2*xmm9], xmm8 
// CHECK: encoding: [0xc4,0x02,0x39,0x90,0x14,0x4f]
          vpgatherdd    xmm10, xmmword ptr [r15 + 2*xmm9], xmm8 

// CHECK: vpgatherqd    xmm10, qword ptr [r15 + 2*xmm9], xmm8 
// CHECK: encoding: [0xc4,0x02,0x39,0x91,0x14,0x4f]
          vpgatherqd    xmm10, qword ptr [r15 + 2*xmm9], xmm8 

// CHECK: vpgatherdd    ymm10, ymmword ptr [r15 + 2*ymm9], ymm8 
// CHECK: encoding: [0xc4,0x02,0x3d,0x90,0x14,0x4f]
          vpgatherdd    ymm10, ymmword ptr [r15 + 2*ymm9], ymm8 

// CHECK: vpgatherqd    xmm10, xmmword ptr [r15 + 2*ymm9], xmm8 
// CHECK: encoding: [0xc4,0x02,0x3d,0x91,0x14,0x4f]
          vpgatherqd    xmm10, xmmword ptr [r15 + 2*ymm9], xmm8 

// CHECK: vcvtpd2ps xmm0, xmm15 
// CHECK: encoding: [0xc4,0xc1,0x79,0x5a,0xc7]
          vcvtpd2ps xmm0, xmm15

// CHECK: vcvtpd2ps xmm0, xmm15 
// CHECK: encoding: [0xc4,0xc1,0x79,0x5a,0xc7]
          vcvtpd2psx xmm0, xmm15

// CHECK: vcvtpd2ps xmm0, xmmword ptr [rax]
// CHECK: encoding: [0xc5,0xf9,0x5a,0x00]
          vcvtpd2ps xmm0, xmmword ptr [rax]

// CHECK: vcvtpd2ps xmm0, xmmword ptr [rax]
// CHECK: encoding: [0xc5,0xf9,0x5a,0x00]
          vcvtpd2psx xmm0, xmmword ptr [rax]

// CHECK: vcvtpd2ps xmm0, ymm15 
// CHECK: encoding: [0xc4,0xc1,0x7d,0x5a,0xc7]
          vcvtpd2ps xmm0, ymm15

// CHECK: vcvtpd2ps xmm0, ymm15 
// CHECK: encoding: [0xc4,0xc1,0x7d,0x5a,0xc7]
          vcvtpd2psy xmm0, ymm15

// CHECK: vcvtpd2ps xmm0, ymmword ptr [rax]
// CHECK: encoding: [0xc5,0xfd,0x5a,0x00]
          vcvtpd2ps xmm0, ymmword ptr [rax]

// CHECK: vcvtpd2ps xmm0, ymmword ptr [rax]
// CHECK: encoding: [0xc5,0xfd,0x5a,0x00]
          vcvtpd2psy xmm0, ymmword ptr [rax]

// CHECK: vcvtpd2dq xmm0, xmm15 
// CHECK: encoding: [0xc4,0xc1,0x7b,0xe6,0xc7]
          vcvtpd2dq xmm0, xmm15

// CHECK: vcvtpd2dq xmm0, xmm15 
// CHECK: encoding: [0xc4,0xc1,0x7b,0xe6,0xc7]
          vcvtpd2dqx xmm0, xmm15

// CHECK: vcvtpd2dq xmm0, xmmword ptr [rax]
// CHECK: encoding: [0xc5,0xfb,0xe6,0x00]
          vcvtpd2dq xmm0, xmmword ptr [rax]

// CHECK: vcvtpd2dq xmm0, xmmword ptr [rax]
// CHECK: encoding: [0xc5,0xfb,0xe6,0x00]
          vcvtpd2dqx xmm0, xmmword ptr [rax]

// CHECK: vcvtpd2dq xmm0, ymm15 
// CHECK: encoding: [0xc4,0xc1,0x7f,0xe6,0xc7]
          vcvtpd2dq xmm0, ymm15

// CHECK: vcvtpd2dq xmm0, ymm15 
// CHECK: encoding: [0xc4,0xc1,0x7f,0xe6,0xc7]
          vcvtpd2dqy xmm0, ymm15

// CHECK: vcvtpd2dq xmm0, ymmword ptr [rax]
// CHECK: encoding: [0xc5,0xff,0xe6,0x00]
          vcvtpd2dq xmm0, ymmword ptr [rax]

// CHECK: vcvtpd2dq xmm0, ymmword ptr [rax]
// CHECK: encoding: [0xc5,0xff,0xe6,0x00]
          vcvtpd2dqy xmm0, ymmword ptr [rax]

// CHECK: vcvttpd2dq xmm0, xmm15 
// CHECK: encoding: [0xc4,0xc1,0x79,0xe6,0xc7]
          vcvttpd2dq xmm0, xmm15

// CHECK: vcvttpd2dq xmm0, xmm15 
// CHECK: encoding: [0xc4,0xc1,0x79,0xe6,0xc7]
          vcvttpd2dqx xmm0, xmm15

// CHECK: vcvttpd2dq xmm0, xmmword ptr [rax]
// CHECK: encoding: [0xc5,0xf9,0xe6,0x00]
          vcvttpd2dq xmm0, xmmword ptr [rax]

// CHECK: vcvttpd2dq xmm0, xmmword ptr [rax]
// CHECK: encoding: [0xc5,0xf9,0xe6,0x00]
          vcvttpd2dqx xmm0, xmmword ptr [rax]

// CHECK: vcvttpd2dq xmm0, ymm15 
// CHECK: encoding: [0xc4,0xc1,0x7d,0xe6,0xc7]
          vcvttpd2dq xmm0, ymm15

// CHECK: vcvttpd2dq xmm0, ymm15 
// CHECK: encoding: [0xc4,0xc1,0x7d,0xe6,0xc7]
          vcvttpd2dqy xmm0, ymm15

// CHECK: vcvttpd2dq xmm0, ymmword ptr [rax]
// CHECK: encoding: [0xc5,0xfd,0xe6,0x00]
          vcvttpd2dq xmm0, ymmword ptr [rax]

// CHECK: vcvttpd2dq xmm0, ymmword ptr [rax]
// CHECK: encoding: [0xc5,0xfd,0xe6,0x00]
          vcvttpd2dqy xmm0, ymmword ptr [rax]

// CHECK: vpmaddwd xmm1, xmm2, xmm3
// CHECK: encoding: [0xc5,0xe9,0xf5,0xcb]
          vpmaddwd xmm1, xmm2, xmm3

// CHECK: vpmaddwd xmm1, xmm2, xmmword ptr [rcx]
// CHECK: encoding: [0xc5,0xe9,0xf5,0x09]
          vpmaddwd xmm1, xmm2, xmmword ptr [rcx]

// CHECK: vpmaddwd xmm1, xmm2, xmmword ptr [rsp - 4]
// CHECK: encoding: [0xc5,0xe9,0xf5,0x4c,0x24,0xfc]
          vpmaddwd xmm1, xmm2, xmmword ptr [rsp - 4]

// CHECK: vpmaddwd xmm1, xmm2, xmmword ptr [rsp + 4]
// CHECK: encoding: [0xc5,0xe9,0xf5,0x4c,0x24,0x04]
          vpmaddwd xmm1, xmm2, xmmword ptr [rsp + 4]

// CHECK: vpmaddwd xmm1, xmm2, xmmword ptr [rcx + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0xa1,0x69,0xf5,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vpmaddwd xmm1, xmm2, xmmword ptr [rcx + 8*r14 + 268435456]

// CHECK: vpmaddwd xmm1, xmm2, xmmword ptr [rcx + 8*r14 - 536870912]
// CHECK: encoding: [0xc4,0xa1,0x69,0xf5,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vpmaddwd xmm1, xmm2, xmmword ptr [rcx + 8*r14 - 536870912]

// CHECK: vpmaddwd xmm1, xmm2, xmmword ptr [rcx + 8*r14 - 536870910]
// CHECK: encoding: [0xc4,0xa1,0x69,0xf5,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vpmaddwd xmm1, xmm2, xmmword ptr [rcx + 8*r14 - 536870910]

// CHECK: vpmaddwd ymm1, ymm2, ymm3
// CHECK: encoding: [0xc5,0xed,0xf5,0xcb]
          vpmaddwd ymm1, ymm2, ymm3

// CHECK: vpmaddwd ymm1, ymm2, ymmword ptr [rcx]
// CHECK: encoding: [0xc5,0xed,0xf5,0x09]
          vpmaddwd ymm1, ymm2, ymmword ptr [rcx]

// CHECK: vpmaddwd ymm1, ymm2, ymmword ptr [rsp - 4]
// CHECK: encoding: [0xc5,0xed,0xf5,0x4c,0x24,0xfc]
          vpmaddwd ymm1, ymm2, ymmword ptr [rsp - 4]

// CHECK: vpmaddwd ymm1, ymm2, ymmword ptr [rsp + 4]
// CHECK: encoding: [0xc5,0xed,0xf5,0x4c,0x24,0x04]
          vpmaddwd ymm1, ymm2, ymmword ptr [rsp + 4]

// CHECK: vpmaddwd ymm1, ymm2, ymmword ptr [rcx + 8*r14 + 268435456]
// CHECK: encoding: [0xc4,0xa1,0x6d,0xf5,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vpmaddwd ymm1, ymm2, ymmword ptr [rcx + 8*r14 + 268435456]

// CHECK: vpmaddwd ymm1, ymm2, ymmword ptr [rcx + 8*r14 - 536870912]
// CHECK: encoding: [0xc4,0xa1,0x6d,0xf5,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vpmaddwd ymm1, ymm2, ymmword ptr [rcx + 8*r14 - 536870912]

// CHECK: vpmaddwd ymm1, ymm2, ymmword ptr [rcx + 8*r14 - 536870910]
// CHECK: encoding: [0xc4,0xa1,0x6d,0xf5,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vpmaddwd ymm1, ymm2, ymmword ptr [rcx + 8*r14 - 536870910]
