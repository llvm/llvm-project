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
