// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vaddpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vaddpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x58,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vaddpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x58,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vaddpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x58,0xa2,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vaddpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096, %xmm1, %xmm1 

// CHECK: vaddpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x58,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096, %ymm4, %ymm4 

// CHECK: vaddpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x58,0x4c,0x02,0x40]      
vaddpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vaddpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x58,0x64,0x02,0x40]      
vaddpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vaddpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x58,0x0a]      
vaddpd (%edx), %xmm1, %xmm1 

// CHECK: vaddpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x58,0x22]      
vaddpd (%edx), %ymm4, %ymm4 

// CHECK: vaddpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x58,0xc9]      
vaddpd %xmm1, %xmm1, %xmm1 

// CHECK: vaddpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x58,0xe4]      
vaddpd %ymm4, %ymm4, %ymm4 

// CHECK: vaddps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vaddps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x58,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vaddps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x58,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vaddps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x58,0xa2,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vaddps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096, %xmm1, %xmm1 

// CHECK: vaddps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x58,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096, %ymm4, %ymm4 

// CHECK: vaddps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x58,0x4c,0x02,0x40]      
vaddps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vaddps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x58,0x64,0x02,0x40]      
vaddps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vaddps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x58,0x0a]      
vaddps (%edx), %xmm1, %xmm1 

// CHECK: vaddps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x58,0x22]      
vaddps (%edx), %ymm4, %ymm4 

// CHECK: vaddps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x58,0xc9]      
vaddps %xmm1, %xmm1, %xmm1 

// CHECK: vaddps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x58,0xe4]      
vaddps %ymm4, %ymm4, %ymm4 

// CHECK: vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]      
vaddsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vaddsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]      
vaddsd 485498096, %xmm1, %xmm1 

// CHECK: vaddsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x58,0x4c,0x02,0x40]      
vaddsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vaddsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x58,0x0a]      
vaddsd (%edx), %xmm1, %xmm1 

// CHECK: vaddsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x58,0xc9]      
vaddsd %xmm1, %xmm1, %xmm1 

// CHECK: vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]      
vaddss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vaddss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]      
vaddss 485498096, %xmm1, %xmm1 

// CHECK: vaddss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x58,0x4c,0x02,0x40]      
vaddss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vaddss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x58,0x0a]      
vaddss (%edx), %xmm1, %xmm1 

// CHECK: vaddss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x58,0xc9]      
vaddss %xmm1, %xmm1, %xmm1 

// CHECK: vaddsubpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd0,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vaddsubpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddsubpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd0,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddsubpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd0,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vaddsubpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddsubpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd0,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddsubpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd0,0x8a,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vaddsubpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd0,0xa2,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vaddsubpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd0,0x0d,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096, %xmm1, %xmm1 

// CHECK: vaddsubpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd0,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096, %ymm4, %ymm4 

// CHECK: vaddsubpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd0,0x4c,0x02,0x40]      
vaddsubpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vaddsubpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd0,0x64,0x02,0x40]      
vaddsubpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vaddsubpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd0,0x0a]      
vaddsubpd (%edx), %xmm1, %xmm1 

// CHECK: vaddsubpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd0,0x22]      
vaddsubpd (%edx), %ymm4, %ymm4 

// CHECK: vaddsubpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd0,0xc9]      
vaddsubpd %xmm1, %xmm1, %xmm1 

// CHECK: vaddsubpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd0,0xe4]      
vaddsubpd %ymm4, %ymm4, %ymm4 

// CHECK: vaddsubps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xd0,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vaddsubps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddsubps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xd0,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vaddsubps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0xd0,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vaddsubps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddsubps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0xd0,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vaddsubps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xd0,0x8a,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vaddsubps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0xd0,0xa2,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vaddsubps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xd0,0x0d,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096, %xmm1, %xmm1 

// CHECK: vaddsubps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0xd0,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096, %ymm4, %ymm4 

// CHECK: vaddsubps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xd0,0x4c,0x02,0x40]      
vaddsubps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vaddsubps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0xd0,0x64,0x02,0x40]      
vaddsubps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vaddsubps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xd0,0x0a]      
vaddsubps (%edx), %xmm1, %xmm1 

// CHECK: vaddsubps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0xd0,0x22]      
vaddsubps (%edx), %ymm4, %ymm4 

// CHECK: vaddsubps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xd0,0xc9]      
vaddsubps %xmm1, %xmm1, %xmm1 

// CHECK: vaddsubps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0xd0,0xe4]      
vaddsubps %ymm4, %ymm4, %ymm4 

// CHECK: vandnpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vandnpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandnpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandnpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x55,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vandnpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandnpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x55,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandnpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x55,0x8a,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vandnpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x55,0xa2,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vandnpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x55,0x0d,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096, %xmm1, %xmm1 

// CHECK: vandnpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x55,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096, %ymm4, %ymm4 

// CHECK: vandnpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x55,0x4c,0x02,0x40]      
vandnpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vandnpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x55,0x64,0x02,0x40]      
vandnpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vandnpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x55,0x0a]      
vandnpd (%edx), %xmm1, %xmm1 

// CHECK: vandnpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x55,0x22]      
vandnpd (%edx), %ymm4, %ymm4 

// CHECK: vandnpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x55,0xc9]      
vandnpd %xmm1, %xmm1, %xmm1 

// CHECK: vandnpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x55,0xe4]      
vandnpd %ymm4, %ymm4, %ymm4 

// CHECK: vandnps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vandnps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandnps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandnps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x55,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vandnps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandnps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x55,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandnps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x55,0x8a,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vandnps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x55,0xa2,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vandnps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x55,0x0d,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096, %xmm1, %xmm1 

// CHECK: vandnps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x55,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096, %ymm4, %ymm4 

// CHECK: vandnps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x55,0x4c,0x02,0x40]      
vandnps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vandnps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x55,0x64,0x02,0x40]      
vandnps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vandnps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x55,0x0a]      
vandnps (%edx), %xmm1, %xmm1 

// CHECK: vandnps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x55,0x22]      
vandnps (%edx), %ymm4, %ymm4 

// CHECK: vandnps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x55,0xc9]      
vandnps %xmm1, %xmm1, %xmm1 

// CHECK: vandnps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x55,0xe4]      
vandnps %ymm4, %ymm4, %ymm4 

// CHECK: vandpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x54,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vandpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x54,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x54,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vandpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x54,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x54,0x8a,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vandpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x54,0xa2,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vandpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x54,0x0d,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096, %xmm1, %xmm1 

// CHECK: vandpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x54,0x25,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096, %ymm4, %ymm4 

// CHECK: vandpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x54,0x4c,0x02,0x40]      
vandpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vandpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x54,0x64,0x02,0x40]      
vandpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vandpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x54,0x0a]      
vandpd (%edx), %xmm1, %xmm1 

// CHECK: vandpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x54,0x22]      
vandpd (%edx), %ymm4, %ymm4 

// CHECK: vandpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x54,0xc9]      
vandpd %xmm1, %xmm1, %xmm1 

// CHECK: vandpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x54,0xe4]      
vandpd %ymm4, %ymm4, %ymm4 

// CHECK: vandps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x54,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vandps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x54,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vandps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x54,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vandps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x54,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vandps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x54,0x8a,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vandps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x54,0xa2,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vandps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x54,0x0d,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096, %xmm1, %xmm1 

// CHECK: vandps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x54,0x25,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096, %ymm4, %ymm4 

// CHECK: vandps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x54,0x4c,0x02,0x40]      
vandps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vandps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x54,0x64,0x02,0x40]      
vandps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vandps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x54,0x0a]      
vandps (%edx), %xmm1, %xmm1 

// CHECK: vandps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x54,0x22]      
vandps (%edx), %ymm4, %ymm4 

// CHECK: vandps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x54,0xc9]      
vandps %xmm1, %xmm1, %xmm1 

// CHECK: vandps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x54,0xe4]      
vandps %ymm4, %ymm4, %ymm4 

// CHECK: vblendpd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0d,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vblendpd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendpd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendpd $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0d,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vblendpd $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendpd $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendpd $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0d,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vblendpd $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0d,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vblendpd $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0d,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096, %xmm1, %xmm1 

// CHECK: vblendpd $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0d,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096, %ymm4, %ymm4 

// CHECK: vblendpd $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0d,0x4c,0x02,0x40,0x00]     
vblendpd $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vblendpd $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0d,0x64,0x02,0x40,0x00]     
vblendpd $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vblendpd $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0d,0x0a,0x00]     
vblendpd $0, (%edx), %xmm1, %xmm1 

// CHECK: vblendpd $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0d,0x22,0x00]     
vblendpd $0, (%edx), %ymm4, %ymm4 

// CHECK: vblendpd $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0d,0xc9,0x00]     
vblendpd $0, %xmm1, %xmm1, %xmm1 

// CHECK: vblendpd $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0d,0xe4,0x00]     
vblendpd $0, %ymm4, %ymm4, %ymm4 

// CHECK: vblendps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0c,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vblendps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendps $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0c,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vblendps $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendps $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendps $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0c,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vblendps $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0c,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vblendps $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0c,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096, %xmm1, %xmm1 

// CHECK: vblendps $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096, %ymm4, %ymm4 

// CHECK: vblendps $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0c,0x4c,0x02,0x40,0x00]     
vblendps $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vblendps $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0c,0x64,0x02,0x40,0x00]     
vblendps $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vblendps $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0c,0x0a,0x00]     
vblendps $0, (%edx), %xmm1, %xmm1 

// CHECK: vblendps $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0c,0x22,0x00]     
vblendps $0, (%edx), %ymm4, %ymm4 

// CHECK: vblendps $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0c,0xc9,0x00]     
vblendps $0, %xmm1, %xmm1, %xmm1 

// CHECK: vblendps $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0c,0xe4,0x00]     
vblendps $0, %ymm4, %ymm4, %ymm4 

// CHECK: vblendvpd %xmm1, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x10]     
vblendvpd %xmm1, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendvpd %xmm1, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x10]     
vblendvpd %xmm1, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendvpd %xmm1, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0x8a,0xf0,0x1c,0xf0,0x1c,0x10]     
vblendvpd %xmm1, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vblendvpd %xmm1, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0x0d,0xf0,0x1c,0xf0,0x1c,0x10]     
vblendvpd %xmm1, 485498096, %xmm1, %xmm1 

// CHECK: vblendvpd %xmm1, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0x4c,0x02,0x40,0x10]     
vblendvpd %xmm1, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vblendvpd %xmm1, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0x0a,0x10]     
vblendvpd %xmm1, (%edx), %xmm1, %xmm1 

// CHECK: vblendvpd %xmm1, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0xc9,0x10]     
vblendvpd %xmm1, %xmm1, %xmm1, %xmm1 

// CHECK: vblendvpd %ymm4, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4b,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x40]     
vblendvpd %ymm4, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendvpd %ymm4, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x40]     
vblendvpd %ymm4, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendvpd %ymm4, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4b,0xa2,0xf0,0x1c,0xf0,0x1c,0x40]     
vblendvpd %ymm4, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vblendvpd %ymm4, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4b,0x25,0xf0,0x1c,0xf0,0x1c,0x40]     
vblendvpd %ymm4, 485498096, %ymm4, %ymm4 

// CHECK: vblendvpd %ymm4, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4b,0x64,0x02,0x40,0x40]     
vblendvpd %ymm4, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vblendvpd %ymm4, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4b,0x22,0x40]     
vblendvpd %ymm4, (%edx), %ymm4, %ymm4 

// CHECK: vblendvpd %ymm4, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4b,0xe4,0x40]     
vblendvpd %ymm4, %ymm4, %ymm4, %ymm4 

// CHECK: vblendvps %xmm1, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x10]     
vblendvps %xmm1, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendvps %xmm1, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x10]     
vblendvps %xmm1, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vblendvps %xmm1, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0x8a,0xf0,0x1c,0xf0,0x1c,0x10]     
vblendvps %xmm1, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vblendvps %xmm1, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0x0d,0xf0,0x1c,0xf0,0x1c,0x10]     
vblendvps %xmm1, 485498096, %xmm1, %xmm1 

// CHECK: vblendvps %xmm1, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0x4c,0x02,0x40,0x10]     
vblendvps %xmm1, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vblendvps %xmm1, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0x0a,0x10]     
vblendvps %xmm1, (%edx), %xmm1, %xmm1 

// CHECK: vblendvps %xmm1, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0xc9,0x10]     
vblendvps %xmm1, %xmm1, %xmm1, %xmm1 

// CHECK: vblendvps %ymm4, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4a,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x40]     
vblendvps %ymm4, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendvps %ymm4, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x40]     
vblendvps %ymm4, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vblendvps %ymm4, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4a,0xa2,0xf0,0x1c,0xf0,0x1c,0x40]     
vblendvps %ymm4, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vblendvps %ymm4, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4a,0x25,0xf0,0x1c,0xf0,0x1c,0x40]     
vblendvps %ymm4, 485498096, %ymm4, %ymm4 

// CHECK: vblendvps %ymm4, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4a,0x64,0x02,0x40,0x40]     
vblendvps %ymm4, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vblendvps %ymm4, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4a,0x22,0x40]     
vblendvps %ymm4, (%edx), %ymm4, %ymm4 

// CHECK: vblendvps %ymm4, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4a,0xe4,0x40]     
vblendvps %ymm4, %ymm4, %ymm4, %ymm4 

// CHECK: vbroadcastf128 -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vbroadcastf128 -485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcastf128 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vbroadcastf128 485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcastf128 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0xa2,0xf0,0x1c,0xf0,0x1c]       
vbroadcastf128 485498096(%edx), %ymm4 

// CHECK: vbroadcastf128 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastf128 485498096, %ymm4 

// CHECK: vbroadcastf128 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x64,0x02,0x40]       
vbroadcastf128 64(%edx,%eax), %ymm4 

// CHECK: vbroadcastf128 (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x22]       
vbroadcastf128 (%edx), %ymm4 

// CHECK: vbroadcastsd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vbroadcastsd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcastsd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vbroadcastsd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcastsd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0xa2,0xf0,0x1c,0xf0,0x1c]       
vbroadcastsd 485498096(%edx), %ymm4 

// CHECK: vbroadcastsd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastsd 485498096, %ymm4 

// CHECK: vbroadcastsd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x64,0x02,0x40]       
vbroadcastsd 64(%edx,%eax), %ymm4 

// CHECK: vbroadcastsd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x22]       
vbroadcastsd (%edx), %ymm4 

// CHECK: vbroadcastss -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vbroadcastss -485498096(%edx,%eax,4), %xmm1 

// CHECK: vbroadcastss 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096(%edx,%eax,4), %xmm1 

// CHECK: vbroadcastss -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vbroadcastss -485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcastss 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcastss 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x8a,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096(%edx), %xmm1 

// CHECK: vbroadcastss 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0xa2,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096(%edx), %ymm4 

// CHECK: vbroadcastss 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x0d,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096, %xmm1 

// CHECK: vbroadcastss 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096, %ymm4 

// CHECK: vbroadcastss 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x4c,0x02,0x40]       
vbroadcastss 64(%edx,%eax), %xmm1 

// CHECK: vbroadcastss 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x64,0x02,0x40]       
vbroadcastss 64(%edx,%eax), %ymm4 

// CHECK: vbroadcastss (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x0a]       
vbroadcastss (%edx), %xmm1 

// CHECK: vbroadcastss (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x22]       
vbroadcastss (%edx), %ymm4 

// CHECK: vcmpeqpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc2,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vcmpeqpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc2,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vcmpeqpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vcmpeqpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vcmpeqpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc2,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcmpeqpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc2,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vcmpeqpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc2,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096, %xmm1, %xmm1 

// CHECK: vcmpeqpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc2,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096, %ymm4, %ymm4 

// CHECK: vcmpeqpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc2,0x4c,0x02,0x40,0x00]      
vcmpeqpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcmpeqpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc2,0x64,0x02,0x40,0x00]      
vcmpeqpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vcmpeqpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc2,0x0a,0x00]      
vcmpeqpd (%edx), %xmm1, %xmm1 

// CHECK: vcmpeqpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc2,0x22,0x00]      
vcmpeqpd (%edx), %ymm4, %ymm4 

// CHECK: vcmpeqpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc2,0xc9,0x00]      
vcmpeqpd %xmm1, %xmm1, %xmm1 

// CHECK: vcmpeqpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc2,0xe4,0x00]      
vcmpeqpd %ymm4, %ymm4, %ymm4 

// CHECK: vcmpeqps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc2,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vcmpeqps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc2,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vcmpeqps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vcmpeqps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vcmpeqps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc2,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcmpeqps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc2,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vcmpeqps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc2,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096, %xmm1, %xmm1 

// CHECK: vcmpeqps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc2,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096, %ymm4, %ymm4 

// CHECK: vcmpeqps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc2,0x4c,0x02,0x40,0x00]      
vcmpeqps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcmpeqps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc2,0x64,0x02,0x40,0x00]      
vcmpeqps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vcmpeqps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc2,0x0a,0x00]      
vcmpeqps (%edx), %xmm1, %xmm1 

// CHECK: vcmpeqps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc2,0x22,0x00]      
vcmpeqps (%edx), %ymm4, %ymm4 

// CHECK: vcmpeqps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc2,0xc9,0x00]      
vcmpeqps %xmm1, %xmm1, %xmm1 

// CHECK: vcmpeqps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc2,0xe4,0x00]      
vcmpeqps %ymm4, %ymm4, %ymm4 

// CHECK: vcmpeqsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xc2,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vcmpeqsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xc2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xc2,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcmpeqsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xc2,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqsd 485498096, %xmm1, %xmm1 

// CHECK: vcmpeqsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xc2,0x4c,0x02,0x40,0x00]      
vcmpeqsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcmpeqsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xc2,0x0a,0x00]      
vcmpeqsd (%edx), %xmm1, %xmm1 

// CHECK: vcmpeqsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0xc2,0xc9,0x00]      
vcmpeqsd %xmm1, %xmm1, %xmm1 

// CHECK: vcmpeqss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0xc2,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vcmpeqss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0xc2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcmpeqss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0xc2,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcmpeqss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0xc2,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqss 485498096, %xmm1, %xmm1 

// CHECK: vcmpeqss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0xc2,0x4c,0x02,0x40,0x00]      
vcmpeqss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcmpeqss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0xc2,0x0a,0x00]      
vcmpeqss (%edx), %xmm1, %xmm1 

// CHECK: vcmpeqss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0xc2,0xc9,0x00]      
vcmpeqss %xmm1, %xmm1, %xmm1 

// CHECK: vcomisd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcomisd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcomisd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcomisd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcomisd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcomisd 485498096(%edx), %xmm1 

// CHECK: vcomisd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcomisd 485498096, %xmm1 

// CHECK: vcomisd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x4c,0x02,0x40]       
vcomisd 64(%edx,%eax), %xmm1 

// CHECK: vcomisd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x0a]       
vcomisd (%edx), %xmm1 

// CHECK: vcomisd %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2f,0xc9]       
vcomisd %xmm1, %xmm1 

// CHECK: vcomiss -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcomiss -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcomiss 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcomiss 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcomiss 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcomiss 485498096(%edx), %xmm1 

// CHECK: vcomiss 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcomiss 485498096, %xmm1 

// CHECK: vcomiss 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x4c,0x02,0x40]       
vcomiss 64(%edx,%eax), %xmm1 

// CHECK: vcomiss (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x0a]       
vcomiss (%edx), %xmm1 

// CHECK: vcomiss %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2f,0xc9]       
vcomiss %xmm1, %xmm1 

// CHECK: vcvtdq2pd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtdq2pd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtdq2pd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtdq2pd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0xe6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtdq2pd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtdq2pd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0xe6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtdq2pd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096(%edx), %xmm1 

// CHECK: vcvtdq2pd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0xe6,0xa2,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096(%edx), %ymm4 

// CHECK: vcvtdq2pd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096, %xmm1 

// CHECK: vcvtdq2pd 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096, %ymm4 

// CHECK: vcvtdq2pd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x4c,0x02,0x40]       
vcvtdq2pd 64(%edx,%eax), %xmm1 

// CHECK: vcvtdq2pd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x64,0x02,0x40]       
vcvtdq2pd 64(%edx,%eax), %ymm4 

// CHECK: vcvtdq2pd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x0a]       
vcvtdq2pd (%edx), %xmm1 

// CHECK: vcvtdq2pd (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x22]       
vcvtdq2pd (%edx), %ymm4 

// CHECK: vcvtdq2pd %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0xe6,0xc9]       
vcvtdq2pd %xmm1, %xmm1 

// CHECK: vcvtdq2pd %xmm1, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0xe6,0xe1]       
vcvtdq2pd %xmm1, %ymm4 

// CHECK: vcvtdq2ps -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtdq2ps -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtdq2ps 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtdq2ps -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtdq2ps -485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtdq2ps 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtdq2ps 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096(%edx), %xmm1 

// CHECK: vcvtdq2ps 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5b,0xa2,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096(%edx), %ymm4 

// CHECK: vcvtdq2ps 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096, %xmm1 

// CHECK: vcvtdq2ps 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096, %ymm4 

// CHECK: vcvtdq2ps 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x4c,0x02,0x40]       
vcvtdq2ps 64(%edx,%eax), %xmm1 

// CHECK: vcvtdq2ps 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x64,0x02,0x40]       
vcvtdq2ps 64(%edx,%eax), %ymm4 

// CHECK: vcvtdq2ps (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x0a]       
vcvtdq2ps (%edx), %xmm1 

// CHECK: vcvtdq2ps (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x22]       
vcvtdq2ps (%edx), %ymm4 

// CHECK: vcvtdq2ps %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5b,0xc9]       
vcvtdq2ps %xmm1, %xmm1 

// CHECK: vcvtdq2ps %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5b,0xe4]       
vcvtdq2ps %ymm4, %ymm4 

// CHECK: vcvtpd2dqx -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtpd2dqx -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2dqx 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqx 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2dqx 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqx 485498096(%edx), %xmm1 

// CHECK: vcvtpd2dqx 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqx 485498096, %xmm1 

// CHECK: vcvtpd2dqx 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x4c,0x02,0x40]       
vcvtpd2dqx 64(%edx,%eax), %xmm1 

// CHECK: vcvtpd2dqx (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x0a]       
vcvtpd2dqx (%edx), %xmm1 

// CHECK: vcvtpd2dq %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xe6,0xc9]       
vcvtpd2dq %xmm1, %xmm1 

// CHECK: vcvtpd2dqy -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xff,0xe6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtpd2dqy -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2dqy 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xff,0xe6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqy 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2dqy 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xff,0xe6,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqy 485498096(%edx), %xmm1 

// CHECK: vcvtpd2dqy 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xff,0xe6,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqy 485498096, %xmm1 

// CHECK: vcvtpd2dqy 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xff,0xe6,0x4c,0x02,0x40]       
vcvtpd2dqy 64(%edx,%eax), %xmm1 

// CHECK: vcvtpd2dqy (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xff,0xe6,0x0a]       
vcvtpd2dqy (%edx), %xmm1 

// CHECK: vcvtpd2dq %ymm4, %xmm1 
// CHECK: encoding: [0xc5,0xff,0xe6,0xcc]       
vcvtpd2dq %ymm4, %xmm1 

// CHECK: vcvtpd2psx -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtpd2psx -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2psx 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psx 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2psx 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psx 485498096(%edx), %xmm1 

// CHECK: vcvtpd2psx 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psx 485498096, %xmm1 

// CHECK: vcvtpd2psx 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x4c,0x02,0x40]       
vcvtpd2psx 64(%edx,%eax), %xmm1 

// CHECK: vcvtpd2psx (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x0a]       
vcvtpd2psx (%edx), %xmm1 

// CHECK: vcvtpd2ps %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5a,0xc9]       
vcvtpd2ps %xmm1, %xmm1 

// CHECK: vcvtpd2psy -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtpd2psy -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2psy 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psy 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtpd2psy 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psy 485498096(%edx), %xmm1 

// CHECK: vcvtpd2psy 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psy 485498096, %xmm1 

// CHECK: vcvtpd2psy 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x4c,0x02,0x40]       
vcvtpd2psy 64(%edx,%eax), %xmm1 

// CHECK: vcvtpd2psy (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x0a]       
vcvtpd2psy (%edx), %xmm1 

// CHECK: vcvtpd2ps %ymm4, %xmm1 
// CHECK: encoding: [0xc5,0xfd,0x5a,0xcc]       
vcvtpd2ps %ymm4, %xmm1 

// CHECK: vcvtps2dq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtps2dq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtps2dq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtps2dq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x5b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtps2dq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtps2dq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x5b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtps2dq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096(%edx), %xmm1 

// CHECK: vcvtps2dq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x5b,0xa2,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096(%edx), %ymm4 

// CHECK: vcvtps2dq 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096, %xmm1 

// CHECK: vcvtps2dq 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096, %ymm4 

// CHECK: vcvtps2dq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x4c,0x02,0x40]       
vcvtps2dq 64(%edx,%eax), %xmm1 

// CHECK: vcvtps2dq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x64,0x02,0x40]       
vcvtps2dq 64(%edx,%eax), %ymm4 

// CHECK: vcvtps2dq (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x0a]       
vcvtps2dq (%edx), %xmm1 

// CHECK: vcvtps2dq (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x22]       
vcvtps2dq (%edx), %ymm4 

// CHECK: vcvtps2dq %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x5b,0xc9]       
vcvtps2dq %xmm1, %xmm1 

// CHECK: vcvtps2dq %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x5b,0xe4]       
vcvtps2dq %ymm4, %ymm4 

// CHECK: vcvtps2pd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtps2pd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtps2pd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvtps2pd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vcvtps2pd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtps2pd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvtps2pd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096(%edx), %xmm1 

// CHECK: vcvtps2pd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5a,0xa2,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096(%edx), %ymm4 

// CHECK: vcvtps2pd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096, %xmm1 

// CHECK: vcvtps2pd 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096, %ymm4 

// CHECK: vcvtps2pd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x4c,0x02,0x40]       
vcvtps2pd 64(%edx,%eax), %xmm1 

// CHECK: vcvtps2pd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x64,0x02,0x40]       
vcvtps2pd 64(%edx,%eax), %ymm4 

// CHECK: vcvtps2pd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x0a]       
vcvtps2pd (%edx), %xmm1 

// CHECK: vcvtps2pd (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x22]       
vcvtps2pd (%edx), %ymm4 

// CHECK: vcvtps2pd %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x5a,0xc9]       
vcvtps2pd %xmm1, %xmm1 

// CHECK: vcvtps2pd %xmm1, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x5a,0xe1]       
vcvtps2pd %xmm1, %ymm4 

// CHECK: vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtsd2ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vcvtsd2ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcvtsd2ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vcvtsd2ss 485498096, %xmm1, %xmm1 

// CHECK: vcvtsd2ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5a,0x4c,0x02,0x40]      
vcvtsd2ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcvtsd2ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5a,0x0a]      
vcvtsd2ss (%edx), %xmm1, %xmm1 

// CHECK: vcvtsd2ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5a,0xc9]      
vcvtsd2ss %xmm1, %xmm1, %xmm1 

// CHECK: vcvtsi2sdl -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x2a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vcvtsi2sdl -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtsi2sdl 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x2a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2sdl 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtsi2sdl 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x2a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2sdl 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcvtsi2sdl 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x2a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2sdl 485498096, %xmm1, %xmm1 

// CHECK: vcvtsi2sdl 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x2a,0x4c,0x02,0x40]      
vcvtsi2sdl 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcvtsi2sdl (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x2a,0x0a]      
vcvtsi2sdl (%edx), %xmm1, %xmm1 

// CHECK: vcvtsi2ssl -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x2a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vcvtsi2ssl -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtsi2ssl 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x2a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2ssl 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtsi2ssl 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x2a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2ssl 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcvtsi2ssl 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x2a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2ssl 485498096, %xmm1, %xmm1 

// CHECK: vcvtsi2ssl 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x2a,0x4c,0x02,0x40]      
vcvtsi2ssl 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcvtsi2ssl (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x2a,0x0a]      
vcvtsi2ssl (%edx), %xmm1, %xmm1 

// CHECK: vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vcvtss2sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vcvtss2sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vcvtss2sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vcvtss2sd 485498096, %xmm1, %xmm1 

// CHECK: vcvtss2sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5a,0x4c,0x02,0x40]      
vcvtss2sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vcvtss2sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5a,0x0a]      
vcvtss2sd (%edx), %xmm1, %xmm1 

// CHECK: vcvtss2sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5a,0xc9]      
vcvtss2sd %xmm1, %xmm1, %xmm1 

// CHECK: vcvttpd2dqx -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvttpd2dqx -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvttpd2dqx 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqx 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvttpd2dqx 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqx 485498096(%edx), %xmm1 

// CHECK: vcvttpd2dqx 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqx 485498096, %xmm1 

// CHECK: vcvttpd2dqx 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x4c,0x02,0x40]       
vcvttpd2dqx 64(%edx,%eax), %xmm1 

// CHECK: vcvttpd2dqx (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x0a]       
vcvttpd2dqx (%edx), %xmm1 

// CHECK: vcvttpd2dq %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xe6,0xc9]       
vcvttpd2dq %xmm1, %xmm1 

// CHECK: vcvttpd2dqy -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvttpd2dqy -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvttpd2dqy 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqy 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvttpd2dqy 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqy 485498096(%edx), %xmm1 

// CHECK: vcvttpd2dqy 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqy 485498096, %xmm1 

// CHECK: vcvttpd2dqy 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x4c,0x02,0x40]       
vcvttpd2dqy 64(%edx,%eax), %xmm1 

// CHECK: vcvttpd2dqy (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x0a]       
vcvttpd2dqy (%edx), %xmm1 

// CHECK: vcvttpd2dq %ymm4, %xmm1 
// CHECK: encoding: [0xc5,0xfd,0xe6,0xcc]       
vcvttpd2dq %ymm4, %xmm1 

// CHECK: vcvttps2dq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vcvttps2dq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvttps2dq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vcvttps2dq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x5b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vcvttps2dq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvttps2dq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x5b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vcvttps2dq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x8a,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096(%edx), %xmm1 

// CHECK: vcvttps2dq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x5b,0xa2,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096(%edx), %ymm4 

// CHECK: vcvttps2dq 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x0d,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096, %xmm1 

// CHECK: vcvttps2dq 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096, %ymm4 

// CHECK: vcvttps2dq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x4c,0x02,0x40]       
vcvttps2dq 64(%edx,%eax), %xmm1 

// CHECK: vcvttps2dq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x64,0x02,0x40]       
vcvttps2dq 64(%edx,%eax), %ymm4 

// CHECK: vcvttps2dq (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x0a]       
vcvttps2dq (%edx), %xmm1 

// CHECK: vcvttps2dq (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x22]       
vcvttps2dq (%edx), %ymm4 

// CHECK: vcvttps2dq %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x5b,0xc9]       
vcvttps2dq %xmm1, %xmm1 

// CHECK: vcvttps2dq %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x5b,0xe4]       
vcvttps2dq %ymm4, %ymm4 

// CHECK: vdivpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vdivpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vdivpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vdivpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vdivpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vdivpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vdivpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096, %xmm1, %xmm1 

// CHECK: vdivpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5e,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096, %ymm4, %ymm4 

// CHECK: vdivpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5e,0x4c,0x02,0x40]      
vdivpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vdivpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5e,0x64,0x02,0x40]      
vdivpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vdivpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5e,0x0a]      
vdivpd (%edx), %xmm1, %xmm1 

// CHECK: vdivpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5e,0x22]      
vdivpd (%edx), %ymm4, %ymm4 

// CHECK: vdivpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5e,0xc9]      
vdivpd %xmm1, %xmm1, %xmm1 

// CHECK: vdivpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5e,0xe4]      
vdivpd %ymm4, %ymm4, %ymm4 

// CHECK: vdivps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vdivps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vdivps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vdivps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vdivps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vdivps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vdivps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096, %xmm1, %xmm1 

// CHECK: vdivps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5e,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096, %ymm4, %ymm4 

// CHECK: vdivps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5e,0x4c,0x02,0x40]      
vdivps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vdivps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5e,0x64,0x02,0x40]      
vdivps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vdivps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5e,0x0a]      
vdivps (%edx), %xmm1, %xmm1 

// CHECK: vdivps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5e,0x22]      
vdivps (%edx), %ymm4, %ymm4 

// CHECK: vdivps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5e,0xc9]      
vdivps %xmm1, %xmm1, %xmm1 

// CHECK: vdivps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5e,0xe4]      
vdivps %ymm4, %ymm4, %ymm4 

// CHECK: vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vdivsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vdivsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vdivsd 485498096, %xmm1, %xmm1 

// CHECK: vdivsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5e,0x4c,0x02,0x40]      
vdivsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vdivsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5e,0x0a]      
vdivsd (%edx), %xmm1, %xmm1 

// CHECK: vdivsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5e,0xc9]      
vdivsd %xmm1, %xmm1, %xmm1 

// CHECK: vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdivss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vdivss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vdivss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vdivss 485498096, %xmm1, %xmm1 

// CHECK: vdivss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5e,0x4c,0x02,0x40]      
vdivss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vdivss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5e,0x0a]      
vdivss (%edx), %xmm1, %xmm1 

// CHECK: vdivss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5e,0xc9]      
vdivss %xmm1, %xmm1, %xmm1 

// CHECK: vdppd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x41,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vdppd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdppd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x41,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vdppd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdppd $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x41,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vdppd $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vdppd $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x41,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vdppd $0, 485498096, %xmm1, %xmm1 

// CHECK: vdppd $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x41,0x4c,0x02,0x40,0x00]     
vdppd $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vdppd $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x41,0x0a,0x00]     
vdppd $0, (%edx), %xmm1, %xmm1 

// CHECK: vdppd $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x41,0xc9,0x00]     
vdppd $0, %xmm1, %xmm1, %xmm1 

// CHECK: vdpps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x40,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vdpps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdpps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x40,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vdpps $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x40,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vdpps $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vdpps $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x40,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vdpps $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x40,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vdpps $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x40,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vdpps $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x40,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096, %xmm1, %xmm1 

// CHECK: vdpps $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x40,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096, %ymm4, %ymm4 

// CHECK: vdpps $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x40,0x4c,0x02,0x40,0x00]     
vdpps $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vdpps $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x40,0x64,0x02,0x40,0x00]     
vdpps $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vdpps $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x40,0x0a,0x00]     
vdpps $0, (%edx), %xmm1, %xmm1 

// CHECK: vdpps $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x40,0x22,0x00]     
vdpps $0, (%edx), %ymm4, %ymm4 

// CHECK: vdpps $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x40,0xc9,0x00]     
vdpps $0, %xmm1, %xmm1, %xmm1 

// CHECK: vdpps $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x40,0xe4,0x00]     
vdpps $0, %ymm4, %ymm4, %ymm4 

// CHECK: vextractf128 $0, %ymm4, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractf128 $0, %ymm4, 485498096 

// CHECK: vextractf128 $0, %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractf128 $0, %ymm4, 485498096(%edx) 

// CHECK: vextractf128 $0, %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vextractf128 $0, %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vextractf128 $0, %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractf128 $0, %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vextractf128 $0, %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x64,0x02,0x40,0x00]      
vextractf128 $0, %ymm4, 64(%edx,%eax) 

// CHECK: vextractf128 $0, %ymm4, (%edx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x22,0x00]      
vextractf128 $0, %ymm4, (%edx) 

// CHECK: vextractf128 $0, %ymm4, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0xe1,0x00]      
vextractf128 $0, %ymm4, %xmm1 

// CHECK: vextractps $0, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractps $0, %xmm1, 485498096 

// CHECK: vextractps $0, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractps $0, %xmm1, 485498096(%edx) 

// CHECK: vextractps $0, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vextractps $0, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vextractps $0, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractps $0, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vextractps $0, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x4c,0x02,0x40,0x00]      
vextractps $0, %xmm1, 64(%edx,%eax) 

// CHECK: vextractps $0, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x0a,0x00]      
vextractps $0, %xmm1, (%edx) 

// CHECK: vhaddpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vhaddpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhaddpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhaddpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vhaddpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhaddpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhaddpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vhaddpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vhaddpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096, %xmm1, %xmm1 

// CHECK: vhaddpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096, %ymm4, %ymm4 

// CHECK: vhaddpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7c,0x4c,0x02,0x40]      
vhaddpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vhaddpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7c,0x64,0x02,0x40]      
vhaddpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vhaddpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7c,0x0a]      
vhaddpd (%edx), %xmm1, %xmm1 

// CHECK: vhaddpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7c,0x22]      
vhaddpd (%edx), %ymm4, %ymm4 

// CHECK: vhaddpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7c,0xc9]      
vhaddpd %xmm1, %xmm1, %xmm1 

// CHECK: vhaddpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7c,0xe4]      
vhaddpd %ymm4, %ymm4, %ymm4 

// CHECK: vhaddps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vhaddps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhaddps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhaddps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vhaddps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhaddps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhaddps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vhaddps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vhaddps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096, %xmm1, %xmm1 

// CHECK: vhaddps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096, %ymm4, %ymm4 

// CHECK: vhaddps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7c,0x4c,0x02,0x40]      
vhaddps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vhaddps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7c,0x64,0x02,0x40]      
vhaddps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vhaddps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7c,0x0a]      
vhaddps (%edx), %xmm1, %xmm1 

// CHECK: vhaddps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7c,0x22]      
vhaddps (%edx), %ymm4, %ymm4 

// CHECK: vhaddps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7c,0xc9]      
vhaddps %xmm1, %xmm1, %xmm1 

// CHECK: vhaddps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7c,0xe4]      
vhaddps %ymm4, %ymm4, %ymm4 

// CHECK: vhsubpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vhsubpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhsubpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhsubpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vhsubpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhsubpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhsubpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vhsubpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vhsubpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096, %xmm1, %xmm1 

// CHECK: vhsubpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7d,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096, %ymm4, %ymm4 

// CHECK: vhsubpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7d,0x4c,0x02,0x40]      
vhsubpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vhsubpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7d,0x64,0x02,0x40]      
vhsubpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vhsubpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7d,0x0a]      
vhsubpd (%edx), %xmm1, %xmm1 

// CHECK: vhsubpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7d,0x22]      
vhsubpd (%edx), %ymm4, %ymm4 

// CHECK: vhsubpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x7d,0xc9]      
vhsubpd %xmm1, %xmm1, %xmm1 

// CHECK: vhsubpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x7d,0xe4]      
vhsubpd %ymm4, %ymm4, %ymm4 

// CHECK: vhsubps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vhsubps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhsubps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vhsubps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vhsubps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhsubps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vhsubps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vhsubps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vhsubps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096, %xmm1, %xmm1 

// CHECK: vhsubps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7d,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096, %ymm4, %ymm4 

// CHECK: vhsubps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7d,0x4c,0x02,0x40]      
vhsubps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vhsubps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7d,0x64,0x02,0x40]      
vhsubps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vhsubps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7d,0x0a]      
vhsubps (%edx), %xmm1, %xmm1 

// CHECK: vhsubps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7d,0x22]      
vhsubps (%edx), %ymm4, %ymm4 

// CHECK: vhsubps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x7d,0xc9]      
vhsubps %xmm1, %xmm1, %xmm1 

// CHECK: vhsubps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdf,0x7d,0xe4]      
vhsubps %ymm4, %ymm4, %ymm4 

// CHECK: vinsertf128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x18,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vinsertf128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vinsertf128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x18,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertf128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vinsertf128 $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x18,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertf128 $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vinsertf128 $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x18,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertf128 $0, 485498096, %ymm4, %ymm4 

// CHECK: vinsertf128 $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x18,0x64,0x02,0x40,0x00]     
vinsertf128 $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vinsertf128 $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x18,0x22,0x00]     
vinsertf128 $0, (%edx), %ymm4, %ymm4 

// CHECK: vinsertf128 $0, %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x18,0xe1,0x00]     
vinsertf128 $0, %xmm1, %ymm4, %ymm4 

// CHECK: vinsertps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x21,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vinsertps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vinsertps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x21,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vinsertps $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x21,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertps $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vinsertps $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x21,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertps $0, 485498096, %xmm1, %xmm1 

// CHECK: vinsertps $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x21,0x4c,0x02,0x40,0x00]     
vinsertps $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vinsertps $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x21,0x0a,0x00]     
vinsertps $0, (%edx), %xmm1, %xmm1 

// CHECK: vinsertps $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x21,0xc9,0x00]     
vinsertps $0, %xmm1, %xmm1, %xmm1 

// CHECK: vlddqu -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vlddqu -485498096(%edx,%eax,4), %xmm1 

// CHECK: vlddqu 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096(%edx,%eax,4), %xmm1 

// CHECK: vlddqu -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xff,0xf0,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vlddqu -485498096(%edx,%eax,4), %ymm4 

// CHECK: vlddqu 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xff,0xf0,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096(%edx,%eax,4), %ymm4 

// CHECK: vlddqu 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x8a,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096(%edx), %xmm1 

// CHECK: vlddqu 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xff,0xf0,0xa2,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096(%edx), %ymm4 

// CHECK: vlddqu 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x0d,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096, %xmm1 

// CHECK: vlddqu 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xff,0xf0,0x25,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096, %ymm4 

// CHECK: vlddqu 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x4c,0x02,0x40]       
vlddqu 64(%edx,%eax), %xmm1 

// CHECK: vlddqu 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xff,0xf0,0x64,0x02,0x40]       
vlddqu 64(%edx,%eax), %ymm4 

// CHECK: vlddqu (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x0a]       
vlddqu (%edx), %xmm1 

// CHECK: vlddqu (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xff,0xf0,0x22]       
vlddqu (%edx), %ymm4 

// CHECK: vldmxcsr -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
vldmxcsr -485498096(%edx,%eax,4) 

// CHECK: vldmxcsr 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
vldmxcsr 485498096(%edx,%eax,4) 

// CHECK: vldmxcsr 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x92,0xf0,0x1c,0xf0,0x1c]        
vldmxcsr 485498096(%edx) 

// CHECK: vldmxcsr 485498096 
// CHECK: encoding: [0xc5,0xf8,0xae,0x15,0xf0,0x1c,0xf0,0x1c]        
vldmxcsr 485498096 

// CHECK: vldmxcsr 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x54,0x02,0x40]        
vldmxcsr 64(%edx,%eax) 

// CHECK: vldmxcsr (%edx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x12]        
vldmxcsr (%edx) 

// CHECK: vmaskmovdqu %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0xf7,0xc9]       
vmaskmovdqu %xmm1, %xmm1 

// CHECK: vmaskmovpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaskmovpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaskmovpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaskmovpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaskmovpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmaskmovpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vmaskmovpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096, %xmm1, %xmm1 

// CHECK: vmaskmovpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2d,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096, %ymm4, %ymm4 

// CHECK: vmaskmovpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2d,0x4c,0x02,0x40]      
vmaskmovpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmaskmovpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2d,0x64,0x02,0x40]      
vmaskmovpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vmaskmovpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2d,0x0a]      
vmaskmovpd (%edx), %xmm1, %xmm1 

// CHECK: vmaskmovpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2d,0x22]      
vmaskmovpd (%edx), %ymm4, %ymm4 

// CHECK: vmaskmovpd %xmm1, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %xmm1, %xmm1, 485498096 

// CHECK: vmaskmovpd %xmm1, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %xmm1, %xmm1, 485498096(%edx) 

// CHECK: vmaskmovpd %xmm1, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovpd %xmm1, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmaskmovpd %xmm1, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %xmm1, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmaskmovpd %xmm1, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2f,0x4c,0x02,0x40]      
vmaskmovpd %xmm1, %xmm1, 64(%edx,%eax) 

// CHECK: vmaskmovpd %xmm1, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2f,0x0a]      
vmaskmovpd %xmm1, %xmm1, (%edx) 

// CHECK: vmaskmovpd %ymm4, %ymm4, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2f,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %ymm4, %ymm4, 485498096 

// CHECK: vmaskmovpd %ymm4, %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2f,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %ymm4, %ymm4, 485498096(%edx) 

// CHECK: vmaskmovpd %ymm4, %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovpd %ymm4, %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmaskmovpd %ymm4, %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %ymm4, %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmaskmovpd %ymm4, %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2f,0x64,0x02,0x40]      
vmaskmovpd %ymm4, %ymm4, 64(%edx,%eax) 

// CHECK: vmaskmovpd %ymm4, %ymm4, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2f,0x22]      
vmaskmovpd %ymm4, %ymm4, (%edx) 

// CHECK: vmaskmovps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaskmovps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaskmovps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaskmovps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaskmovps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmaskmovps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vmaskmovps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096, %xmm1, %xmm1 

// CHECK: vmaskmovps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096, %ymm4, %ymm4 

// CHECK: vmaskmovps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2c,0x4c,0x02,0x40]      
vmaskmovps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmaskmovps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2c,0x64,0x02,0x40]      
vmaskmovps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vmaskmovps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2c,0x0a]      
vmaskmovps (%edx), %xmm1, %xmm1 

// CHECK: vmaskmovps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2c,0x22]      
vmaskmovps (%edx), %ymm4, %ymm4 

// CHECK: vmaskmovps %xmm1, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %xmm1, %xmm1, 485498096 

// CHECK: vmaskmovps %xmm1, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %xmm1, %xmm1, 485498096(%edx) 

// CHECK: vmaskmovps %xmm1, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovps %xmm1, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmaskmovps %xmm1, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %xmm1, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmaskmovps %xmm1, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2e,0x4c,0x02,0x40]      
vmaskmovps %xmm1, %xmm1, 64(%edx,%eax) 

// CHECK: vmaskmovps %xmm1, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2e,0x0a]      
vmaskmovps %xmm1, %xmm1, (%edx) 

// CHECK: vmaskmovps %ymm4, %ymm4, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2e,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %ymm4, %ymm4, 485498096 

// CHECK: vmaskmovps %ymm4, %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %ymm4, %ymm4, 485498096(%edx) 

// CHECK: vmaskmovps %ymm4, %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmaskmovps %ymm4, %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmaskmovps %ymm4, %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %ymm4, %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmaskmovps %ymm4, %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2e,0x64,0x02,0x40]      
vmaskmovps %ymm4, %ymm4, 64(%edx,%eax) 

// CHECK: vmaskmovps %ymm4, %ymm4, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2e,0x22]      
vmaskmovps %ymm4, %ymm4, (%edx) 

// CHECK: vmaxpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaxpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmaxpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaxpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaxpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmaxpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5f,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vmaxpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096, %xmm1, %xmm1 

// CHECK: vmaxpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5f,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096, %ymm4, %ymm4 

// CHECK: vmaxpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5f,0x4c,0x02,0x40]      
vmaxpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmaxpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5f,0x64,0x02,0x40]      
vmaxpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vmaxpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5f,0x0a]      
vmaxpd (%edx), %xmm1, %xmm1 

// CHECK: vmaxpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5f,0x22]      
vmaxpd (%edx), %ymm4, %ymm4 

// CHECK: vmaxpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5f,0xc9]      
vmaxpd %xmm1, %xmm1, %xmm1 

// CHECK: vmaxpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5f,0xe4]      
vmaxpd %ymm4, %ymm4, %ymm4 

// CHECK: vmaxps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaxps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmaxps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaxps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmaxps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmaxps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5f,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vmaxps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096, %xmm1, %xmm1 

// CHECK: vmaxps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5f,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096, %ymm4, %ymm4 

// CHECK: vmaxps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5f,0x4c,0x02,0x40]      
vmaxps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmaxps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5f,0x64,0x02,0x40]      
vmaxps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vmaxps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5f,0x0a]      
vmaxps (%edx), %xmm1, %xmm1 

// CHECK: vmaxps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5f,0x22]      
vmaxps (%edx), %ymm4, %ymm4 

// CHECK: vmaxps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5f,0xc9]      
vmaxps %xmm1, %xmm1, %xmm1 

// CHECK: vmaxps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5f,0xe4]      
vmaxps %ymm4, %ymm4, %ymm4 

// CHECK: vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaxsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmaxsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaxsd 485498096, %xmm1, %xmm1 

// CHECK: vmaxsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5f,0x4c,0x02,0x40]      
vmaxsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmaxsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5f,0x0a]      
vmaxsd (%edx), %xmm1, %xmm1 

// CHECK: vmaxsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5f,0xc9]      
vmaxsd %xmm1, %xmm1, %xmm1 

// CHECK: vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmaxss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmaxss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmaxss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmaxss 485498096, %xmm1, %xmm1 

// CHECK: vmaxss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5f,0x4c,0x02,0x40]      
vmaxss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmaxss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5f,0x0a]      
vmaxss (%edx), %xmm1, %xmm1 

// CHECK: vmaxss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5f,0xc9]      
vmaxss %xmm1, %xmm1, %xmm1 

// CHECK: vminpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vminpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vminpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vminpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vminpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vminpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vminpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096, %xmm1, %xmm1 

// CHECK: vminpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5d,0x25,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096, %ymm4, %ymm4 

// CHECK: vminpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5d,0x4c,0x02,0x40]      
vminpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vminpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5d,0x64,0x02,0x40]      
vminpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vminpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5d,0x0a]      
vminpd (%edx), %xmm1, %xmm1 

// CHECK: vminpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5d,0x22]      
vminpd (%edx), %ymm4, %ymm4 

// CHECK: vminpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5d,0xc9]      
vminpd %xmm1, %xmm1, %xmm1 

// CHECK: vminpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5d,0xe4]      
vminpd %ymm4, %ymm4, %ymm4 

// CHECK: vminps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vminps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vminps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vminps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vminps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vminps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vminps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096, %xmm1, %xmm1 

// CHECK: vminps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5d,0x25,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096, %ymm4, %ymm4 

// CHECK: vminps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5d,0x4c,0x02,0x40]      
vminps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vminps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5d,0x64,0x02,0x40]      
vminps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vminps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5d,0x0a]      
vminps (%edx), %xmm1, %xmm1 

// CHECK: vminps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5d,0x22]      
vminps (%edx), %ymm4, %ymm4 

// CHECK: vminps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5d,0xc9]      
vminps %xmm1, %xmm1, %xmm1 

// CHECK: vminps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5d,0xe4]      
vminps %ymm4, %ymm4, %ymm4 

// CHECK: vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vminsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vminsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vminsd 485498096, %xmm1, %xmm1 

// CHECK: vminsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5d,0x4c,0x02,0x40]      
vminsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vminsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5d,0x0a]      
vminsd (%edx), %xmm1, %xmm1 

// CHECK: vminsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5d,0xc9]      
vminsd %xmm1, %xmm1, %xmm1 

// CHECK: vminss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vminss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vminss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vminss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vminss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vminss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vminss 485498096, %xmm1, %xmm1 

// CHECK: vminss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5d,0x4c,0x02,0x40]      
vminss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vminss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5d,0x0a]      
vminss (%edx), %xmm1, %xmm1 

// CHECK: vminss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5d,0xc9]      
vminss %xmm1, %xmm1, %xmm1 

// CHECK: vmovapd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x28,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovapd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovapd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x28,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovapd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x28,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovapd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovapd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x28,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovapd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x28,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096(%edx), %xmm1 

// CHECK: vmovapd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x28,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096(%edx), %ymm4 

// CHECK: vmovapd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x28,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096, %xmm1 

// CHECK: vmovapd 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x28,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096, %ymm4 

// CHECK: vmovapd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x28,0x4c,0x02,0x40]       
vmovapd 64(%edx,%eax), %xmm1 

// CHECK: vmovapd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x28,0x64,0x02,0x40]       
vmovapd 64(%edx,%eax), %ymm4 

// CHECK: vmovapd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x28,0x0a]       
vmovapd (%edx), %xmm1 

// CHECK: vmovapd (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x28,0x22]       
vmovapd (%edx), %ymm4 

// CHECK: vmovapd %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x29,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovapd %xmm1, 485498096 

// CHECK: vmovapd %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovapd %xmm1, 485498096(%edx) 

// CHECK: vmovapd %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovapd %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovapd %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovapd %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovapd %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x4c,0x02,0x40]       
vmovapd %xmm1, 64(%edx,%eax) 

// CHECK: vmovapd %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x0a]       
vmovapd %xmm1, (%edx) 

// CHECK: vmovapd %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x28,0xc9]       
vmovapd %xmm1, %xmm1 

// CHECK: vmovapd %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x29,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd %ymm4, 485498096 

// CHECK: vmovapd %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfd,0x29,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovapd %ymm4, 485498096(%edx) 

// CHECK: vmovapd %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x29,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovapd %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovapd %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x29,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovapd %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovapd %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfd,0x29,0x64,0x02,0x40]       
vmovapd %ymm4, 64(%edx,%eax) 

// CHECK: vmovapd %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfd,0x29,0x22]       
vmovapd %ymm4, (%edx) 

// CHECK: vmovapd %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x28,0xe4]       
vmovapd %ymm4, %ymm4 

// CHECK: vmovaps -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x28,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovaps -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovaps 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x28,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovaps -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x28,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovaps -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovaps 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x28,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovaps 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x28,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096(%edx), %xmm1 

// CHECK: vmovaps 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x28,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096(%edx), %ymm4 

// CHECK: vmovaps 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x28,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096, %xmm1 

// CHECK: vmovaps 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x28,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096, %ymm4 

// CHECK: vmovaps 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x28,0x4c,0x02,0x40]       
vmovaps 64(%edx,%eax), %xmm1 

// CHECK: vmovaps 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x28,0x64,0x02,0x40]       
vmovaps 64(%edx,%eax), %ymm4 

// CHECK: vmovaps (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x28,0x0a]       
vmovaps (%edx), %xmm1 

// CHECK: vmovaps (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x28,0x22]       
vmovaps (%edx), %ymm4 

// CHECK: vmovaps %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x29,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovaps %xmm1, 485498096 

// CHECK: vmovaps %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovaps %xmm1, 485498096(%edx) 

// CHECK: vmovaps %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovaps %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovaps %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovaps %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovaps %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x4c,0x02,0x40]       
vmovaps %xmm1, 64(%edx,%eax) 

// CHECK: vmovaps %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x0a]       
vmovaps %xmm1, (%edx) 

// CHECK: vmovaps %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x28,0xc9]       
vmovaps %xmm1, %xmm1 

// CHECK: vmovaps %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfc,0x29,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps %ymm4, 485498096 

// CHECK: vmovaps %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfc,0x29,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovaps %ymm4, 485498096(%edx) 

// CHECK: vmovaps %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfc,0x29,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovaps %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovaps %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfc,0x29,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovaps %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovaps %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfc,0x29,0x64,0x02,0x40]       
vmovaps %ymm4, 64(%edx,%eax) 

// CHECK: vmovaps %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfc,0x29,0x22]       
vmovaps %ymm4, (%edx) 

// CHECK: vmovaps %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x28,0xe4]       
vmovaps %ymm4, %ymm4 

// CHECK: vmovd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovd 485498096(%edx), %xmm1 

// CHECK: vmovd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovd 485498096, %xmm1 

// CHECK: vmovd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x4c,0x02,0x40]       
vmovd 64(%edx,%eax), %xmm1 

// CHECK: vmovddup -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x12,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovddup -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovddup 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x12,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovddup -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x12,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovddup -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovddup 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x12,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovddup 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x12,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096(%edx), %xmm1 

// CHECK: vmovddup 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x12,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096(%edx), %ymm4 

// CHECK: vmovddup 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x12,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096, %xmm1 

// CHECK: vmovddup 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xff,0x12,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096, %ymm4 

// CHECK: vmovddup 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x12,0x4c,0x02,0x40]       
vmovddup 64(%edx,%eax), %xmm1 

// CHECK: vmovddup 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x12,0x64,0x02,0x40]       
vmovddup 64(%edx,%eax), %ymm4 

// CHECK: vmovddup (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x12,0x0a]       
vmovddup (%edx), %xmm1 

// CHECK: vmovddup (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x12,0x22]       
vmovddup (%edx), %ymm4 

// CHECK: vmovddup %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x12,0xc9]       
vmovddup %xmm1, %xmm1 

// CHECK: vmovddup %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xff,0x12,0xe4]       
vmovddup %ymm4, %ymm4 

// CHECK: vmovd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x0a]       
vmovd (%edx), %xmm1 

// CHECK: vmovdqa -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqa -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovdqa 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovdqa -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x6f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqa -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovdqa 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x6f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovdqa 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096(%edx), %xmm1 

// CHECK: vmovdqa 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x6f,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096(%edx), %ymm4 

// CHECK: vmovdqa 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096, %xmm1 

// CHECK: vmovdqa 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096, %ymm4 

// CHECK: vmovdqa 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x4c,0x02,0x40]       
vmovdqa 64(%edx,%eax), %xmm1 

// CHECK: vmovdqa 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x64,0x02,0x40]       
vmovdqa 64(%edx,%eax), %ymm4 

// CHECK: vmovdqa (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x0a]       
vmovdqa (%edx), %xmm1 

// CHECK: vmovdqa (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x22]       
vmovdqa (%edx), %ymm4 

// CHECK: vmovdqa %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %xmm1, 485498096 

// CHECK: vmovdqa %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %xmm1, 485498096(%edx) 

// CHECK: vmovdqa %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqa %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovdqa %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovdqa %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x4c,0x02,0x40]       
vmovdqa %xmm1, 64(%edx,%eax) 

// CHECK: vmovdqa %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x0a]       
vmovdqa %xmm1, (%edx) 

// CHECK: vmovdqa %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x6f,0xc9]       
vmovdqa %xmm1, %xmm1 

// CHECK: vmovdqa %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %ymm4, 485498096 

// CHECK: vmovdqa %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %ymm4, 485498096(%edx) 

// CHECK: vmovdqa %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqa %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovdqa %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovdqa %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x64,0x02,0x40]       
vmovdqa %ymm4, 64(%edx,%eax) 

// CHECK: vmovdqa %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x22]       
vmovdqa %ymm4, (%edx) 

// CHECK: vmovdqa %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x6f,0xe4]       
vmovdqa %ymm4, %ymm4 

// CHECK: vmovdqu -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqu -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovdqu 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovdqu -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x6f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqu -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovdqu 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x6f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovdqu 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096(%edx), %xmm1 

// CHECK: vmovdqu 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x6f,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096(%edx), %ymm4 

// CHECK: vmovdqu 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096, %xmm1 

// CHECK: vmovdqu 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096, %ymm4 

// CHECK: vmovdqu 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x4c,0x02,0x40]       
vmovdqu 64(%edx,%eax), %xmm1 

// CHECK: vmovdqu 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x64,0x02,0x40]       
vmovdqu 64(%edx,%eax), %ymm4 

// CHECK: vmovdqu (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x0a]       
vmovdqu (%edx), %xmm1 

// CHECK: vmovdqu (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x22]       
vmovdqu (%edx), %ymm4 

// CHECK: vmovdqu %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %xmm1, 485498096 

// CHECK: vmovdqu %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %xmm1, 485498096(%edx) 

// CHECK: vmovdqu %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqu %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovdqu %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovdqu %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x4c,0x02,0x40]       
vmovdqu %xmm1, 64(%edx,%eax) 

// CHECK: vmovdqu %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x0a]       
vmovdqu %xmm1, (%edx) 

// CHECK: vmovdqu %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x6f,0xc9]       
vmovdqu %xmm1, %xmm1 

// CHECK: vmovdqu %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %ymm4, 485498096 

// CHECK: vmovdqu %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %ymm4, 485498096(%edx) 

// CHECK: vmovdqu %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovdqu %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovdqu %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovdqu %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x64,0x02,0x40]       
vmovdqu %ymm4, 64(%edx,%eax) 

// CHECK: vmovdqu %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x22]       
vmovdqu %ymm4, (%edx) 

// CHECK: vmovdqu %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x6f,0xe4]       
vmovdqu %ymm4, %ymm4 

// CHECK: vmovd %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovd %xmm1, 485498096 

// CHECK: vmovd %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovd %xmm1, 485498096(%edx) 

// CHECK: vmovd %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovd %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovd %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovd %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovd %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x4c,0x02,0x40]       
vmovd %xmm1, 64(%edx,%eax) 

// CHECK: vmovd %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x0a]       
vmovd %xmm1, (%edx) 

// CHECK: vmovhlps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x12,0xc9]      
vmovhlps %xmm1, %xmm1, %xmm1 

// CHECK: vmovhpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x16,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmovhpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovhpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x16,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmovhpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovhpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x16,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmovhpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmovhpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x16,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmovhpd 485498096, %xmm1, %xmm1 

// CHECK: vmovhpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x16,0x4c,0x02,0x40]      
vmovhpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmovhpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x16,0x0a]      
vmovhpd (%edx), %xmm1, %xmm1 

// CHECK: vmovhpd %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x17,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovhpd %xmm1, 485498096 

// CHECK: vmovhpd %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovhpd %xmm1, 485498096(%edx) 

// CHECK: vmovhpd %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovhpd %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovhpd %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovhpd %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovhpd %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x4c,0x02,0x40]       
vmovhpd %xmm1, 64(%edx,%eax) 

// CHECK: vmovhpd %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x0a]       
vmovhpd %xmm1, (%edx) 

// CHECK: vmovhps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x16,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmovhps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovhps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x16,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmovhps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovhps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x16,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmovhps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmovhps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x16,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmovhps 485498096, %xmm1, %xmm1 

// CHECK: vmovhps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x16,0x4c,0x02,0x40]      
vmovhps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmovhps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x16,0x0a]      
vmovhps (%edx), %xmm1, %xmm1 

// CHECK: vmovhps %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x17,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovhps %xmm1, 485498096 

// CHECK: vmovhps %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovhps %xmm1, 485498096(%edx) 

// CHECK: vmovhps %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovhps %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovhps %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovhps %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovhps %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x4c,0x02,0x40]       
vmovhps %xmm1, 64(%edx,%eax) 

// CHECK: vmovhps %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x0a]       
vmovhps %xmm1, (%edx) 

// CHECK: vmovlhps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x16,0xc9]      
vmovlhps %xmm1, %xmm1, %xmm1 

// CHECK: vmovlpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x12,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmovlpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovlpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x12,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmovlpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovlpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x12,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmovlpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmovlpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x12,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmovlpd 485498096, %xmm1, %xmm1 

// CHECK: vmovlpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x12,0x4c,0x02,0x40]      
vmovlpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmovlpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x12,0x0a]      
vmovlpd (%edx), %xmm1, %xmm1 

// CHECK: vmovlpd %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x13,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovlpd %xmm1, 485498096 

// CHECK: vmovlpd %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovlpd %xmm1, 485498096(%edx) 

// CHECK: vmovlpd %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovlpd %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovlpd %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovlpd %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovlpd %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x4c,0x02,0x40]       
vmovlpd %xmm1, 64(%edx,%eax) 

// CHECK: vmovlpd %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x0a]       
vmovlpd %xmm1, (%edx) 

// CHECK: vmovlps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x12,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmovlps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovlps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x12,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmovlps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmovlps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x12,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmovlps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmovlps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x12,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmovlps 485498096, %xmm1, %xmm1 

// CHECK: vmovlps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x12,0x4c,0x02,0x40]      
vmovlps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmovlps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x12,0x0a]      
vmovlps (%edx), %xmm1, %xmm1 

// CHECK: vmovlps %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x13,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovlps %xmm1, 485498096 

// CHECK: vmovlps %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovlps %xmm1, 485498096(%edx) 

// CHECK: vmovlps %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovlps %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovlps %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovlps %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovlps %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x4c,0x02,0x40]       
vmovlps %xmm1, 64(%edx,%eax) 

// CHECK: vmovlps %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x0a]       
vmovlps %xmm1, (%edx) 

// CHECK: vmovntdqa -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntdqa -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovntdqa 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovntdqa 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096(%edx), %xmm1 

// CHECK: vmovntdqa 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096, %xmm1 

// CHECK: vmovntdqa 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x4c,0x02,0x40]       
vmovntdqa 64(%edx,%eax), %xmm1 

// CHECK: vmovntdqa (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x0a]       
vmovntdqa (%edx), %xmm1 

// CHECK: vmovntdq %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %xmm1, 485498096 

// CHECK: vmovntdq %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %xmm1, 485498096(%edx) 

// CHECK: vmovntdq %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntdq %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovntdq %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovntdq %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x4c,0x02,0x40]       
vmovntdq %xmm1, 64(%edx,%eax) 

// CHECK: vmovntdq %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x0a]       
vmovntdq %xmm1, (%edx) 

// CHECK: vmovntdq %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %ymm4, 485498096 

// CHECK: vmovntdq %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %ymm4, 485498096(%edx) 

// CHECK: vmovntdq %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntdq %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovntdq %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovntdq %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x64,0x02,0x40]       
vmovntdq %ymm4, 64(%edx,%eax) 

// CHECK: vmovntdq %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x22]       
vmovntdq %ymm4, (%edx) 

// CHECK: vmovntpd %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %xmm1, 485498096 

// CHECK: vmovntpd %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %xmm1, 485498096(%edx) 

// CHECK: vmovntpd %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntpd %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovntpd %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovntpd %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x4c,0x02,0x40]       
vmovntpd %xmm1, 64(%edx,%eax) 

// CHECK: vmovntpd %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x0a]       
vmovntpd %xmm1, (%edx) 

// CHECK: vmovntpd %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %ymm4, 485498096 

// CHECK: vmovntpd %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %ymm4, 485498096(%edx) 

// CHECK: vmovntpd %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntpd %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovntpd %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovntpd %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x64,0x02,0x40]       
vmovntpd %ymm4, 64(%edx,%eax) 

// CHECK: vmovntpd %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x22]       
vmovntpd %ymm4, (%edx) 

// CHECK: vmovntps %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovntps %xmm1, 485498096 

// CHECK: vmovntps %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovntps %xmm1, 485498096(%edx) 

// CHECK: vmovntps %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntps %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovntps %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntps %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovntps %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x4c,0x02,0x40]       
vmovntps %xmm1, 64(%edx,%eax) 

// CHECK: vmovntps %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x0a]       
vmovntps %xmm1, (%edx) 

// CHECK: vmovntps %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntps %ymm4, 485498096 

// CHECK: vmovntps %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovntps %ymm4, 485498096(%edx) 

// CHECK: vmovntps %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntps %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovntps %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntps %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovntps %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x64,0x02,0x40]       
vmovntps %ymm4, 64(%edx,%eax) 

// CHECK: vmovntps %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x22]       
vmovntps %ymm4, (%edx) 

// CHECK: vmovq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovq 485498096(%edx), %xmm1 

// CHECK: vmovq 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovq 485498096, %xmm1 

// CHECK: vmovq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x4c,0x02,0x40]       
vmovq 64(%edx,%eax), %xmm1 

// CHECK: vmovq (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x0a]       
vmovq (%edx), %xmm1 

// CHECK: vmovq %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovq %xmm1, 485498096 

// CHECK: vmovq %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovq %xmm1, 485498096(%edx) 

// CHECK: vmovq %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovq %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovq %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovq %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovq %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x4c,0x02,0x40]       
vmovq %xmm1, 64(%edx,%eax) 

// CHECK: vmovq %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x0a]       
vmovq %xmm1, (%edx) 

// CHECK: vmovq %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x7e,0xc9]       
vmovq %xmm1, %xmm1 

// CHECK: vmovsd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovsd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovsd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovsd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovsd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovsd 485498096(%edx), %xmm1 

// CHECK: vmovsd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovsd 485498096, %xmm1 

// CHECK: vmovsd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x10,0x4c,0x02,0x40]       
vmovsd 64(%edx,%eax), %xmm1 

// CHECK: vmovsd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x10,0x0a]       
vmovsd (%edx), %xmm1 

// CHECK: vmovsd %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xfb,0x11,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovsd %xmm1, 485498096 

// CHECK: vmovsd %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovsd %xmm1, 485498096(%edx) 

// CHECK: vmovsd %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovsd %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovsd %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovsd %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovsd %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x4c,0x02,0x40]       
vmovsd %xmm1, 64(%edx,%eax) 

// CHECK: vmovsd %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x0a]       
vmovsd %xmm1, (%edx) 

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x10,0xc9]      
vmovsd %xmm1, %xmm1, %xmm1 

// CHECK: vmovshdup -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x16,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovshdup -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovshdup 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x16,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovshdup -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x16,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovshdup -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovshdup 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x16,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovshdup 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x16,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096(%edx), %xmm1 

// CHECK: vmovshdup 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x16,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096(%edx), %ymm4 

// CHECK: vmovshdup 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x16,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096, %xmm1 

// CHECK: vmovshdup 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x16,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096, %ymm4 

// CHECK: vmovshdup 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x16,0x4c,0x02,0x40]       
vmovshdup 64(%edx,%eax), %xmm1 

// CHECK: vmovshdup 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x16,0x64,0x02,0x40]       
vmovshdup 64(%edx,%eax), %ymm4 

// CHECK: vmovshdup (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x16,0x0a]       
vmovshdup (%edx), %xmm1 

// CHECK: vmovshdup (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x16,0x22]       
vmovshdup (%edx), %ymm4 

// CHECK: vmovshdup %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x16,0xc9]       
vmovshdup %xmm1, %xmm1 

// CHECK: vmovshdup %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x16,0xe4]       
vmovshdup %ymm4, %ymm4 

// CHECK: vmovsldup -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x12,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovsldup -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovsldup 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x12,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovsldup -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x12,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovsldup -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovsldup 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x12,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovsldup 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x12,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096(%edx), %xmm1 

// CHECK: vmovsldup 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x12,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096(%edx), %ymm4 

// CHECK: vmovsldup 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x12,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096, %xmm1 

// CHECK: vmovsldup 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x12,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096, %ymm4 

// CHECK: vmovsldup 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x12,0x4c,0x02,0x40]       
vmovsldup 64(%edx,%eax), %xmm1 

// CHECK: vmovsldup 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x12,0x64,0x02,0x40]       
vmovsldup 64(%edx,%eax), %ymm4 

// CHECK: vmovsldup (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x12,0x0a]       
vmovsldup (%edx), %xmm1 

// CHECK: vmovsldup (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x12,0x22]       
vmovsldup (%edx), %ymm4 

// CHECK: vmovsldup %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x12,0xc9]       
vmovsldup %xmm1, %xmm1 

// CHECK: vmovsldup %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x12,0xe4]       
vmovsldup %ymm4, %ymm4 

// CHECK: vmovss -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovss -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovss 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovss 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovss 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovss 485498096(%edx), %xmm1 

// CHECK: vmovss 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovss 485498096, %xmm1 

// CHECK: vmovss 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x10,0x4c,0x02,0x40]       
vmovss 64(%edx,%eax), %xmm1 

// CHECK: vmovss (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x10,0x0a]       
vmovss (%edx), %xmm1 

// CHECK: vmovss %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xfa,0x11,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovss %xmm1, 485498096 

// CHECK: vmovss %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovss %xmm1, 485498096(%edx) 

// CHECK: vmovss %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovss %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovss %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovss %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovss %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x4c,0x02,0x40]       
vmovss %xmm1, 64(%edx,%eax) 

// CHECK: vmovss %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x0a]       
vmovss %xmm1, (%edx) 

// CHECK: vmovss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x10,0xc9]      
vmovss %xmm1, %xmm1, %xmm1 

// CHECK: vmovupd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovupd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovupd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovupd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x10,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovupd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovupd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x10,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovupd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096(%edx), %xmm1 

// CHECK: vmovupd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x10,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096(%edx), %ymm4 

// CHECK: vmovupd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096, %xmm1 

// CHECK: vmovupd 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x10,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096, %ymm4 

// CHECK: vmovupd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x10,0x4c,0x02,0x40]       
vmovupd 64(%edx,%eax), %xmm1 

// CHECK: vmovupd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x10,0x64,0x02,0x40]       
vmovupd 64(%edx,%eax), %ymm4 

// CHECK: vmovupd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x10,0x0a]       
vmovupd (%edx), %xmm1 

// CHECK: vmovupd (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x10,0x22]       
vmovupd (%edx), %ymm4 

// CHECK: vmovupd %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x11,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovupd %xmm1, 485498096 

// CHECK: vmovupd %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovupd %xmm1, 485498096(%edx) 

// CHECK: vmovupd %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovupd %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovupd %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovupd %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovupd %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x4c,0x02,0x40]       
vmovupd %xmm1, 64(%edx,%eax) 

// CHECK: vmovupd %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x0a]       
vmovupd %xmm1, (%edx) 

// CHECK: vmovupd %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x10,0xc9]       
vmovupd %xmm1, %xmm1 

// CHECK: vmovupd %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x11,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd %ymm4, 485498096 

// CHECK: vmovupd %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfd,0x11,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovupd %ymm4, 485498096(%edx) 

// CHECK: vmovupd %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x11,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovupd %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovupd %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfd,0x11,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovupd %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovupd %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfd,0x11,0x64,0x02,0x40]       
vmovupd %ymm4, 64(%edx,%eax) 

// CHECK: vmovupd %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfd,0x11,0x22]       
vmovupd %ymm4, (%edx) 

// CHECK: vmovupd %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x10,0xe4]       
vmovupd %ymm4, %ymm4 

// CHECK: vmovups -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovups -485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovups 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096(%edx,%eax,4), %xmm1 

// CHECK: vmovups -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x10,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovups -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovups 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x10,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovups 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096(%edx), %xmm1 

// CHECK: vmovups 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x10,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096(%edx), %ymm4 

// CHECK: vmovups 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096, %xmm1 

// CHECK: vmovups 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x10,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096, %ymm4 

// CHECK: vmovups 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x10,0x4c,0x02,0x40]       
vmovups 64(%edx,%eax), %xmm1 

// CHECK: vmovups 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x10,0x64,0x02,0x40]       
vmovups 64(%edx,%eax), %ymm4 

// CHECK: vmovups (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x10,0x0a]       
vmovups (%edx), %xmm1 

// CHECK: vmovups (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x10,0x22]       
vmovups (%edx), %ymm4 

// CHECK: vmovups %xmm1, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x11,0x0d,0xf0,0x1c,0xf0,0x1c]       
vmovups %xmm1, 485498096 

// CHECK: vmovups %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]       
vmovups %xmm1, 485498096(%edx) 

// CHECK: vmovups %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vmovups %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vmovups %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovups %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vmovups %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x4c,0x02,0x40]       
vmovups %xmm1, 64(%edx,%eax) 

// CHECK: vmovups %xmm1, (%edx) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x0a]       
vmovups %xmm1, (%edx) 

// CHECK: vmovups %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x10,0xc9]       
vmovups %xmm1, %xmm1 

// CHECK: vmovups %ymm4, 485498096 
// CHECK: encoding: [0xc5,0xfc,0x11,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups %ymm4, 485498096 

// CHECK: vmovups %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc5,0xfc,0x11,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovups %ymm4, 485498096(%edx) 

// CHECK: vmovups %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfc,0x11,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovups %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vmovups %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xfc,0x11,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovups %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vmovups %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xfc,0x11,0x64,0x02,0x40]       
vmovups %ymm4, 64(%edx,%eax) 

// CHECK: vmovups %ymm4, (%edx) 
// CHECK: encoding: [0xc5,0xfc,0x11,0x22]       
vmovups %ymm4, (%edx) 

// CHECK: vmovups %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x10,0xe4]       
vmovups %ymm4, %ymm4 

// CHECK: vmpsadbw $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x42,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vmpsadbw $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmpsadbw $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x42,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmpsadbw $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x42,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmpsadbw $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x42,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096, %xmm1, %xmm1 

// CHECK: vmpsadbw $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x42,0x4c,0x02,0x40,0x00]     
vmpsadbw $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmpsadbw $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x42,0x0a,0x00]     
vmpsadbw $0, (%edx), %xmm1, %xmm1 

// CHECK: vmpsadbw $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x42,0xc9,0x00]     
vmpsadbw $0, %xmm1, %xmm1, %xmm1 

// CHECK: vmulpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmulpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x59,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmulpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmulpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x59,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmulpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmulpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x59,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vmulpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096, %xmm1, %xmm1 

// CHECK: vmulpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x59,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096, %ymm4, %ymm4 

// CHECK: vmulpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x59,0x4c,0x02,0x40]      
vmulpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmulpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x59,0x64,0x02,0x40]      
vmulpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vmulpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x59,0x0a]      
vmulpd (%edx), %xmm1, %xmm1 

// CHECK: vmulpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x59,0x22]      
vmulpd (%edx), %ymm4, %ymm4 

// CHECK: vmulpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x59,0xc9]      
vmulpd %xmm1, %xmm1, %xmm1 

// CHECK: vmulpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x59,0xe4]      
vmulpd %ymm4, %ymm4, %ymm4 

// CHECK: vmulps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmulps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x59,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vmulps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmulps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x59,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmulps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmulps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x59,0xa2,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vmulps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096, %xmm1, %xmm1 

// CHECK: vmulps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x59,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096, %ymm4, %ymm4 

// CHECK: vmulps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x59,0x4c,0x02,0x40]      
vmulps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmulps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x59,0x64,0x02,0x40]      
vmulps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vmulps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x59,0x0a]      
vmulps (%edx), %xmm1, %xmm1 

// CHECK: vmulps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x59,0x22]      
vmulps (%edx), %ymm4, %ymm4 

// CHECK: vmulps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x59,0xc9]      
vmulps %xmm1, %xmm1, %xmm1 

// CHECK: vmulps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x59,0xe4]      
vmulps %ymm4, %ymm4, %ymm4 

// CHECK: vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmulsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmulsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmulsd 485498096, %xmm1, %xmm1 

// CHECK: vmulsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x59,0x4c,0x02,0x40]      
vmulsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmulsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x59,0x0a]      
vmulsd (%edx), %xmm1, %xmm1 

// CHECK: vmulsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x59,0xc9]      
vmulsd %xmm1, %xmm1, %xmm1 

// CHECK: vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vmulss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]      
vmulss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vmulss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]      
vmulss 485498096, %xmm1, %xmm1 

// CHECK: vmulss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x59,0x4c,0x02,0x40]      
vmulss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vmulss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x59,0x0a]      
vmulss (%edx), %xmm1, %xmm1 

// CHECK: vmulss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x59,0xc9]      
vmulss %xmm1, %xmm1, %xmm1 

// CHECK: vorpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x56,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vorpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vorpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x56,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vorpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x56,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vorpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vorpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x56,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vorpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x56,0x8a,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vorpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x56,0xa2,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vorpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x56,0x0d,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096, %xmm1, %xmm1 

// CHECK: vorpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x56,0x25,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096, %ymm4, %ymm4 

// CHECK: vorpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x56,0x4c,0x02,0x40]      
vorpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vorpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x56,0x64,0x02,0x40]      
vorpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vorpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x56,0x0a]      
vorpd (%edx), %xmm1, %xmm1 

// CHECK: vorpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x56,0x22]      
vorpd (%edx), %ymm4, %ymm4 

// CHECK: vorpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x56,0xc9]      
vorpd %xmm1, %xmm1, %xmm1 

// CHECK: vorpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x56,0xe4]      
vorpd %ymm4, %ymm4, %ymm4 

// CHECK: vorps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x56,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vorps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vorps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x56,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vorps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x56,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vorps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vorps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x56,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vorps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x56,0x8a,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vorps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x56,0xa2,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vorps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x56,0x0d,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096, %xmm1, %xmm1 

// CHECK: vorps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x56,0x25,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096, %ymm4, %ymm4 

// CHECK: vorps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x56,0x4c,0x02,0x40]      
vorps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vorps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x56,0x64,0x02,0x40]      
vorps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vorps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x56,0x0a]      
vorps (%edx), %xmm1, %xmm1 

// CHECK: vorps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x56,0x22]      
vorps (%edx), %ymm4, %ymm4 

// CHECK: vorps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x56,0xc9]      
vorps %xmm1, %xmm1, %xmm1 

// CHECK: vorps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x56,0xe4]      
vorps %ymm4, %ymm4, %ymm4 

// CHECK: vpabsb -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpabsb -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpabsb 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpabsb 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096(%edx), %xmm1 

// CHECK: vpabsb 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096, %xmm1 

// CHECK: vpabsb 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x4c,0x02,0x40]       
vpabsb 64(%edx,%eax), %xmm1 

// CHECK: vpabsb (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x0a]       
vpabsb (%edx), %xmm1 

// CHECK: vpabsb %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0xc9]       
vpabsb %xmm1, %xmm1 

// CHECK: vpabsd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpabsd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpabsd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpabsd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096(%edx), %xmm1 

// CHECK: vpabsd 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096, %xmm1 

// CHECK: vpabsd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x4c,0x02,0x40]       
vpabsd 64(%edx,%eax), %xmm1 

// CHECK: vpabsd (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x0a]       
vpabsd (%edx), %xmm1 

// CHECK: vpabsd %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0xc9]       
vpabsd %xmm1, %xmm1 

// CHECK: vpabsw -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpabsw -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpabsw 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpabsw 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096(%edx), %xmm1 

// CHECK: vpabsw 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096, %xmm1 

// CHECK: vpabsw 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x4c,0x02,0x40]       
vpabsw 64(%edx,%eax), %xmm1 

// CHECK: vpabsw (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x0a]       
vpabsw (%edx), %xmm1 

// CHECK: vpabsw %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0xc9]       
vpabsw %xmm1, %xmm1 

// CHECK: vpackssdw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpackssdw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpackssdw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpackssdw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6b,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpackssdw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6b,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096, %xmm1, %xmm1 

// CHECK: vpackssdw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6b,0x4c,0x02,0x40]      
vpackssdw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpackssdw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6b,0x0a]      
vpackssdw (%edx), %xmm1, %xmm1 

// CHECK: vpackssdw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6b,0xc9]      
vpackssdw %xmm1, %xmm1, %xmm1 

// CHECK: vpacksswb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x63,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpacksswb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpacksswb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x63,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpacksswb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x63,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpacksswb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x63,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096, %xmm1, %xmm1 

// CHECK: vpacksswb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x63,0x4c,0x02,0x40]      
vpacksswb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpacksswb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x63,0x0a]      
vpacksswb (%edx), %xmm1, %xmm1 

// CHECK: vpacksswb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x63,0xc9]      
vpacksswb %xmm1, %xmm1, %xmm1 

// CHECK: vpackusdw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpackusdw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpackusdw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpackusdw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2b,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpackusdw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2b,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096, %xmm1, %xmm1 

// CHECK: vpackusdw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2b,0x4c,0x02,0x40]      
vpackusdw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpackusdw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2b,0x0a]      
vpackusdw (%edx), %xmm1, %xmm1 

// CHECK: vpackusdw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x2b,0xc9]      
vpackusdw %xmm1, %xmm1, %xmm1 

// CHECK: vpackuswb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x67,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpackuswb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpackuswb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x67,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpackuswb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x67,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpackuswb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x67,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096, %xmm1, %xmm1 

// CHECK: vpackuswb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x67,0x4c,0x02,0x40]      
vpackuswb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpackuswb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x67,0x0a]      
vpackuswb (%edx), %xmm1, %xmm1 

// CHECK: vpackuswb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x67,0xc9]      
vpackuswb %xmm1, %xmm1, %xmm1 

// CHECK: vpaddb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfc,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfc,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfc,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfc,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096, %xmm1, %xmm1 

// CHECK: vpaddb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfc,0x4c,0x02,0x40]      
vpaddb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfc,0x0a]      
vpaddb (%edx), %xmm1, %xmm1 

// CHECK: vpaddb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfc,0xc9]      
vpaddb %xmm1, %xmm1, %xmm1 

// CHECK: vpaddd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfe,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfe,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfe,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfe,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096, %xmm1, %xmm1 

// CHECK: vpaddd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfe,0x4c,0x02,0x40]      
vpaddd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfe,0x0a]      
vpaddd (%edx), %xmm1, %xmm1 

// CHECK: vpaddd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfe,0xc9]      
vpaddd %xmm1, %xmm1, %xmm1 

// CHECK: vpaddq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd4,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd4,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd4,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd4,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096, %xmm1, %xmm1 

// CHECK: vpaddq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd4,0x4c,0x02,0x40]      
vpaddq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd4,0x0a]      
vpaddq (%edx), %xmm1, %xmm1 

// CHECK: vpaddq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd4,0xc9]      
vpaddq %xmm1, %xmm1, %xmm1 

// CHECK: vpaddsb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xec,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddsb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddsb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xec,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddsb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xec,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddsb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xec,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096, %xmm1, %xmm1 

// CHECK: vpaddsb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xec,0x4c,0x02,0x40]      
vpaddsb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddsb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xec,0x0a]      
vpaddsb (%edx), %xmm1, %xmm1 

// CHECK: vpaddsb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xec,0xc9]      
vpaddsb %xmm1, %xmm1, %xmm1 

// CHECK: vpaddsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xed,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xed,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xed,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xed,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096, %xmm1, %xmm1 

// CHECK: vpaddsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xed,0x4c,0x02,0x40]      
vpaddsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xed,0x0a]      
vpaddsw (%edx), %xmm1, %xmm1 

// CHECK: vpaddsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xed,0xc9]      
vpaddsw %xmm1, %xmm1, %xmm1 

// CHECK: vpaddusb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdc,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddusb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddusb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdc,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddusb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdc,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddusb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdc,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096, %xmm1, %xmm1 

// CHECK: vpaddusb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdc,0x4c,0x02,0x40]      
vpaddusb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddusb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdc,0x0a]      
vpaddusb (%edx), %xmm1, %xmm1 

// CHECK: vpaddusb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdc,0xc9]      
vpaddusb %xmm1, %xmm1, %xmm1 

// CHECK: vpaddusw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddusw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddusw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddusw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdd,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddusw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdd,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096, %xmm1, %xmm1 

// CHECK: vpaddusw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdd,0x4c,0x02,0x40]      
vpaddusw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddusw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdd,0x0a]      
vpaddusw (%edx), %xmm1, %xmm1 

// CHECK: vpaddusw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdd,0xc9]      
vpaddusw %xmm1, %xmm1, %xmm1 

// CHECK: vpaddw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpaddw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfd,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpaddw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfd,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096, %xmm1, %xmm1 

// CHECK: vpaddw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfd,0x4c,0x02,0x40]      
vpaddw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpaddw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfd,0x0a]      
vpaddw (%edx), %xmm1, %xmm1 

// CHECK: vpaddw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfd,0xc9]      
vpaddw %xmm1, %xmm1, %xmm1 

// CHECK: vpalignr $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0f,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpalignr $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpalignr $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpalignr $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0f,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpalignr $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0f,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096, %xmm1, %xmm1 

// CHECK: vpalignr $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0f,0x4c,0x02,0x40,0x00]     
vpalignr $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpalignr $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0f,0x0a,0x00]     
vpalignr $0, (%edx), %xmm1, %xmm1 

// CHECK: vpalignr $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0f,0xc9,0x00]     
vpalignr $0, %xmm1, %xmm1, %xmm1 

// CHECK: vpand -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpand -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpand 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpand 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdb,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpand 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdb,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096, %xmm1, %xmm1 

// CHECK: vpand 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdb,0x4c,0x02,0x40]      
vpand 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpand (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdb,0x0a]      
vpand (%edx), %xmm1, %xmm1 

// CHECK: vpandn -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpandn -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpandn 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpandn 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdf,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpandn 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdf,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096, %xmm1, %xmm1 

// CHECK: vpandn 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdf,0x4c,0x02,0x40]      
vpandn 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpandn (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdf,0x0a]      
vpandn (%edx), %xmm1, %xmm1 

// CHECK: vpandn %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdf,0xc9]      
vpandn %xmm1, %xmm1, %xmm1 

// CHECK: vpand %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xdb,0xc9]      
vpand %xmm1, %xmm1, %xmm1 

// CHECK: vpavgb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe0,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpavgb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpavgb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe0,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpavgb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe0,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpavgb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe0,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096, %xmm1, %xmm1 

// CHECK: vpavgb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe0,0x4c,0x02,0x40]      
vpavgb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpavgb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe0,0x0a]      
vpavgb (%edx), %xmm1, %xmm1 

// CHECK: vpavgb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe0,0xc9]      
vpavgb %xmm1, %xmm1, %xmm1 

// CHECK: vpavgw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe3,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpavgw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpavgw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe3,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpavgw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe3,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpavgw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe3,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096, %xmm1, %xmm1 

// CHECK: vpavgw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe3,0x4c,0x02,0x40]      
vpavgw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpavgw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe3,0x0a]      
vpavgw (%edx), %xmm1, %xmm1 

// CHECK: vpavgw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe3,0xc9]      
vpavgw %xmm1, %xmm1, %xmm1 

// CHECK: vpblendvb %xmm1, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x10]     
vpblendvb %xmm1, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpblendvb %xmm1, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x10]     
vpblendvb %xmm1, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpblendvb %xmm1, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0x8a,0xf0,0x1c,0xf0,0x1c,0x10]     
vpblendvb %xmm1, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpblendvb %xmm1, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0x0d,0xf0,0x1c,0xf0,0x1c,0x10]     
vpblendvb %xmm1, 485498096, %xmm1, %xmm1 

// CHECK: vpblendvb %xmm1, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0x4c,0x02,0x40,0x10]     
vpblendvb %xmm1, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpblendvb %xmm1, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0x0a,0x10]     
vpblendvb %xmm1, (%edx), %xmm1, %xmm1 

// CHECK: vpblendvb %xmm1, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0xc9,0x10]     
vpblendvb %xmm1, %xmm1, %xmm1, %xmm1 

// CHECK: vpblendw $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0e,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpblendw $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpblendw $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpblendw $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0e,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpblendw $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0e,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096, %xmm1, %xmm1 

// CHECK: vpblendw $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0e,0x4c,0x02,0x40,0x00]     
vpblendw $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpblendw $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0e,0x0a,0x00]     
vpblendw $0, (%edx), %xmm1, %xmm1 

// CHECK: vpblendw $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0e,0xc9,0x00]     
vpblendw $0, %xmm1, %xmm1, %xmm1 

// CHECK: vpclmulqdq $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x44,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpclmulqdq $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpclmulqdq $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x44,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpclmulqdq $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpclmulqdq $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x44,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vpclmulqdq $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpclmulqdq $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x44,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vpclmulqdq $0, 485498096, %xmm1, %xmm1 

// CHECK: vpclmulqdq $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x44,0x4c,0x02,0x40,0x00]     
vpclmulqdq $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpclmulqdq $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x44,0x0a,0x00]     
vpclmulqdq $0, (%edx), %xmm1, %xmm1 

// CHECK: vpclmulqdq $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x44,0xc9,0x00]     
vpclmulqdq $0, %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpeqb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x74,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x74,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x74,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x74,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096, %xmm1, %xmm1 

// CHECK: vpcmpeqb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x74,0x4c,0x02,0x40]      
vpcmpeqb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpeqb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x74,0x0a]      
vpcmpeqb (%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x74,0xc9]      
vpcmpeqb %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpeqd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x76,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x76,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x76,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x76,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096, %xmm1, %xmm1 

// CHECK: vpcmpeqd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x76,0x4c,0x02,0x40]      
vpcmpeqd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpeqd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x76,0x0a]      
vpcmpeqd (%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x76,0xc9]      
vpcmpeqd %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpeqq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x29,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x29,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x29,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x29,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096, %xmm1, %xmm1 

// CHECK: vpcmpeqq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x29,0x4c,0x02,0x40]      
vpcmpeqq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpeqq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x29,0x0a]      
vpcmpeqq (%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x29,0xc9]      
vpcmpeqq %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpeqw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x75,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x75,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpeqw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x75,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x75,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096, %xmm1, %xmm1 

// CHECK: vpcmpeqw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x75,0x4c,0x02,0x40]      
vpcmpeqw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpeqw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x75,0x0a]      
vpcmpeqw (%edx), %xmm1, %xmm1 

// CHECK: vpcmpeqw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x75,0xc9]      
vpcmpeqw %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpestri $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpcmpestri $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpestri $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestri $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpestri $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestri $0, 485498096(%edx), %xmm1 

// CHECK: vpcmpestri $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestri $0, 485498096, %xmm1 

// CHECK: vpcmpestri $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x4c,0x02,0x40,0x00]      
vpcmpestri $0, 64(%edx,%eax), %xmm1 

// CHECK: vpcmpestri $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x0a,0x00]      
vpcmpestri $0, (%edx), %xmm1 

// CHECK: vpcmpestri $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0xc9,0x00]      
vpcmpestri $0, %xmm1, %xmm1 

// CHECK: vpcmpestrm $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpcmpestrm $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpestrm $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestrm $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpestrm $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestrm $0, 485498096(%edx), %xmm1 

// CHECK: vpcmpestrm $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestrm $0, 485498096, %xmm1 

// CHECK: vpcmpestrm $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x4c,0x02,0x40,0x00]      
vpcmpestrm $0, 64(%edx,%eax), %xmm1 

// CHECK: vpcmpestrm $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x0a,0x00]      
vpcmpestrm $0, (%edx), %xmm1 

// CHECK: vpcmpestrm $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0xc9,0x00]      
vpcmpestrm $0, %xmm1, %xmm1 

// CHECK: vpcmpgtb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x64,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x64,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x64,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x64,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096, %xmm1, %xmm1 

// CHECK: vpcmpgtb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x64,0x4c,0x02,0x40]      
vpcmpgtb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpgtb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x64,0x0a]      
vpcmpgtb (%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x64,0xc9]      
vpcmpgtb %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpgtd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x66,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x66,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x66,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x66,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096, %xmm1, %xmm1 

// CHECK: vpcmpgtd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x66,0x4c,0x02,0x40]      
vpcmpgtd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpgtd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x66,0x0a]      
vpcmpgtd (%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x66,0xc9]      
vpcmpgtd %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpgtq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x37,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x37,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x37,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x37,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096, %xmm1, %xmm1 

// CHECK: vpcmpgtq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x37,0x4c,0x02,0x40]      
vpcmpgtq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpgtq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x37,0x0a]      
vpcmpgtq (%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x37,0xc9]      
vpcmpgtq %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpgtw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x65,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x65,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpcmpgtw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x65,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x65,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096, %xmm1, %xmm1 

// CHECK: vpcmpgtw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x65,0x4c,0x02,0x40]      
vpcmpgtw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpcmpgtw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x65,0x0a]      
vpcmpgtw (%edx), %xmm1, %xmm1 

// CHECK: vpcmpgtw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x65,0xc9]      
vpcmpgtw %xmm1, %xmm1, %xmm1 

// CHECK: vpcmpistri $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpcmpistri $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpistri $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistri $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpistri $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistri $0, 485498096(%edx), %xmm1 

// CHECK: vpcmpistri $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistri $0, 485498096, %xmm1 

// CHECK: vpcmpistri $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x4c,0x02,0x40,0x00]      
vpcmpistri $0, 64(%edx,%eax), %xmm1 

// CHECK: vpcmpistri $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x0a,0x00]      
vpcmpistri $0, (%edx), %xmm1 

// CHECK: vpcmpistri $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0xc9,0x00]      
vpcmpistri $0, %xmm1, %xmm1 

// CHECK: vpcmpistrm $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpcmpistrm $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpistrm $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistrm $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpcmpistrm $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistrm $0, 485498096(%edx), %xmm1 

// CHECK: vpcmpistrm $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistrm $0, 485498096, %xmm1 

// CHECK: vpcmpistrm $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x4c,0x02,0x40,0x00]      
vpcmpistrm $0, 64(%edx,%eax), %xmm1 

// CHECK: vpcmpistrm $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x0a,0x00]      
vpcmpistrm $0, (%edx), %xmm1 

// CHECK: vpcmpistrm $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0xc9,0x00]      
vpcmpistrm $0, %xmm1, %xmm1 

// CHECK: vperm2f128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x06,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vperm2f128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vperm2f128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x06,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2f128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vperm2f128 $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x06,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2f128 $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vperm2f128 $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x06,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2f128 $0, 485498096, %ymm4, %ymm4 

// CHECK: vperm2f128 $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x06,0x64,0x02,0x40,0x00]     
vperm2f128 $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vperm2f128 $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x06,0x22,0x00]     
vperm2f128 $0, (%edx), %ymm4, %ymm4 

// CHECK: vperm2f128 $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x06,0xe4,0x00]     
vperm2f128 $0, %ymm4, %ymm4, %ymm4 

// CHECK: vpermilpd $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpermilpd $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpermilpd $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpermilpd $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpermilpd $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermilpd $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermilpd $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096(%edx), %xmm1 

// CHECK: vpermilpd $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096(%edx), %ymm4 

// CHECK: vpermilpd $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096, %xmm1 

// CHECK: vpermilpd $0, 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096, %ymm4 

// CHECK: vpermilpd $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x4c,0x02,0x40,0x00]      
vpermilpd $0, 64(%edx,%eax), %xmm1 

// CHECK: vpermilpd $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x64,0x02,0x40,0x00]      
vpermilpd $0, 64(%edx,%eax), %ymm4 

// CHECK: vpermilpd $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x0a,0x00]      
vpermilpd $0, (%edx), %xmm1 

// CHECK: vpermilpd $0, (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x22,0x00]      
vpermilpd $0, (%edx), %ymm4 

// CHECK: vpermilpd $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0xc9,0x00]      
vpermilpd $0, %xmm1, %xmm1 

// CHECK: vpermilpd $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0xe4,0x00]      
vpermilpd $0, %ymm4, %ymm4 

// CHECK: vpermilpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpermilpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpermilpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpermilpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpermilpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermilpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermilpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpermilpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpermilpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096, %xmm1, %xmm1 

// CHECK: vpermilpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0d,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096, %ymm4, %ymm4 

// CHECK: vpermilpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0d,0x4c,0x02,0x40]      
vpermilpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpermilpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0d,0x64,0x02,0x40]      
vpermilpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpermilpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0d,0x0a]      
vpermilpd (%edx), %xmm1, %xmm1 

// CHECK: vpermilpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0d,0x22]      
vpermilpd (%edx), %ymm4, %ymm4 

// CHECK: vpermilpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0d,0xc9]      
vpermilpd %xmm1, %xmm1, %xmm1 

// CHECK: vpermilpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0d,0xe4]      
vpermilpd %ymm4, %ymm4, %ymm4 

// CHECK: vpermilps $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpermilps $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpermilps $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpermilps $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpermilps $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermilps $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermilps $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096(%edx), %xmm1 

// CHECK: vpermilps $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096(%edx), %ymm4 

// CHECK: vpermilps $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096, %xmm1 

// CHECK: vpermilps $0, 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096, %ymm4 

// CHECK: vpermilps $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x4c,0x02,0x40,0x00]      
vpermilps $0, 64(%edx,%eax), %xmm1 

// CHECK: vpermilps $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x64,0x02,0x40,0x00]      
vpermilps $0, 64(%edx,%eax), %ymm4 

// CHECK: vpermilps $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x0a,0x00]      
vpermilps $0, (%edx), %xmm1 

// CHECK: vpermilps $0, (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x22,0x00]      
vpermilps $0, (%edx), %ymm4 

// CHECK: vpermilps $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0xc9,0x00]      
vpermilps $0, %xmm1, %xmm1 

// CHECK: vpermilps $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0xe4,0x00]      
vpermilps $0, %ymm4, %ymm4 

// CHECK: vpermilps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpermilps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpermilps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpermilps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpermilps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermilps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermilps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpermilps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpermilps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096, %xmm1, %xmm1 

// CHECK: vpermilps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096, %ymm4, %ymm4 

// CHECK: vpermilps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0c,0x4c,0x02,0x40]      
vpermilps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpermilps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0c,0x64,0x02,0x40]      
vpermilps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpermilps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0c,0x0a]      
vpermilps (%edx), %xmm1, %xmm1 

// CHECK: vpermilps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0c,0x22]      
vpermilps (%edx), %ymm4, %ymm4 

// CHECK: vpermilps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0c,0xc9]      
vpermilps %xmm1, %xmm1, %xmm1 

// CHECK: vpermilps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0c,0xe4]      
vpermilps %ymm4, %ymm4, %ymm4 

// CHECK: vpextrb $0, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrb $0, %xmm1, 485498096 

// CHECK: vpextrb $0, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrb $0, %xmm1, 485498096(%edx) 

// CHECK: vpextrb $0, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpextrb $0, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vpextrb $0, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrb $0, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vpextrb $0, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x4c,0x02,0x40,0x00]      
vpextrb $0, %xmm1, 64(%edx,%eax) 

// CHECK: vpextrb $0, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x0a,0x00]      
vpextrb $0, %xmm1, (%edx) 

// CHECK: vpextrd $0, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrd $0, %xmm1, 485498096 

// CHECK: vpextrd $0, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrd $0, %xmm1, 485498096(%edx) 

// CHECK: vpextrd $0, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpextrd $0, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vpextrd $0, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrd $0, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vpextrd $0, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x4c,0x02,0x40,0x00]      
vpextrd $0, %xmm1, 64(%edx,%eax) 

// CHECK: vpextrd $0, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x0a,0x00]      
vpextrd $0, %xmm1, (%edx) 

// CHECK: vpextrw $0, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrw $0, %xmm1, 485498096 

// CHECK: vpextrw $0, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrw $0, %xmm1, 485498096(%edx) 

// CHECK: vpextrw $0, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpextrw $0, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vpextrw $0, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrw $0, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vpextrw $0, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x4c,0x02,0x40,0x00]      
vpextrw $0, %xmm1, 64(%edx,%eax) 

// CHECK: vpextrw $0, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x0a,0x00]      
vpextrw $0, %xmm1, (%edx) 

// CHECK: vphaddd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x02,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vphaddd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphaddd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x02,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphaddd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x02,0x8a,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vphaddd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x02,0x0d,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096, %xmm1, %xmm1 

// CHECK: vphaddd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x02,0x4c,0x02,0x40]      
vphaddd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vphaddd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x02,0x0a]      
vphaddd (%edx), %xmm1, %xmm1 

// CHECK: vphaddd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x02,0xc9]      
vphaddd %xmm1, %xmm1, %xmm1 

// CHECK: vphaddsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x03,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vphaddsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphaddsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x03,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphaddsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x03,0x8a,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vphaddsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x03,0x0d,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096, %xmm1, %xmm1 

// CHECK: vphaddsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x03,0x4c,0x02,0x40]      
vphaddsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vphaddsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x03,0x0a]      
vphaddsw (%edx), %xmm1, %xmm1 

// CHECK: vphaddsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x03,0xc9]      
vphaddsw %xmm1, %xmm1, %xmm1 

// CHECK: vphaddw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x01,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vphaddw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphaddw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x01,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphaddw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x01,0x8a,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vphaddw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x01,0x0d,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096, %xmm1, %xmm1 

// CHECK: vphaddw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x01,0x4c,0x02,0x40]      
vphaddw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vphaddw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x01,0x0a]      
vphaddw (%edx), %xmm1, %xmm1 

// CHECK: vphaddw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x01,0xc9]      
vphaddw %xmm1, %xmm1, %xmm1 

// CHECK: vphminposuw -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vphminposuw -485498096(%edx,%eax,4), %xmm1 

// CHECK: vphminposuw 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vphminposuw 485498096(%edx,%eax,4), %xmm1 

// CHECK: vphminposuw 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x8a,0xf0,0x1c,0xf0,0x1c]       
vphminposuw 485498096(%edx), %xmm1 

// CHECK: vphminposuw 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x0d,0xf0,0x1c,0xf0,0x1c]       
vphminposuw 485498096, %xmm1 

// CHECK: vphminposuw 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x4c,0x02,0x40]       
vphminposuw 64(%edx,%eax), %xmm1 

// CHECK: vphminposuw (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x0a]       
vphminposuw (%edx), %xmm1 

// CHECK: vphminposuw %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0xc9]       
vphminposuw %xmm1, %xmm1 

// CHECK: vphsubd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x06,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vphsubd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphsubd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x06,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphsubd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x06,0x8a,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vphsubd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x06,0x0d,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096, %xmm1, %xmm1 

// CHECK: vphsubd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x06,0x4c,0x02,0x40]      
vphsubd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vphsubd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x06,0x0a]      
vphsubd (%edx), %xmm1, %xmm1 

// CHECK: vphsubd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x06,0xc9]      
vphsubd %xmm1, %xmm1, %xmm1 

// CHECK: vphsubsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x07,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vphsubsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphsubsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x07,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphsubsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x07,0x8a,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vphsubsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x07,0x0d,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096, %xmm1, %xmm1 

// CHECK: vphsubsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x07,0x4c,0x02,0x40]      
vphsubsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vphsubsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x07,0x0a]      
vphsubsw (%edx), %xmm1, %xmm1 

// CHECK: vphsubsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x07,0xc9]      
vphsubsw %xmm1, %xmm1, %xmm1 

// CHECK: vphsubw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x05,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vphsubw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphsubw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x05,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vphsubw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x05,0x8a,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vphsubw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x05,0x0d,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096, %xmm1, %xmm1 

// CHECK: vphsubw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x05,0x4c,0x02,0x40]      
vphsubw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vphsubw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x05,0x0a]      
vphsubw (%edx), %xmm1, %xmm1 

// CHECK: vphsubw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x05,0xc9]      
vphsubw %xmm1, %xmm1, %xmm1 

// CHECK: vpinsrb $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x20,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpinsrb $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpinsrb $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x20,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrb $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpinsrb $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x20,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrb $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpinsrb $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x20,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrb $0, 485498096, %xmm1, %xmm1 

// CHECK: vpinsrb $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x20,0x4c,0x02,0x40,0x00]     
vpinsrb $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpinsrb $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x20,0x0a,0x00]     
vpinsrb $0, (%edx), %xmm1, %xmm1 

// CHECK: vpinsrd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x22,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpinsrd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpinsrd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x22,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpinsrd $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x22,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrd $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpinsrd $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x22,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrd $0, 485498096, %xmm1, %xmm1 

// CHECK: vpinsrd $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x22,0x4c,0x02,0x40,0x00]     
vpinsrd $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpinsrd $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x22,0x0a,0x00]     
vpinsrd $0, (%edx), %xmm1, %xmm1 

// CHECK: vpinsrw $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc4,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpinsrw $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpinsrw $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc4,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrw $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpinsrw $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc4,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrw $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpinsrw $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc4,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrw $0, 485498096, %xmm1, %xmm1 

// CHECK: vpinsrw $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc4,0x4c,0x02,0x40,0x00]     
vpinsrw $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpinsrw $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc4,0x0a,0x00]     
vpinsrw $0, (%edx), %xmm1, %xmm1 

// CHECK: vpmaddubsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x04,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaddubsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaddubsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x04,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaddubsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x04,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaddubsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x04,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096, %xmm1, %xmm1 

// CHECK: vpmaddubsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x04,0x4c,0x02,0x40]      
vpmaddubsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaddubsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x04,0x0a]      
vpmaddubsw (%edx), %xmm1, %xmm1 

// CHECK: vpmaddubsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x04,0xc9]      
vpmaddubsw %xmm1, %xmm1, %xmm1 

// CHECK: vpmaddwd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf5,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaddwd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaddwd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf5,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaddwd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf5,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaddwd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf5,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096, %xmm1, %xmm1 

// CHECK: vpmaddwd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf5,0x4c,0x02,0x40]      
vpmaddwd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaddwd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf5,0x0a]      
vpmaddwd (%edx), %xmm1, %xmm1 

// CHECK: vpmaddwd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf5,0xc9]      
vpmaddwd %xmm1, %xmm1, %xmm1 

// CHECK: vpmaxsb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxsb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxsb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxsb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaxsb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096, %xmm1, %xmm1 

// CHECK: vpmaxsb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3c,0x4c,0x02,0x40]      
vpmaxsb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaxsb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3c,0x0a]      
vpmaxsb (%edx), %xmm1, %xmm1 

// CHECK: vpmaxsb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3c,0xc9]      
vpmaxsb %xmm1, %xmm1, %xmm1 

// CHECK: vpmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaxsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096, %xmm1, %xmm1 

// CHECK: vpmaxsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3d,0x4c,0x02,0x40]      
vpmaxsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaxsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3d,0x0a]      
vpmaxsd (%edx), %xmm1, %xmm1 

// CHECK: vpmaxsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3d,0xc9]      
vpmaxsd %xmm1, %xmm1, %xmm1 

// CHECK: vpmaxsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xee,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xee,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xee,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaxsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xee,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096, %xmm1, %xmm1 

// CHECK: vpmaxsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xee,0x4c,0x02,0x40]      
vpmaxsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaxsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xee,0x0a]      
vpmaxsw (%edx), %xmm1, %xmm1 

// CHECK: vpmaxsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xee,0xc9]      
vpmaxsw %xmm1, %xmm1, %xmm1 

// CHECK: vpmaxub -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xde,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxub -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxub 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xde,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxub 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xde,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaxub 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xde,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096, %xmm1, %xmm1 

// CHECK: vpmaxub 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xde,0x4c,0x02,0x40]      
vpmaxub 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaxub (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xde,0x0a]      
vpmaxub (%edx), %xmm1, %xmm1 

// CHECK: vpmaxub %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xde,0xc9]      
vpmaxub %xmm1, %xmm1, %xmm1 

// CHECK: vpmaxud -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxud -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxud 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxud 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaxud 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096, %xmm1, %xmm1 

// CHECK: vpmaxud 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3f,0x4c,0x02,0x40]      
vpmaxud 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaxud (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3f,0x0a]      
vpmaxud (%edx), %xmm1, %xmm1 

// CHECK: vpmaxud %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3f,0xc9]      
vpmaxud %xmm1, %xmm1, %xmm1 

// CHECK: vpmaxuw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxuw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxuw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaxuw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaxuw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096, %xmm1, %xmm1 

// CHECK: vpmaxuw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3e,0x4c,0x02,0x40]      
vpmaxuw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaxuw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3e,0x0a]      
vpmaxuw (%edx), %xmm1, %xmm1 

// CHECK: vpmaxuw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3e,0xc9]      
vpmaxuw %xmm1, %xmm1, %xmm1 

// CHECK: vpminsb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x38,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpminsb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminsb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x38,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminsb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x38,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpminsb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x38,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096, %xmm1, %xmm1 

// CHECK: vpminsb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x38,0x4c,0x02,0x40]      
vpminsb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpminsb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x38,0x0a]      
vpminsb (%edx), %xmm1, %xmm1 

// CHECK: vpminsb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x38,0xc9]      
vpminsb %xmm1, %xmm1, %xmm1 

// CHECK: vpminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x39,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x39,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x39,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpminsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x39,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096, %xmm1, %xmm1 

// CHECK: vpminsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x39,0x4c,0x02,0x40]      
vpminsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpminsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x39,0x0a]      
vpminsd (%edx), %xmm1, %xmm1 

// CHECK: vpminsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x39,0xc9]      
vpminsd %xmm1, %xmm1, %xmm1 

// CHECK: vpminsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xea,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpminsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xea,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xea,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpminsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xea,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096, %xmm1, %xmm1 

// CHECK: vpminsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xea,0x4c,0x02,0x40]      
vpminsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpminsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xea,0x0a]      
vpminsw (%edx), %xmm1, %xmm1 

// CHECK: vpminsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xea,0xc9]      
vpminsw %xmm1, %xmm1, %xmm1 

// CHECK: vpminub -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xda,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpminub -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminub 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xda,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminub 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xda,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpminub 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xda,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096, %xmm1, %xmm1 

// CHECK: vpminub 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xda,0x4c,0x02,0x40]      
vpminub 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpminub (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xda,0x0a]      
vpminub (%edx), %xmm1, %xmm1 

// CHECK: vpminub %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xda,0xc9]      
vpminub %xmm1, %xmm1, %xmm1 

// CHECK: vpminud -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpminud -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminud 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminud 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3b,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpminud 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3b,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096, %xmm1, %xmm1 

// CHECK: vpminud 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3b,0x4c,0x02,0x40]      
vpminud 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpminud (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3b,0x0a]      
vpminud (%edx), %xmm1, %xmm1 

// CHECK: vpminud %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3b,0xc9]      
vpminud %xmm1, %xmm1, %xmm1 

// CHECK: vpminuw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpminuw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminuw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpminuw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpminuw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096, %xmm1, %xmm1 

// CHECK: vpminuw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3a,0x4c,0x02,0x40]      
vpminuw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpminuw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3a,0x0a]      
vpminuw (%edx), %xmm1, %xmm1 

// CHECK: vpminuw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x3a,0xc9]      
vpminuw %xmm1, %xmm1, %xmm1 

// CHECK: vpmovsxbd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxbd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxbd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxbd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096(%edx), %xmm1 

// CHECK: vpmovsxbd 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096, %xmm1 

// CHECK: vpmovsxbd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x4c,0x02,0x40]       
vpmovsxbd 64(%edx,%eax), %xmm1 

// CHECK: vpmovsxbd (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x0a]       
vpmovsxbd (%edx), %xmm1 

// CHECK: vpmovsxbd %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0xc9]       
vpmovsxbd %xmm1, %xmm1 

// CHECK: vpmovsxbq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxbq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxbq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxbq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096(%edx), %xmm1 

// CHECK: vpmovsxbq 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096, %xmm1 

// CHECK: vpmovsxbq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x4c,0x02,0x40]       
vpmovsxbq 64(%edx,%eax), %xmm1 

// CHECK: vpmovsxbq (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x0a]       
vpmovsxbq (%edx), %xmm1 

// CHECK: vpmovsxbq %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0xc9]       
vpmovsxbq %xmm1, %xmm1 

// CHECK: vpmovsxbw -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxbw -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxbw 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxbw 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096(%edx), %xmm1 

// CHECK: vpmovsxbw 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096, %xmm1 

// CHECK: vpmovsxbw 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x4c,0x02,0x40]       
vpmovsxbw 64(%edx,%eax), %xmm1 

// CHECK: vpmovsxbw (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x0a]       
vpmovsxbw (%edx), %xmm1 

// CHECK: vpmovsxbw %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0xc9]       
vpmovsxbw %xmm1, %xmm1 

// CHECK: vpmovsxdq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxdq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxdq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxdq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096(%edx), %xmm1 

// CHECK: vpmovsxdq 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096, %xmm1 

// CHECK: vpmovsxdq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x4c,0x02,0x40]       
vpmovsxdq 64(%edx,%eax), %xmm1 

// CHECK: vpmovsxdq (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x0a]       
vpmovsxdq (%edx), %xmm1 

// CHECK: vpmovsxdq %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0xc9]       
vpmovsxdq %xmm1, %xmm1 

// CHECK: vpmovsxwd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxwd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxwd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxwd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096(%edx), %xmm1 

// CHECK: vpmovsxwd 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096, %xmm1 

// CHECK: vpmovsxwd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x4c,0x02,0x40]       
vpmovsxwd 64(%edx,%eax), %xmm1 

// CHECK: vpmovsxwd (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x0a]       
vpmovsxwd (%edx), %xmm1 

// CHECK: vpmovsxwd %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0xc9]       
vpmovsxwd %xmm1, %xmm1 

// CHECK: vpmovsxwq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxwq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxwq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovsxwq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096(%edx), %xmm1 

// CHECK: vpmovsxwq 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096, %xmm1 

// CHECK: vpmovsxwq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x4c,0x02,0x40]       
vpmovsxwq 64(%edx,%eax), %xmm1 

// CHECK: vpmovsxwq (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x0a]       
vpmovsxwq (%edx), %xmm1 

// CHECK: vpmovsxwq %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0xc9]       
vpmovsxwq %xmm1, %xmm1 

// CHECK: vpmovzxbd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxbd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxbd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxbd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096(%edx), %xmm1 

// CHECK: vpmovzxbd 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096, %xmm1 

// CHECK: vpmovzxbd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x4c,0x02,0x40]       
vpmovzxbd 64(%edx,%eax), %xmm1 

// CHECK: vpmovzxbd (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x0a]       
vpmovzxbd (%edx), %xmm1 

// CHECK: vpmovzxbd %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0xc9]       
vpmovzxbd %xmm1, %xmm1 

// CHECK: vpmovzxbq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxbq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxbq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxbq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096(%edx), %xmm1 

// CHECK: vpmovzxbq 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096, %xmm1 

// CHECK: vpmovzxbq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x4c,0x02,0x40]       
vpmovzxbq 64(%edx,%eax), %xmm1 

// CHECK: vpmovzxbq (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x0a]       
vpmovzxbq (%edx), %xmm1 

// CHECK: vpmovzxbq %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0xc9]       
vpmovzxbq %xmm1, %xmm1 

// CHECK: vpmovzxbw -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxbw -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxbw 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxbw 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096(%edx), %xmm1 

// CHECK: vpmovzxbw 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096, %xmm1 

// CHECK: vpmovzxbw 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x4c,0x02,0x40]       
vpmovzxbw 64(%edx,%eax), %xmm1 

// CHECK: vpmovzxbw (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x0a]       
vpmovzxbw (%edx), %xmm1 

// CHECK: vpmovzxbw %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0xc9]       
vpmovzxbw %xmm1, %xmm1 

// CHECK: vpmovzxdq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxdq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxdq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxdq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096(%edx), %xmm1 

// CHECK: vpmovzxdq 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096, %xmm1 

// CHECK: vpmovzxdq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x4c,0x02,0x40]       
vpmovzxdq 64(%edx,%eax), %xmm1 

// CHECK: vpmovzxdq (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x0a]       
vpmovzxdq (%edx), %xmm1 

// CHECK: vpmovzxdq %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0xc9]       
vpmovzxdq %xmm1, %xmm1 

// CHECK: vpmovzxwd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxwd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxwd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxwd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096(%edx), %xmm1 

// CHECK: vpmovzxwd 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096, %xmm1 

// CHECK: vpmovzxwd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x4c,0x02,0x40]       
vpmovzxwd 64(%edx,%eax), %xmm1 

// CHECK: vpmovzxwd (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x0a]       
vpmovzxwd (%edx), %xmm1 

// CHECK: vpmovzxwd %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0xc9]       
vpmovzxwd %xmm1, %xmm1 

// CHECK: vpmovzxwq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxwq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxwq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpmovzxwq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096(%edx), %xmm1 

// CHECK: vpmovzxwq 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096, %xmm1 

// CHECK: vpmovzxwq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x4c,0x02,0x40]       
vpmovzxwq 64(%edx,%eax), %xmm1 

// CHECK: vpmovzxwq (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x0a]       
vpmovzxwq (%edx), %xmm1 

// CHECK: vpmovzxwq %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0xc9]       
vpmovzxwq %xmm1, %xmm1 

// CHECK: vpmuldq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x28,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmuldq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmuldq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x28,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmuldq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x28,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmuldq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x28,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096, %xmm1, %xmm1 

// CHECK: vpmuldq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x28,0x4c,0x02,0x40]      
vpmuldq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmuldq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x28,0x0a]      
vpmuldq (%edx), %xmm1, %xmm1 

// CHECK: vpmuldq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x28,0xc9]      
vpmuldq %xmm1, %xmm1, %xmm1 

// CHECK: vpmulhrsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulhrsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulhrsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulhrsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0b,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmulhrsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0b,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096, %xmm1, %xmm1 

// CHECK: vpmulhrsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0b,0x4c,0x02,0x40]      
vpmulhrsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmulhrsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0b,0x0a]      
vpmulhrsw (%edx), %xmm1, %xmm1 

// CHECK: vpmulhrsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0b,0xc9]      
vpmulhrsw %xmm1, %xmm1, %xmm1 

// CHECK: vpmulhuw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe4,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulhuw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulhuw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe4,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulhuw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe4,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmulhuw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe4,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096, %xmm1, %xmm1 

// CHECK: vpmulhuw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe4,0x4c,0x02,0x40]      
vpmulhuw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmulhuw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe4,0x0a]      
vpmulhuw (%edx), %xmm1, %xmm1 

// CHECK: vpmulhuw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe4,0xc9]      
vpmulhuw %xmm1, %xmm1, %xmm1 

// CHECK: vpmulhw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe5,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulhw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulhw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe5,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulhw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe5,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmulhw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe5,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096, %xmm1, %xmm1 

// CHECK: vpmulhw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe5,0x4c,0x02,0x40]      
vpmulhw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmulhw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe5,0x0a]      
vpmulhw (%edx), %xmm1, %xmm1 

// CHECK: vpmulhw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe5,0xc9]      
vpmulhw %xmm1, %xmm1, %xmm1 

// CHECK: vpmulld -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x40,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulld -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulld 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x40,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmulld 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x40,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmulld 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x40,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096, %xmm1, %xmm1 

// CHECK: vpmulld 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x40,0x4c,0x02,0x40]      
vpmulld 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmulld (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x40,0x0a]      
vpmulld (%edx), %xmm1, %xmm1 

// CHECK: vpmulld %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x40,0xc9]      
vpmulld %xmm1, %xmm1, %xmm1 

// CHECK: vpmullw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd5,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmullw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmullw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd5,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmullw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd5,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmullw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd5,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096, %xmm1, %xmm1 

// CHECK: vpmullw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd5,0x4c,0x02,0x40]      
vpmullw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmullw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd5,0x0a]      
vpmullw (%edx), %xmm1, %xmm1 

// CHECK: vpmullw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd5,0xc9]      
vpmullw %xmm1, %xmm1, %xmm1 

// CHECK: vpmuludq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf4,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmuludq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmuludq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf4,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmuludq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf4,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmuludq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf4,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096, %xmm1, %xmm1 

// CHECK: vpmuludq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf4,0x4c,0x02,0x40]      
vpmuludq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmuludq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf4,0x0a]      
vpmuludq (%edx), %xmm1, %xmm1 

// CHECK: vpmuludq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf4,0xc9]      
vpmuludq %xmm1, %xmm1, %xmm1 

// CHECK: vpor -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xeb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpor -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpor 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xeb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpor 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xeb,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpor 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xeb,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096, %xmm1, %xmm1 

// CHECK: vpor 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xeb,0x4c,0x02,0x40]      
vpor 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpor (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xeb,0x0a]      
vpor (%edx), %xmm1, %xmm1 

// CHECK: vpor %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xeb,0xc9]      
vpor %xmm1, %xmm1, %xmm1 

// CHECK: vpsadbw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsadbw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsadbw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsadbw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf6,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsadbw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf6,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096, %xmm1, %xmm1 

// CHECK: vpsadbw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf6,0x4c,0x02,0x40]      
vpsadbw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsadbw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf6,0x0a]      
vpsadbw (%edx), %xmm1, %xmm1 

// CHECK: vpsadbw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf6,0xc9]      
vpsadbw %xmm1, %xmm1, %xmm1 

// CHECK: vpshufb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x00,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpshufb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpshufb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x00,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpshufb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x00,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpshufb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x00,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096, %xmm1, %xmm1 

// CHECK: vpshufb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x00,0x4c,0x02,0x40]      
vpshufb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpshufb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x00,0x0a]      
vpshufb (%edx), %xmm1, %xmm1 

// CHECK: vpshufb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x00,0xc9]      
vpshufb %xmm1, %xmm1, %xmm1 

// CHECK: vpshufd $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x70,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpshufd $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpshufd $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x70,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpshufd $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x70,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096(%edx), %xmm1 

// CHECK: vpshufd $0, 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x70,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096, %xmm1 

// CHECK: vpshufd $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x70,0x4c,0x02,0x40,0x00]      
vpshufd $0, 64(%edx,%eax), %xmm1 

// CHECK: vpshufd $0, (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x70,0x0a,0x00]      
vpshufd $0, (%edx), %xmm1 

// CHECK: vpshufd $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x70,0xc9,0x00]      
vpshufd $0, %xmm1, %xmm1 

// CHECK: vpshufhw $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x70,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpshufhw $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpshufhw $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x70,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpshufhw $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x70,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096(%edx), %xmm1 

// CHECK: vpshufhw $0, 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x70,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096, %xmm1 

// CHECK: vpshufhw $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x70,0x4c,0x02,0x40,0x00]      
vpshufhw $0, 64(%edx,%eax), %xmm1 

// CHECK: vpshufhw $0, (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x70,0x0a,0x00]      
vpshufhw $0, (%edx), %xmm1 

// CHECK: vpshufhw $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfa,0x70,0xc9,0x00]      
vpshufhw $0, %xmm1, %xmm1 

// CHECK: vpshuflw $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x70,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpshuflw $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpshuflw $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x70,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpshuflw $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x70,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096(%edx), %xmm1 

// CHECK: vpshuflw $0, 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x70,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096, %xmm1 

// CHECK: vpshuflw $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x70,0x4c,0x02,0x40,0x00]      
vpshuflw $0, 64(%edx,%eax), %xmm1 

// CHECK: vpshuflw $0, (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x70,0x0a,0x00]      
vpshuflw $0, (%edx), %xmm1 

// CHECK: vpshuflw $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xfb,0x70,0xc9,0x00]      
vpshuflw $0, %xmm1, %xmm1 

// CHECK: vpsignb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x08,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsignb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsignb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x08,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsignb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x08,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsignb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x08,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096, %xmm1, %xmm1 

// CHECK: vpsignb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x08,0x4c,0x02,0x40]      
vpsignb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsignb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x08,0x0a]      
vpsignb (%edx), %xmm1, %xmm1 

// CHECK: vpsignb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x08,0xc9]      
vpsignb %xmm1, %xmm1, %xmm1 

// CHECK: vpsignd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsignd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsignd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsignd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsignd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096, %xmm1, %xmm1 

// CHECK: vpsignd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0a,0x4c,0x02,0x40]      
vpsignd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsignd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0a,0x0a]      
vpsignd (%edx), %xmm1, %xmm1 

// CHECK: vpsignd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x0a,0xc9]      
vpsignd %xmm1, %xmm1, %xmm1 

// CHECK: vpsignw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x09,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsignw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsignw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x09,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsignw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x09,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsignw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x09,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096, %xmm1, %xmm1 

// CHECK: vpsignw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x09,0x4c,0x02,0x40]      
vpsignw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsignw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x09,0x0a]      
vpsignw (%edx), %xmm1, %xmm1 

// CHECK: vpsignw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x09,0xc9]      
vpsignw %xmm1, %xmm1, %xmm1 

// CHECK: vpslld $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x72,0xf1,0x00]      
vpslld $0, %xmm1, %xmm1 

// CHECK: vpslld -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf2,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpslld -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpslld 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpslld 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf2,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpslld 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf2,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096, %xmm1, %xmm1 

// CHECK: vpslld 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf2,0x4c,0x02,0x40]      
vpslld 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpslld (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf2,0x0a]      
vpslld (%edx), %xmm1, %xmm1 

// CHECK: vpslldq $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x73,0xf9,0x00]      
vpslldq $0, %xmm1, %xmm1 

// CHECK: vpslld %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf2,0xc9]      
vpslld %xmm1, %xmm1, %xmm1 

// CHECK: vpsllq $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x73,0xf1,0x00]      
vpsllq $0, %xmm1, %xmm1 

// CHECK: vpsllq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf3,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf3,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf3,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsllq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf3,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096, %xmm1, %xmm1 

// CHECK: vpsllq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf3,0x4c,0x02,0x40]      
vpsllq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsllq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf3,0x0a]      
vpsllq (%edx), %xmm1, %xmm1 

// CHECK: vpsllq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf3,0xc9]      
vpsllq %xmm1, %xmm1, %xmm1 

// CHECK: vpsllw $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x71,0xf1,0x00]      
vpsllw $0, %xmm1, %xmm1 

// CHECK: vpsllw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf1,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf1,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf1,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsllw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf1,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096, %xmm1, %xmm1 

// CHECK: vpsllw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf1,0x4c,0x02,0x40]      
vpsllw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsllw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf1,0x0a]      
vpsllw (%edx), %xmm1, %xmm1 

// CHECK: vpsllw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf1,0xc9]      
vpsllw %xmm1, %xmm1, %xmm1 

// CHECK: vpsrad $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x72,0xe1,0x00]      
vpsrad $0, %xmm1, %xmm1 

// CHECK: vpsrad -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe2,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrad -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrad 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrad 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe2,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsrad 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe2,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096, %xmm1, %xmm1 

// CHECK: vpsrad 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe2,0x4c,0x02,0x40]      
vpsrad 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsrad (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe2,0x0a]      
vpsrad (%edx), %xmm1, %xmm1 

// CHECK: vpsrad %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe2,0xc9]      
vpsrad %xmm1, %xmm1, %xmm1 

// CHECK: vpsraw $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x71,0xe1,0x00]      
vpsraw $0, %xmm1, %xmm1 

// CHECK: vpsraw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe1,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsraw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsraw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe1,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsraw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe1,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsraw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe1,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096, %xmm1, %xmm1 

// CHECK: vpsraw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe1,0x4c,0x02,0x40]      
vpsraw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsraw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe1,0x0a]      
vpsraw (%edx), %xmm1, %xmm1 

// CHECK: vpsraw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe1,0xc9]      
vpsraw %xmm1, %xmm1, %xmm1 

// CHECK: vpsrld $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x72,0xd1,0x00]      
vpsrld $0, %xmm1, %xmm1 

// CHECK: vpsrld -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd2,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrld -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrld 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrld 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd2,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsrld 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd2,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096, %xmm1, %xmm1 

// CHECK: vpsrld 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd2,0x4c,0x02,0x40]      
vpsrld 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsrld (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd2,0x0a]      
vpsrld (%edx), %xmm1, %xmm1 

// CHECK: vpsrldq $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x73,0xd9,0x00]      
vpsrldq $0, %xmm1, %xmm1 

// CHECK: vpsrld %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd2,0xc9]      
vpsrld %xmm1, %xmm1, %xmm1 

// CHECK: vpsrlq $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x73,0xd1,0x00]      
vpsrlq $0, %xmm1, %xmm1 

// CHECK: vpsrlq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd3,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd3,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd3,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsrlq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd3,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096, %xmm1, %xmm1 

// CHECK: vpsrlq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd3,0x4c,0x02,0x40]      
vpsrlq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsrlq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd3,0x0a]      
vpsrlq (%edx), %xmm1, %xmm1 

// CHECK: vpsrlq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd3,0xc9]      
vpsrlq %xmm1, %xmm1, %xmm1 

// CHECK: vpsrlw $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x71,0xd1,0x00]      
vpsrlw $0, %xmm1, %xmm1 

// CHECK: vpsrlw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd1,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd1,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd1,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsrlw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd1,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096, %xmm1, %xmm1 

// CHECK: vpsrlw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd1,0x4c,0x02,0x40]      
vpsrlw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsrlw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd1,0x0a]      
vpsrlw (%edx), %xmm1, %xmm1 

// CHECK: vpsrlw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd1,0xc9]      
vpsrlw %xmm1, %xmm1, %xmm1 

// CHECK: vpsubb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf8,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf8,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096, %xmm1, %xmm1 

// CHECK: vpsubb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf8,0x4c,0x02,0x40]      
vpsubb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf8,0x0a]      
vpsubb (%edx), %xmm1, %xmm1 

// CHECK: vpsubb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf8,0xc9]      
vpsubb %xmm1, %xmm1, %xmm1 

// CHECK: vpsubd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfa,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfa,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfa,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfa,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096, %xmm1, %xmm1 

// CHECK: vpsubd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfa,0x4c,0x02,0x40]      
vpsubd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfa,0x0a]      
vpsubd (%edx), %xmm1, %xmm1 

// CHECK: vpsubd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfa,0xc9]      
vpsubd %xmm1, %xmm1, %xmm1 

// CHECK: vpsubq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfb,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfb,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096, %xmm1, %xmm1 

// CHECK: vpsubq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfb,0x4c,0x02,0x40]      
vpsubq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfb,0x0a]      
vpsubq (%edx), %xmm1, %xmm1 

// CHECK: vpsubq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xfb,0xc9]      
vpsubq %xmm1, %xmm1, %xmm1 

// CHECK: vpsubsb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubsb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubsb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubsb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe8,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubsb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe8,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096, %xmm1, %xmm1 

// CHECK: vpsubsb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe8,0x4c,0x02,0x40]      
vpsubsb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubsb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe8,0x0a]      
vpsubsb (%edx), %xmm1, %xmm1 

// CHECK: vpsubsb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe8,0xc9]      
vpsubsb %xmm1, %xmm1, %xmm1 

// CHECK: vpsubsw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubsw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubsw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubsw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe9,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubsw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe9,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096, %xmm1, %xmm1 

// CHECK: vpsubsw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe9,0x4c,0x02,0x40]      
vpsubsw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubsw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe9,0x0a]      
vpsubsw (%edx), %xmm1, %xmm1 

// CHECK: vpsubsw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xe9,0xc9]      
vpsubsw %xmm1, %xmm1, %xmm1 

// CHECK: vpsubusb -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubusb -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubusb 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubusb 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd8,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubusb 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd8,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096, %xmm1, %xmm1 

// CHECK: vpsubusb 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd8,0x4c,0x02,0x40]      
vpsubusb 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubusb (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd8,0x0a]      
vpsubusb (%edx), %xmm1, %xmm1 

// CHECK: vpsubusb %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd8,0xc9]      
vpsubusb %xmm1, %xmm1, %xmm1 

// CHECK: vpsubusw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubusw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubusw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubusw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd9,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubusw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd9,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096, %xmm1, %xmm1 

// CHECK: vpsubusw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd9,0x4c,0x02,0x40]      
vpsubusw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubusw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd9,0x0a]      
vpsubusw (%edx), %xmm1, %xmm1 

// CHECK: vpsubusw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xd9,0xc9]      
vpsubusw %xmm1, %xmm1, %xmm1 

// CHECK: vpsubw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsubw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf9,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsubw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf9,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096, %xmm1, %xmm1 

// CHECK: vpsubw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf9,0x4c,0x02,0x40]      
vpsubw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsubw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf9,0x0a]      
vpsubw (%edx), %xmm1, %xmm1 

// CHECK: vpsubw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xf9,0xc9]      
vpsubw %xmm1, %xmm1, %xmm1 

// CHECK: vptest -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vptest -485498096(%edx,%eax,4), %xmm1 

// CHECK: vptest 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096(%edx,%eax,4), %xmm1 

// CHECK: vptest -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vptest -485498096(%edx,%eax,4), %ymm4 

// CHECK: vptest 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096(%edx,%eax,4), %ymm4 

// CHECK: vptest 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x8a,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096(%edx), %xmm1 

// CHECK: vptest 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0xa2,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096(%edx), %ymm4 

// CHECK: vptest 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x0d,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096, %xmm1 

// CHECK: vptest 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x25,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096, %ymm4 

// CHECK: vptest 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x4c,0x02,0x40]       
vptest 64(%edx,%eax), %xmm1 

// CHECK: vptest 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x64,0x02,0x40]       
vptest 64(%edx,%eax), %ymm4 

// CHECK: vptest (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x0a]       
vptest (%edx), %xmm1 

// CHECK: vptest (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x22]       
vptest (%edx), %ymm4 

// CHECK: vptest %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0xc9]       
vptest %xmm1, %xmm1 

// CHECK: vptest %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0xe4]       
vptest %ymm4, %ymm4 

// CHECK: vpunpckhbw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x68,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhbw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhbw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x68,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhbw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x68,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpckhbw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x68,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096, %xmm1, %xmm1 

// CHECK: vpunpckhbw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x68,0x4c,0x02,0x40]      
vpunpckhbw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpckhbw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x68,0x0a]      
vpunpckhbw (%edx), %xmm1, %xmm1 

// CHECK: vpunpckhbw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x68,0xc9]      
vpunpckhbw %xmm1, %xmm1, %xmm1 

// CHECK: vpunpckhdq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhdq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhdq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhdq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpckhdq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096, %xmm1, %xmm1 

// CHECK: vpunpckhdq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6a,0x4c,0x02,0x40]      
vpunpckhdq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpckhdq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6a,0x0a]      
vpunpckhdq (%edx), %xmm1, %xmm1 

// CHECK: vpunpckhdq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6a,0xc9]      
vpunpckhdq %xmm1, %xmm1, %xmm1 

// CHECK: vpunpckhqdq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhqdq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhqdq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhqdq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpckhqdq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096, %xmm1, %xmm1 

// CHECK: vpunpckhqdq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6d,0x4c,0x02,0x40]      
vpunpckhqdq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpckhqdq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6d,0x0a]      
vpunpckhqdq (%edx), %xmm1, %xmm1 

// CHECK: vpunpckhqdq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6d,0xc9]      
vpunpckhqdq %xmm1, %xmm1, %xmm1 

// CHECK: vpunpckhwd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x69,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhwd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhwd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x69,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckhwd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x69,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpckhwd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x69,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096, %xmm1, %xmm1 

// CHECK: vpunpckhwd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x69,0x4c,0x02,0x40]      
vpunpckhwd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpckhwd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x69,0x0a]      
vpunpckhwd (%edx), %xmm1, %xmm1 

// CHECK: vpunpckhwd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x69,0xc9]      
vpunpckhwd %xmm1, %xmm1, %xmm1 

// CHECK: vpunpcklbw -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x60,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpcklbw -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpcklbw 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x60,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpcklbw 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x60,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpcklbw 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x60,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096, %xmm1, %xmm1 

// CHECK: vpunpcklbw 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x60,0x4c,0x02,0x40]      
vpunpcklbw 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpcklbw (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x60,0x0a]      
vpunpcklbw (%edx), %xmm1, %xmm1 

// CHECK: vpunpcklbw %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x60,0xc9]      
vpunpcklbw %xmm1, %xmm1, %xmm1 

// CHECK: vpunpckldq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x62,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckldq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckldq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x62,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpckldq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x62,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpckldq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x62,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096, %xmm1, %xmm1 

// CHECK: vpunpckldq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x62,0x4c,0x02,0x40]      
vpunpckldq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpckldq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x62,0x0a]      
vpunpckldq (%edx), %xmm1, %xmm1 

// CHECK: vpunpckldq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x62,0xc9]      
vpunpckldq %xmm1, %xmm1, %xmm1 

// CHECK: vpunpcklqdq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpcklqdq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpcklqdq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpcklqdq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpcklqdq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096, %xmm1, %xmm1 

// CHECK: vpunpcklqdq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6c,0x4c,0x02,0x40]      
vpunpcklqdq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpcklqdq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6c,0x0a]      
vpunpcklqdq (%edx), %xmm1, %xmm1 

// CHECK: vpunpcklqdq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x6c,0xc9]      
vpunpcklqdq %xmm1, %xmm1, %xmm1 

// CHECK: vpunpcklwd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x61,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpcklwd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpcklwd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x61,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpunpcklwd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x61,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpunpcklwd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x61,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096, %xmm1, %xmm1 

// CHECK: vpunpcklwd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x61,0x4c,0x02,0x40]      
vpunpcklwd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpunpcklwd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x61,0x0a]      
vpunpcklwd (%edx), %xmm1, %xmm1 

// CHECK: vpunpcklwd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x61,0xc9]      
vpunpcklwd %xmm1, %xmm1, %xmm1 

// CHECK: vpxor -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xef,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpxor -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpxor 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xef,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpxor 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xef,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpxor 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xef,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096, %xmm1, %xmm1 

// CHECK: vpxor 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xef,0x4c,0x02,0x40]      
vpxor 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpxor (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xef,0x0a]      
vpxor (%edx), %xmm1, %xmm1 

// CHECK: vpxor %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xef,0xc9]      
vpxor %xmm1, %xmm1, %xmm1 

// CHECK: vrcpps -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x53,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vrcpps -485498096(%edx,%eax,4), %xmm1 

// CHECK: vrcpps 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x53,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096(%edx,%eax,4), %xmm1 

// CHECK: vrcpps -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x53,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vrcpps -485498096(%edx,%eax,4), %ymm4 

// CHECK: vrcpps 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x53,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096(%edx,%eax,4), %ymm4 

// CHECK: vrcpps 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x53,0x8a,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096(%edx), %xmm1 

// CHECK: vrcpps 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x53,0xa2,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096(%edx), %ymm4 

// CHECK: vrcpps 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x53,0x0d,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096, %xmm1 

// CHECK: vrcpps 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x53,0x25,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096, %ymm4 

// CHECK: vrcpps 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x53,0x4c,0x02,0x40]       
vrcpps 64(%edx,%eax), %xmm1 

// CHECK: vrcpps 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x53,0x64,0x02,0x40]       
vrcpps 64(%edx,%eax), %ymm4 

// CHECK: vrcpps (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x53,0x0a]       
vrcpps (%edx), %xmm1 

// CHECK: vrcpps (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x53,0x22]       
vrcpps (%edx), %ymm4 

// CHECK: vrcpps %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x53,0xc9]       
vrcpps %xmm1, %xmm1 

// CHECK: vrcpps %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x53,0xe4]       
vrcpps %ymm4, %ymm4 

// CHECK: vrcpss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x53,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vrcpss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vrcpss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x53,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vrcpss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vrcpss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x53,0x8a,0xf0,0x1c,0xf0,0x1c]      
vrcpss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vrcpss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x53,0x0d,0xf0,0x1c,0xf0,0x1c]      
vrcpss 485498096, %xmm1, %xmm1 

// CHECK: vrcpss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x53,0x4c,0x02,0x40]      
vrcpss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vrcpss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x53,0x0a]      
vrcpss (%edx), %xmm1, %xmm1 

// CHECK: vrcpss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x53,0xc9]      
vrcpss %xmm1, %xmm1, %xmm1 

// CHECK: vroundpd $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vroundpd $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vroundpd $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vroundpd $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vroundpd $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vroundpd $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vroundpd $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096(%edx), %xmm1 

// CHECK: vroundpd $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096(%edx), %ymm4 

// CHECK: vroundpd $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096, %xmm1 

// CHECK: vroundpd $0, 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096, %ymm4 

// CHECK: vroundpd $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x4c,0x02,0x40,0x00]      
vroundpd $0, 64(%edx,%eax), %xmm1 

// CHECK: vroundpd $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x64,0x02,0x40,0x00]      
vroundpd $0, 64(%edx,%eax), %ymm4 

// CHECK: vroundpd $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x0a,0x00]      
vroundpd $0, (%edx), %xmm1 

// CHECK: vroundpd $0, (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x22,0x00]      
vroundpd $0, (%edx), %ymm4 

// CHECK: vroundpd $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0xc9,0x00]      
vroundpd $0, %xmm1, %xmm1 

// CHECK: vroundpd $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0xe4,0x00]      
vroundpd $0, %ymm4, %ymm4 

// CHECK: vroundps $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vroundps $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: vroundps $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: vroundps $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vroundps $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vroundps $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vroundps $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096(%edx), %xmm1 

// CHECK: vroundps $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096(%edx), %ymm4 

// CHECK: vroundps $0, 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096, %xmm1 

// CHECK: vroundps $0, 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096, %ymm4 

// CHECK: vroundps $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x4c,0x02,0x40,0x00]      
vroundps $0, 64(%edx,%eax), %xmm1 

// CHECK: vroundps $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x64,0x02,0x40,0x00]      
vroundps $0, 64(%edx,%eax), %ymm4 

// CHECK: vroundps $0, (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x0a,0x00]      
vroundps $0, (%edx), %xmm1 

// CHECK: vroundps $0, (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x22,0x00]      
vroundps $0, (%edx), %ymm4 

// CHECK: vroundps $0, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0xc9,0x00]      
vroundps $0, %xmm1, %xmm1 

// CHECK: vroundps $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0xe4,0x00]      
vroundps $0, %ymm4, %ymm4 

// CHECK: vroundsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0b,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vroundsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vroundsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vroundsd $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0b,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundsd $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vroundsd $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0b,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundsd $0, 485498096, %xmm1, %xmm1 

// CHECK: vroundsd $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0b,0x4c,0x02,0x40,0x00]     
vroundsd $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vroundsd $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0b,0x0a,0x00]     
vroundsd $0, (%edx), %xmm1, %xmm1 

// CHECK: vroundsd $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0b,0xc9,0x00]     
vroundsd $0, %xmm1, %xmm1, %xmm1 

// CHECK: vroundss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0a,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vroundss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vroundss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vroundss $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0a,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundss $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vroundss $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0a,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundss $0, 485498096, %xmm1, %xmm1 

// CHECK: vroundss $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0a,0x4c,0x02,0x40,0x00]     
vroundss $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vroundss $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0a,0x0a,0x00]     
vroundss $0, (%edx), %xmm1, %xmm1 

// CHECK: vroundss $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x0a,0xc9,0x00]     
vroundss $0, %xmm1, %xmm1, %xmm1 

// CHECK: vrsqrtps -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x52,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vrsqrtps -485498096(%edx,%eax,4), %xmm1 

// CHECK: vrsqrtps 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x52,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096(%edx,%eax,4), %xmm1 

// CHECK: vrsqrtps -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x52,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vrsqrtps -485498096(%edx,%eax,4), %ymm4 

// CHECK: vrsqrtps 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x52,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096(%edx,%eax,4), %ymm4 

// CHECK: vrsqrtps 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x52,0x8a,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096(%edx), %xmm1 

// CHECK: vrsqrtps 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x52,0xa2,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096(%edx), %ymm4 

// CHECK: vrsqrtps 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x52,0x0d,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096, %xmm1 

// CHECK: vrsqrtps 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x52,0x25,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096, %ymm4 

// CHECK: vrsqrtps 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x52,0x4c,0x02,0x40]       
vrsqrtps 64(%edx,%eax), %xmm1 

// CHECK: vrsqrtps 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x52,0x64,0x02,0x40]       
vrsqrtps 64(%edx,%eax), %ymm4 

// CHECK: vrsqrtps (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x52,0x0a]       
vrsqrtps (%edx), %xmm1 

// CHECK: vrsqrtps (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x52,0x22]       
vrsqrtps (%edx), %ymm4 

// CHECK: vrsqrtps %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x52,0xc9]       
vrsqrtps %xmm1, %xmm1 

// CHECK: vrsqrtps %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x52,0xe4]       
vrsqrtps %ymm4, %ymm4 

// CHECK: vrsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x52,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vrsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vrsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x52,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vrsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vrsqrtss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x52,0x8a,0xf0,0x1c,0xf0,0x1c]      
vrsqrtss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vrsqrtss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x52,0x0d,0xf0,0x1c,0xf0,0x1c]      
vrsqrtss 485498096, %xmm1, %xmm1 

// CHECK: vrsqrtss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x52,0x4c,0x02,0x40]      
vrsqrtss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vrsqrtss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x52,0x0a]      
vrsqrtss (%edx), %xmm1, %xmm1 

// CHECK: vrsqrtss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x52,0xc9]      
vrsqrtss %xmm1, %xmm1, %xmm1 

// CHECK: vshufpd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc6,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vshufpd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vshufpd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vshufpd $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc6,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vshufpd $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vshufpd $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vshufpd $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc6,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vshufpd $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc6,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vshufpd $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc6,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096, %xmm1, %xmm1 

// CHECK: vshufpd $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc6,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096, %ymm4, %ymm4 

// CHECK: vshufpd $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc6,0x4c,0x02,0x40,0x00]     
vshufpd $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vshufpd $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc6,0x64,0x02,0x40,0x00]     
vshufpd $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vshufpd $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc6,0x0a,0x00]     
vshufpd $0, (%edx), %xmm1, %xmm1 

// CHECK: vshufpd $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc6,0x22,0x00]     
vshufpd $0, (%edx), %ymm4, %ymm4 

// CHECK: vshufpd $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0xc6,0xc9,0x00]     
vshufpd $0, %xmm1, %xmm1, %xmm1 

// CHECK: vshufpd $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xc6,0xe4,0x00]     
vshufpd $0, %ymm4, %ymm4, %ymm4 

// CHECK: vshufps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc6,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vshufps $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vshufps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vshufps $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc6,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vshufps $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vshufps $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vshufps $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc6,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vshufps $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc6,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vshufps $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc6,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096, %xmm1, %xmm1 

// CHECK: vshufps $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc6,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096, %ymm4, %ymm4 

// CHECK: vshufps $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc6,0x4c,0x02,0x40,0x00]     
vshufps $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vshufps $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc6,0x64,0x02,0x40,0x00]     
vshufps $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vshufps $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc6,0x0a,0x00]     
vshufps $0, (%edx), %xmm1, %xmm1 

// CHECK: vshufps $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc6,0x22,0x00]     
vshufps $0, (%edx), %ymm4, %ymm4 

// CHECK: vshufps $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0xc6,0xc9,0x00]     
vshufps $0, %xmm1, %xmm1, %xmm1 

// CHECK: vshufps $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0xc6,0xe4,0x00]     
vshufps $0, %ymm4, %ymm4, %ymm4 

// CHECK: vsqrtpd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vsqrtpd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vsqrtpd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vsqrtpd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x51,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vsqrtpd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vsqrtpd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x51,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vsqrtpd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096(%edx), %xmm1 

// CHECK: vsqrtpd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x51,0xa2,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096(%edx), %ymm4 

// CHECK: vsqrtpd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096, %xmm1 

// CHECK: vsqrtpd 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x51,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096, %ymm4 

// CHECK: vsqrtpd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x51,0x4c,0x02,0x40]       
vsqrtpd 64(%edx,%eax), %xmm1 

// CHECK: vsqrtpd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x51,0x64,0x02,0x40]       
vsqrtpd 64(%edx,%eax), %ymm4 

// CHECK: vsqrtpd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x51,0x0a]       
vsqrtpd (%edx), %xmm1 

// CHECK: vsqrtpd (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x51,0x22]       
vsqrtpd (%edx), %ymm4 

// CHECK: vsqrtpd %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x51,0xc9]       
vsqrtpd %xmm1, %xmm1 

// CHECK: vsqrtpd %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x51,0xe4]       
vsqrtpd %ymm4, %ymm4 

// CHECK: vsqrtps -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vsqrtps -485498096(%edx,%eax,4), %xmm1 

// CHECK: vsqrtps 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096(%edx,%eax,4), %xmm1 

// CHECK: vsqrtps -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x51,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vsqrtps -485498096(%edx,%eax,4), %ymm4 

// CHECK: vsqrtps 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x51,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096(%edx,%eax,4), %ymm4 

// CHECK: vsqrtps 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096(%edx), %xmm1 

// CHECK: vsqrtps 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x51,0xa2,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096(%edx), %ymm4 

// CHECK: vsqrtps 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096, %xmm1 

// CHECK: vsqrtps 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x51,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096, %ymm4 

// CHECK: vsqrtps 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x51,0x4c,0x02,0x40]       
vsqrtps 64(%edx,%eax), %xmm1 

// CHECK: vsqrtps 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x51,0x64,0x02,0x40]       
vsqrtps 64(%edx,%eax), %ymm4 

// CHECK: vsqrtps (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x51,0x0a]       
vsqrtps (%edx), %xmm1 

// CHECK: vsqrtps (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x51,0x22]       
vsqrtps (%edx), %ymm4 

// CHECK: vsqrtps %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x51,0xc9]       
vsqrtps %xmm1, %xmm1 

// CHECK: vsqrtps %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfc,0x51,0xe4]       
vsqrtps %ymm4, %ymm4 

// CHECK: vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsqrtsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]      
vsqrtsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vsqrtsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]      
vsqrtsd 485498096, %xmm1, %xmm1 

// CHECK: vsqrtsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x51,0x4c,0x02,0x40]      
vsqrtsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vsqrtsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x51,0x0a]      
vsqrtsd (%edx), %xmm1, %xmm1 

// CHECK: vsqrtsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x51,0xc9]      
vsqrtsd %xmm1, %xmm1, %xmm1 

// CHECK: vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsqrtss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]      
vsqrtss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vsqrtss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]      
vsqrtss 485498096, %xmm1, %xmm1 

// CHECK: vsqrtss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x51,0x4c,0x02,0x40]      
vsqrtss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vsqrtss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x51,0x0a]      
vsqrtss (%edx), %xmm1, %xmm1 

// CHECK: vsqrtss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x51,0xc9]      
vsqrtss %xmm1, %xmm1, %xmm1 

// CHECK: vstmxcsr -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
vstmxcsr -485498096(%edx,%eax,4) 

// CHECK: vstmxcsr 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
vstmxcsr 485498096(%edx,%eax,4) 

// CHECK: vstmxcsr 485498096(%edx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x9a,0xf0,0x1c,0xf0,0x1c]        
vstmxcsr 485498096(%edx) 

// CHECK: vstmxcsr 485498096 
// CHECK: encoding: [0xc5,0xf8,0xae,0x1d,0xf0,0x1c,0xf0,0x1c]        
vstmxcsr 485498096 

// CHECK: vstmxcsr 64(%edx,%eax) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x5c,0x02,0x40]        
vstmxcsr 64(%edx,%eax) 

// CHECK: vstmxcsr (%edx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x1a]        
vstmxcsr (%edx) 

// CHECK: vsubpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vsubpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vsubpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vsubpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vsubpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vsubpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vsubpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096, %xmm1, %xmm1 

// CHECK: vsubpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096, %ymm4, %ymm4 

// CHECK: vsubpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5c,0x4c,0x02,0x40]      
vsubpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vsubpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5c,0x64,0x02,0x40]      
vsubpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vsubpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5c,0x0a]      
vsubpd (%edx), %xmm1, %xmm1 

// CHECK: vsubpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5c,0x22]      
vsubpd (%edx), %ymm4, %ymm4 

// CHECK: vsubpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x5c,0xc9]      
vsubpd %xmm1, %xmm1, %xmm1 

// CHECK: vsubpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x5c,0xe4]      
vsubpd %ymm4, %ymm4, %ymm4 

// CHECK: vsubps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vsubps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vsubps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vsubps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vsubps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vsubps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vsubps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096, %xmm1, %xmm1 

// CHECK: vsubps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096, %ymm4, %ymm4 

// CHECK: vsubps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5c,0x4c,0x02,0x40]      
vsubps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vsubps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5c,0x64,0x02,0x40]      
vsubps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vsubps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5c,0x0a]      
vsubps (%edx), %xmm1, %xmm1 

// CHECK: vsubps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5c,0x22]      
vsubps (%edx), %ymm4, %ymm4 

// CHECK: vsubps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x5c,0xc9]      
vsubps %xmm1, %xmm1, %xmm1 

// CHECK: vsubps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x5c,0xe4]      
vsubps %ymm4, %ymm4, %ymm4 

// CHECK: vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubsd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vsubsd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vsubsd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vsubsd 485498096, %xmm1, %xmm1 

// CHECK: vsubsd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5c,0x4c,0x02,0x40]      
vsubsd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vsubsd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5c,0x0a]      
vsubsd (%edx), %xmm1, %xmm1 

// CHECK: vsubsd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf3,0x5c,0xc9]      
vsubsd %xmm1, %xmm1, %xmm1 

// CHECK: vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vsubss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vsubss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vsubss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vsubss 485498096, %xmm1, %xmm1 

// CHECK: vsubss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5c,0x4c,0x02,0x40]      
vsubss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vsubss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5c,0x0a]      
vsubss (%edx), %xmm1, %xmm1 

// CHECK: vsubss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf2,0x5c,0xc9]      
vsubss %xmm1, %xmm1, %xmm1 

// CHECK: vtestpd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vtestpd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vtestpd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vtestpd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vtestpd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vtestpd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vtestpd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x8a,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096(%edx), %xmm1 

// CHECK: vtestpd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0xa2,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096(%edx), %ymm4 

// CHECK: vtestpd 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x0d,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096, %xmm1 

// CHECK: vtestpd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096, %ymm4 

// CHECK: vtestpd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x4c,0x02,0x40]       
vtestpd 64(%edx,%eax), %xmm1 

// CHECK: vtestpd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x64,0x02,0x40]       
vtestpd 64(%edx,%eax), %ymm4 

// CHECK: vtestpd (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x0a]       
vtestpd (%edx), %xmm1 

// CHECK: vtestpd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x22]       
vtestpd (%edx), %ymm4 

// CHECK: vtestpd %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0xc9]       
vtestpd %xmm1, %xmm1 

// CHECK: vtestpd %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0xe4]       
vtestpd %ymm4, %ymm4 

// CHECK: vtestps -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vtestps -485498096(%edx,%eax,4), %xmm1 

// CHECK: vtestps 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096(%edx,%eax,4), %xmm1 

// CHECK: vtestps -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vtestps -485498096(%edx,%eax,4), %ymm4 

// CHECK: vtestps 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096(%edx,%eax,4), %ymm4 

// CHECK: vtestps 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x8a,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096(%edx), %xmm1 

// CHECK: vtestps 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0xa2,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096(%edx), %ymm4 

// CHECK: vtestps 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x0d,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096, %xmm1 

// CHECK: vtestps 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096, %ymm4 

// CHECK: vtestps 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x4c,0x02,0x40]       
vtestps 64(%edx,%eax), %xmm1 

// CHECK: vtestps 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x64,0x02,0x40]       
vtestps 64(%edx,%eax), %ymm4 

// CHECK: vtestps (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x0a]       
vtestps (%edx), %xmm1 

// CHECK: vtestps (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x22]       
vtestps (%edx), %ymm4 

// CHECK: vtestps %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0xc9]       
vtestps %xmm1, %xmm1 

// CHECK: vtestps %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0xe4]       
vtestps %ymm4, %ymm4 

// CHECK: vucomisd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vucomisd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vucomisd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vucomisd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vucomisd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x8a,0xf0,0x1c,0xf0,0x1c]       
vucomisd 485498096(%edx), %xmm1 

// CHECK: vucomisd 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x0d,0xf0,0x1c,0xf0,0x1c]       
vucomisd 485498096, %xmm1 

// CHECK: vucomisd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x4c,0x02,0x40]       
vucomisd 64(%edx,%eax), %xmm1 

// CHECK: vucomisd (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x0a]       
vucomisd (%edx), %xmm1 

// CHECK: vucomisd %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf9,0x2e,0xc9]       
vucomisd %xmm1, %xmm1 

// CHECK: vucomiss -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vucomiss -485498096(%edx,%eax,4), %xmm1 

// CHECK: vucomiss 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vucomiss 485498096(%edx,%eax,4), %xmm1 

// CHECK: vucomiss 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x8a,0xf0,0x1c,0xf0,0x1c]       
vucomiss 485498096(%edx), %xmm1 

// CHECK: vucomiss 485498096, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x0d,0xf0,0x1c,0xf0,0x1c]       
vucomiss 485498096, %xmm1 

// CHECK: vucomiss 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x4c,0x02,0x40]       
vucomiss 64(%edx,%eax), %xmm1 

// CHECK: vucomiss (%edx), %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x0a]       
vucomiss (%edx), %xmm1 

// CHECK: vucomiss %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf8,0x2e,0xc9]       
vucomiss %xmm1, %xmm1 

// CHECK: vunpckhpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x15,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vunpckhpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpckhpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x15,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpckhpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x15,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vunpckhpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpckhpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x15,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpckhpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x15,0x8a,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vunpckhpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x15,0xa2,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vunpckhpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x15,0x0d,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096, %xmm1, %xmm1 

// CHECK: vunpckhpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x15,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096, %ymm4, %ymm4 

// CHECK: vunpckhpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x15,0x4c,0x02,0x40]      
vunpckhpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vunpckhpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x15,0x64,0x02,0x40]      
vunpckhpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vunpckhpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x15,0x0a]      
vunpckhpd (%edx), %xmm1, %xmm1 

// CHECK: vunpckhpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x15,0x22]      
vunpckhpd (%edx), %ymm4, %ymm4 

// CHECK: vunpckhpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x15,0xc9]      
vunpckhpd %xmm1, %xmm1, %xmm1 

// CHECK: vunpckhpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x15,0xe4]      
vunpckhpd %ymm4, %ymm4, %ymm4 

// CHECK: vunpckhps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x15,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vunpckhps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpckhps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x15,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpckhps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x15,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vunpckhps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpckhps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x15,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpckhps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x15,0x8a,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vunpckhps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x15,0xa2,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vunpckhps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x15,0x0d,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096, %xmm1, %xmm1 

// CHECK: vunpckhps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x15,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096, %ymm4, %ymm4 

// CHECK: vunpckhps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x15,0x4c,0x02,0x40]      
vunpckhps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vunpckhps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x15,0x64,0x02,0x40]      
vunpckhps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vunpckhps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x15,0x0a]      
vunpckhps (%edx), %xmm1, %xmm1 

// CHECK: vunpckhps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x15,0x22]      
vunpckhps (%edx), %ymm4, %ymm4 

// CHECK: vunpckhps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x15,0xc9]      
vunpckhps %xmm1, %xmm1, %xmm1 

// CHECK: vunpckhps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x15,0xe4]      
vunpckhps %ymm4, %ymm4, %ymm4 

// CHECK: vunpcklpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x14,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vunpcklpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpcklpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x14,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpcklpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x14,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vunpcklpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpcklpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x14,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpcklpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x14,0x8a,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vunpcklpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x14,0xa2,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vunpcklpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x14,0x0d,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096, %xmm1, %xmm1 

// CHECK: vunpcklpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x14,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096, %ymm4, %ymm4 

// CHECK: vunpcklpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x14,0x4c,0x02,0x40]      
vunpcklpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vunpcklpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x14,0x64,0x02,0x40]      
vunpcklpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vunpcklpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x14,0x0a]      
vunpcklpd (%edx), %xmm1, %xmm1 

// CHECK: vunpcklpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x14,0x22]      
vunpcklpd (%edx), %ymm4, %ymm4 

// CHECK: vunpcklpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x14,0xc9]      
vunpcklpd %xmm1, %xmm1, %xmm1 

// CHECK: vunpcklpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x14,0xe4]      
vunpcklpd %ymm4, %ymm4, %ymm4 

// CHECK: vunpcklps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x14,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vunpcklps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpcklps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x14,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vunpcklps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x14,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vunpcklps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpcklps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x14,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vunpcklps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x14,0x8a,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vunpcklps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x14,0xa2,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vunpcklps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x14,0x0d,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096, %xmm1, %xmm1 

// CHECK: vunpcklps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x14,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096, %ymm4, %ymm4 

// CHECK: vunpcklps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x14,0x4c,0x02,0x40]      
vunpcklps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vunpcklps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x14,0x64,0x02,0x40]      
vunpcklps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vunpcklps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x14,0x0a]      
vunpcklps (%edx), %xmm1, %xmm1 

// CHECK: vunpcklps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x14,0x22]      
vunpcklps (%edx), %ymm4, %ymm4 

// CHECK: vunpcklps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x14,0xc9]      
vunpcklps %xmm1, %xmm1, %xmm1 

// CHECK: vunpcklps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x14,0xe4]      
vunpcklps %ymm4, %ymm4, %ymm4 

// CHECK: vxorpd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x57,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vxorpd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vxorpd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x57,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vxorpd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x57,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vxorpd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vxorpd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x57,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vxorpd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x57,0x8a,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vxorpd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x57,0xa2,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vxorpd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x57,0x0d,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096, %xmm1, %xmm1 

// CHECK: vxorpd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x57,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096, %ymm4, %ymm4 

// CHECK: vxorpd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x57,0x4c,0x02,0x40]      
vxorpd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vxorpd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x57,0x64,0x02,0x40]      
vxorpd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vxorpd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x57,0x0a]      
vxorpd (%edx), %xmm1, %xmm1 

// CHECK: vxorpd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x57,0x22]      
vxorpd (%edx), %ymm4, %ymm4 

// CHECK: vxorpd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf1,0x57,0xc9]      
vxorpd %xmm1, %xmm1, %xmm1 

// CHECK: vxorpd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x57,0xe4]      
vxorpd %ymm4, %ymm4, %ymm4 

// CHECK: vxorps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x57,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vxorps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vxorps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x57,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vxorps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x57,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vxorps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vxorps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x57,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vxorps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x57,0x8a,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vxorps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x57,0xa2,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vxorps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x57,0x0d,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096, %xmm1, %xmm1 

// CHECK: vxorps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x57,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096, %ymm4, %ymm4 

// CHECK: vxorps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x57,0x4c,0x02,0x40]      
vxorps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vxorps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x57,0x64,0x02,0x40]      
vxorps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vxorps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x57,0x0a]      
vxorps (%edx), %xmm1, %xmm1 

// CHECK: vxorps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x57,0x22]      
vxorps (%edx), %ymm4, %ymm4 

// CHECK: vxorps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc5,0xf0,0x57,0xc9]      
vxorps %xmm1, %xmm1, %xmm1 

// CHECK: vxorps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdc,0x57,0xe4]      
vxorps %ymm4, %ymm4, %ymm4 

// CHECK: vzeroall 
// CHECK: encoding: [0xc5,0xfc,0x77]         
vzeroall 

// CHECK: vzeroupper 
// CHECK: encoding: [0xc5,0xf8,0x77]         
vzeroupper 

