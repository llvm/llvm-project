// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vaddpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096, %xmm15, %xmm15 

// CHECK: vaddpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x58,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096, %xmm6, %xmm6 

// CHECK: vaddpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096, %ymm7, %ymm7 

// CHECK: vaddpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddpd 485498096, %ymm9, %ymm9 

// CHECK: vaddpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x58,0x7c,0x82,0xc0]      
vaddpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x58,0x7c,0x82,0x40]      
vaddpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x58,0x74,0x82,0xc0]      
vaddpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x58,0x74,0x82,0x40]      
vaddpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x58,0x7c,0x82,0xc0]      
vaddpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x58,0x7c,0x82,0x40]      
vaddpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x58,0x4c,0x82,0xc0]      
vaddpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x58,0x4c,0x82,0x40]      
vaddpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x58,0x7c,0x02,0x40]      
vaddpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaddpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x58,0x74,0x02,0x40]      
vaddpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaddpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x58,0x7c,0x02,0x40]      
vaddpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vaddpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x58,0x4c,0x02,0x40]      
vaddpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vaddpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x58,0x7a,0x40]      
vaddpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vaddpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x58,0x72,0x40]      
vaddpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vaddpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x58,0x7a,0x40]      
vaddpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vaddpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x58,0x4a,0x40]      
vaddpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vaddpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x58,0x3a]      
vaddpd (%rdx), %xmm15, %xmm15 

// CHECK: vaddpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x58,0x32]      
vaddpd (%rdx), %xmm6, %xmm6 

// CHECK: vaddpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x58,0x3a]      
vaddpd (%rdx), %ymm7, %ymm7 

// CHECK: vaddpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x58,0x0a]      
vaddpd (%rdx), %ymm9, %ymm9 

// CHECK: vaddpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x58,0xff]      
vaddpd %xmm15, %xmm15, %xmm15 

// CHECK: vaddpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x58,0xf6]      
vaddpd %xmm6, %xmm6, %xmm6 

// CHECK: vaddpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x58,0xff]      
vaddpd %ymm7, %ymm7, %ymm7 

// CHECK: vaddpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x58,0xc9]      
vaddpd %ymm9, %ymm9, %ymm9 

// CHECK: vaddps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096, %xmm15, %xmm15 

// CHECK: vaddps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x58,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096, %xmm6, %xmm6 

// CHECK: vaddps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096, %ymm7, %ymm7 

// CHECK: vaddps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddps 485498096, %ymm9, %ymm9 

// CHECK: vaddps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x58,0x7c,0x82,0xc0]      
vaddps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x58,0x7c,0x82,0x40]      
vaddps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x58,0x74,0x82,0xc0]      
vaddps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x58,0x74,0x82,0x40]      
vaddps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x58,0x7c,0x82,0xc0]      
vaddps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x58,0x7c,0x82,0x40]      
vaddps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x58,0x4c,0x82,0xc0]      
vaddps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x58,0x4c,0x82,0x40]      
vaddps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x58,0x7c,0x02,0x40]      
vaddps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaddps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x58,0x74,0x02,0x40]      
vaddps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaddps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x58,0x7c,0x02,0x40]      
vaddps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vaddps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x58,0x4c,0x02,0x40]      
vaddps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vaddps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x58,0x7a,0x40]      
vaddps 64(%rdx), %xmm15, %xmm15 

// CHECK: vaddps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x58,0x72,0x40]      
vaddps 64(%rdx), %xmm6, %xmm6 

// CHECK: vaddps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x58,0x7a,0x40]      
vaddps 64(%rdx), %ymm7, %ymm7 

// CHECK: vaddps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x58,0x4a,0x40]      
vaddps 64(%rdx), %ymm9, %ymm9 

// CHECK: vaddps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x58,0x3a]      
vaddps (%rdx), %xmm15, %xmm15 

// CHECK: vaddps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x58,0x32]      
vaddps (%rdx), %xmm6, %xmm6 

// CHECK: vaddps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x58,0x3a]      
vaddps (%rdx), %ymm7, %ymm7 

// CHECK: vaddps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x58,0x0a]      
vaddps (%rdx), %ymm9, %ymm9 

// CHECK: vaddps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x58,0xff]      
vaddps %xmm15, %xmm15, %xmm15 

// CHECK: vaddps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x58,0xf6]      
vaddps %xmm6, %xmm6, %xmm6 

// CHECK: vaddps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x58,0xff]      
vaddps %ymm7, %ymm7, %ymm7 

// CHECK: vaddps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x58,0xc9]      
vaddps %ymm9, %ymm9, %ymm9 

// CHECK: vaddsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsd 485498096, %xmm15, %xmm15 

// CHECK: vaddsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x58,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsd 485498096, %xmm6, %xmm6 

// CHECK: vaddsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x58,0x7c,0x82,0xc0]      
vaddsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x58,0x7c,0x82,0x40]      
vaddsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x58,0x74,0x82,0xc0]      
vaddsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x58,0x74,0x82,0x40]      
vaddsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x58,0x7c,0x02,0x40]      
vaddsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaddsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x58,0x74,0x02,0x40]      
vaddsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaddsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x58,0x7a,0x40]      
vaddsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vaddsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x58,0x72,0x40]      
vaddsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vaddsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x58,0x3a]      
vaddsd (%rdx), %xmm15, %xmm15 

// CHECK: vaddsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x58,0x32]      
vaddsd (%rdx), %xmm6, %xmm6 

// CHECK: vaddsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x58,0xff]      
vaddsd %xmm15, %xmm15, %xmm15 

// CHECK: vaddsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x58,0xf6]      
vaddsd %xmm6, %xmm6, %xmm6 

// CHECK: vaddss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddss 485498096, %xmm15, %xmm15 

// CHECK: vaddss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x58,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddss 485498096, %xmm6, %xmm6 

// CHECK: vaddss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x58,0x7c,0x82,0xc0]      
vaddss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x58,0x7c,0x82,0x40]      
vaddss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x58,0x74,0x82,0xc0]      
vaddss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x58,0x74,0x82,0x40]      
vaddss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x58,0x7c,0x02,0x40]      
vaddss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaddss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x58,0x74,0x02,0x40]      
vaddss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaddss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x58,0x7a,0x40]      
vaddss 64(%rdx), %xmm15, %xmm15 

// CHECK: vaddss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x58,0x72,0x40]      
vaddss 64(%rdx), %xmm6, %xmm6 

// CHECK: vaddss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x58,0x3a]      
vaddss (%rdx), %xmm15, %xmm15 

// CHECK: vaddss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x58,0x32]      
vaddss (%rdx), %xmm6, %xmm6 

// CHECK: vaddss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x58,0xff]      
vaddss %xmm15, %xmm15, %xmm15 

// CHECK: vaddss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x58,0xf6]      
vaddss %xmm6, %xmm6, %xmm6 

// CHECK: vaddsubpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096, %xmm15, %xmm15 

// CHECK: vaddsubpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096, %xmm6, %xmm6 

// CHECK: vaddsubpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096, %ymm7, %ymm7 

// CHECK: vaddsubpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd0,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubpd 485498096, %ymm9, %ymm9 

// CHECK: vaddsubpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd0,0x7c,0x82,0xc0]      
vaddsubpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddsubpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd0,0x7c,0x82,0x40]      
vaddsubpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddsubpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd0,0x74,0x82,0xc0]      
vaddsubpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddsubpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd0,0x74,0x82,0x40]      
vaddsubpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddsubpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd0,0x7c,0x82,0xc0]      
vaddsubpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddsubpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd0,0x7c,0x82,0x40]      
vaddsubpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddsubpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd0,0x4c,0x82,0xc0]      
vaddsubpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddsubpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd0,0x4c,0x82,0x40]      
vaddsubpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddsubpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd0,0x7c,0x02,0x40]      
vaddsubpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaddsubpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd0,0x74,0x02,0x40]      
vaddsubpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaddsubpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd0,0x7c,0x02,0x40]      
vaddsubpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vaddsubpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd0,0x4c,0x02,0x40]      
vaddsubpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vaddsubpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd0,0x7a,0x40]      
vaddsubpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vaddsubpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd0,0x72,0x40]      
vaddsubpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vaddsubpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd0,0x7a,0x40]      
vaddsubpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vaddsubpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd0,0x4a,0x40]      
vaddsubpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vaddsubpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd0,0x3a]      
vaddsubpd (%rdx), %xmm15, %xmm15 

// CHECK: vaddsubpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd0,0x32]      
vaddsubpd (%rdx), %xmm6, %xmm6 

// CHECK: vaddsubpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd0,0x3a]      
vaddsubpd (%rdx), %ymm7, %ymm7 

// CHECK: vaddsubpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd0,0x0a]      
vaddsubpd (%rdx), %ymm9, %ymm9 

// CHECK: vaddsubpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd0,0xff]      
vaddsubpd %xmm15, %xmm15, %xmm15 

// CHECK: vaddsubpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd0,0xf6]      
vaddsubpd %xmm6, %xmm6, %xmm6 

// CHECK: vaddsubpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd0,0xff]      
vaddsubpd %ymm7, %ymm7, %ymm7 

// CHECK: vaddsubpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd0,0xc9]      
vaddsubpd %ymm9, %ymm9, %ymm9 

// CHECK: vaddsubps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xd0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096, %xmm15, %xmm15 

// CHECK: vaddsubps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xd0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096, %xmm6, %xmm6 

// CHECK: vaddsubps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0xd0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096, %ymm7, %ymm7 

// CHECK: vaddsubps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0xd0,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaddsubps 485498096, %ymm9, %ymm9 

// CHECK: vaddsubps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xd0,0x7c,0x82,0xc0]      
vaddsubps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddsubps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xd0,0x7c,0x82,0x40]      
vaddsubps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaddsubps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xd0,0x74,0x82,0xc0]      
vaddsubps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddsubps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xd0,0x74,0x82,0x40]      
vaddsubps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaddsubps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0xd0,0x7c,0x82,0xc0]      
vaddsubps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddsubps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0xd0,0x7c,0x82,0x40]      
vaddsubps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vaddsubps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0xd0,0x4c,0x82,0xc0]      
vaddsubps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddsubps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0xd0,0x4c,0x82,0x40]      
vaddsubps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vaddsubps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xd0,0x7c,0x02,0x40]      
vaddsubps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaddsubps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xd0,0x74,0x02,0x40]      
vaddsubps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaddsubps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0xd0,0x7c,0x02,0x40]      
vaddsubps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vaddsubps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0xd0,0x4c,0x02,0x40]      
vaddsubps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vaddsubps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xd0,0x7a,0x40]      
vaddsubps 64(%rdx), %xmm15, %xmm15 

// CHECK: vaddsubps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xd0,0x72,0x40]      
vaddsubps 64(%rdx), %xmm6, %xmm6 

// CHECK: vaddsubps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0xd0,0x7a,0x40]      
vaddsubps 64(%rdx), %ymm7, %ymm7 

// CHECK: vaddsubps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0xd0,0x4a,0x40]      
vaddsubps 64(%rdx), %ymm9, %ymm9 

// CHECK: vaddsubps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xd0,0x3a]      
vaddsubps (%rdx), %xmm15, %xmm15 

// CHECK: vaddsubps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xd0,0x32]      
vaddsubps (%rdx), %xmm6, %xmm6 

// CHECK: vaddsubps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0xd0,0x3a]      
vaddsubps (%rdx), %ymm7, %ymm7 

// CHECK: vaddsubps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0xd0,0x0a]      
vaddsubps (%rdx), %ymm9, %ymm9 

// CHECK: vaddsubps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0xd0,0xff]      
vaddsubps %xmm15, %xmm15, %xmm15 

// CHECK: vaddsubps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xd0,0xf6]      
vaddsubps %xmm6, %xmm6, %xmm6 

// CHECK: vaddsubps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0xd0,0xff]      
vaddsubps %ymm7, %ymm7, %ymm7 

// CHECK: vaddsubps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x37,0xd0,0xc9]      
vaddsubps %ymm9, %ymm9, %ymm9 

// CHECK: vandnpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096, %xmm15, %xmm15 

// CHECK: vandnpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x55,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096, %xmm6, %xmm6 

// CHECK: vandnpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096, %ymm7, %ymm7 

// CHECK: vandnpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnpd 485498096, %ymm9, %ymm9 

// CHECK: vandnpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x55,0x7c,0x82,0xc0]      
vandnpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandnpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x55,0x7c,0x82,0x40]      
vandnpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandnpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x55,0x74,0x82,0xc0]      
vandnpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandnpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x55,0x74,0x82,0x40]      
vandnpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandnpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x55,0x7c,0x82,0xc0]      
vandnpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandnpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x55,0x7c,0x82,0x40]      
vandnpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandnpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x55,0x4c,0x82,0xc0]      
vandnpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandnpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x55,0x4c,0x82,0x40]      
vandnpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandnpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x55,0x7c,0x02,0x40]      
vandnpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vandnpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x55,0x74,0x02,0x40]      
vandnpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vandnpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x55,0x7c,0x02,0x40]      
vandnpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vandnpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x55,0x4c,0x02,0x40]      
vandnpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vandnpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x55,0x7a,0x40]      
vandnpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vandnpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x55,0x72,0x40]      
vandnpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vandnpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x55,0x7a,0x40]      
vandnpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vandnpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x55,0x4a,0x40]      
vandnpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vandnpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x55,0x3a]      
vandnpd (%rdx), %xmm15, %xmm15 

// CHECK: vandnpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x55,0x32]      
vandnpd (%rdx), %xmm6, %xmm6 

// CHECK: vandnpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x55,0x3a]      
vandnpd (%rdx), %ymm7, %ymm7 

// CHECK: vandnpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x55,0x0a]      
vandnpd (%rdx), %ymm9, %ymm9 

// CHECK: vandnpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x55,0xff]      
vandnpd %xmm15, %xmm15, %xmm15 

// CHECK: vandnpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x55,0xf6]      
vandnpd %xmm6, %xmm6, %xmm6 

// CHECK: vandnpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x55,0xff]      
vandnpd %ymm7, %ymm7, %ymm7 

// CHECK: vandnpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x55,0xc9]      
vandnpd %ymm9, %ymm9, %ymm9 

// CHECK: vandnps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096, %xmm15, %xmm15 

// CHECK: vandnps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x55,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096, %xmm6, %xmm6 

// CHECK: vandnps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096, %ymm7, %ymm7 

// CHECK: vandnps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandnps 485498096, %ymm9, %ymm9 

// CHECK: vandnps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x55,0x7c,0x82,0xc0]      
vandnps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandnps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x55,0x7c,0x82,0x40]      
vandnps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandnps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x55,0x74,0x82,0xc0]      
vandnps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandnps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x55,0x74,0x82,0x40]      
vandnps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandnps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x55,0x7c,0x82,0xc0]      
vandnps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandnps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x55,0x7c,0x82,0x40]      
vandnps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandnps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x55,0x4c,0x82,0xc0]      
vandnps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandnps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x55,0x4c,0x82,0x40]      
vandnps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandnps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x55,0x7c,0x02,0x40]      
vandnps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vandnps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x55,0x74,0x02,0x40]      
vandnps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vandnps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x55,0x7c,0x02,0x40]      
vandnps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vandnps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x55,0x4c,0x02,0x40]      
vandnps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vandnps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x55,0x7a,0x40]      
vandnps 64(%rdx), %xmm15, %xmm15 

// CHECK: vandnps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x55,0x72,0x40]      
vandnps 64(%rdx), %xmm6, %xmm6 

// CHECK: vandnps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x55,0x7a,0x40]      
vandnps 64(%rdx), %ymm7, %ymm7 

// CHECK: vandnps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x55,0x4a,0x40]      
vandnps 64(%rdx), %ymm9, %ymm9 

// CHECK: vandnps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x55,0x3a]      
vandnps (%rdx), %xmm15, %xmm15 

// CHECK: vandnps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x55,0x32]      
vandnps (%rdx), %xmm6, %xmm6 

// CHECK: vandnps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x55,0x3a]      
vandnps (%rdx), %ymm7, %ymm7 

// CHECK: vandnps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x55,0x0a]      
vandnps (%rdx), %ymm9, %ymm9 

// CHECK: vandnps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x55,0xff]      
vandnps %xmm15, %xmm15, %xmm15 

// CHECK: vandnps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x55,0xf6]      
vandnps %xmm6, %xmm6, %xmm6 

// CHECK: vandnps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x55,0xff]      
vandnps %ymm7, %ymm7, %ymm7 

// CHECK: vandnps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x55,0xc9]      
vandnps %ymm9, %ymm9, %ymm9 

// CHECK: vandpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x54,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096, %xmm15, %xmm15 

// CHECK: vandpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x54,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096, %xmm6, %xmm6 

// CHECK: vandpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x54,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096, %ymm7, %ymm7 

// CHECK: vandpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x54,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandpd 485498096, %ymm9, %ymm9 

// CHECK: vandpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x54,0x7c,0x82,0xc0]      
vandpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x54,0x7c,0x82,0x40]      
vandpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x54,0x74,0x82,0xc0]      
vandpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x54,0x74,0x82,0x40]      
vandpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x54,0x7c,0x82,0xc0]      
vandpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x54,0x7c,0x82,0x40]      
vandpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x54,0x4c,0x82,0xc0]      
vandpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x54,0x4c,0x82,0x40]      
vandpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x54,0x7c,0x02,0x40]      
vandpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vandpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x54,0x74,0x02,0x40]      
vandpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vandpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x54,0x7c,0x02,0x40]      
vandpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vandpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x54,0x4c,0x02,0x40]      
vandpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vandpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x54,0x7a,0x40]      
vandpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vandpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x54,0x72,0x40]      
vandpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vandpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x54,0x7a,0x40]      
vandpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vandpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x54,0x4a,0x40]      
vandpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vandpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x54,0x3a]      
vandpd (%rdx), %xmm15, %xmm15 

// CHECK: vandpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x54,0x32]      
vandpd (%rdx), %xmm6, %xmm6 

// CHECK: vandpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x54,0x3a]      
vandpd (%rdx), %ymm7, %ymm7 

// CHECK: vandpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x54,0x0a]      
vandpd (%rdx), %ymm9, %ymm9 

// CHECK: vandpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x54,0xff]      
vandpd %xmm15, %xmm15, %xmm15 

// CHECK: vandpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x54,0xf6]      
vandpd %xmm6, %xmm6, %xmm6 

// CHECK: vandpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x54,0xff]      
vandpd %ymm7, %ymm7, %ymm7 

// CHECK: vandpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x54,0xc9]      
vandpd %ymm9, %ymm9, %ymm9 

// CHECK: vandps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x54,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096, %xmm15, %xmm15 

// CHECK: vandps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x54,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096, %xmm6, %xmm6 

// CHECK: vandps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x54,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096, %ymm7, %ymm7 

// CHECK: vandps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x54,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vandps 485498096, %ymm9, %ymm9 

// CHECK: vandps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x54,0x7c,0x82,0xc0]      
vandps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x54,0x7c,0x82,0x40]      
vandps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vandps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x54,0x74,0x82,0xc0]      
vandps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x54,0x74,0x82,0x40]      
vandps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vandps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x54,0x7c,0x82,0xc0]      
vandps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x54,0x7c,0x82,0x40]      
vandps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vandps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x54,0x4c,0x82,0xc0]      
vandps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x54,0x4c,0x82,0x40]      
vandps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vandps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x54,0x7c,0x02,0x40]      
vandps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vandps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x54,0x74,0x02,0x40]      
vandps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vandps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x54,0x7c,0x02,0x40]      
vandps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vandps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x54,0x4c,0x02,0x40]      
vandps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vandps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x54,0x7a,0x40]      
vandps 64(%rdx), %xmm15, %xmm15 

// CHECK: vandps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x54,0x72,0x40]      
vandps 64(%rdx), %xmm6, %xmm6 

// CHECK: vandps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x54,0x7a,0x40]      
vandps 64(%rdx), %ymm7, %ymm7 

// CHECK: vandps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x54,0x4a,0x40]      
vandps 64(%rdx), %ymm9, %ymm9 

// CHECK: vandps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x54,0x3a]      
vandps (%rdx), %xmm15, %xmm15 

// CHECK: vandps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x54,0x32]      
vandps (%rdx), %xmm6, %xmm6 

// CHECK: vandps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x54,0x3a]      
vandps (%rdx), %ymm7, %ymm7 

// CHECK: vandps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x54,0x0a]      
vandps (%rdx), %ymm9, %ymm9 

// CHECK: vandps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x54,0xff]      
vandps %xmm15, %xmm15, %xmm15 

// CHECK: vandps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x54,0xf6]      
vandps %xmm6, %xmm6, %xmm6 

// CHECK: vandps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x54,0xff]      
vandps %ymm7, %ymm7, %ymm7 

// CHECK: vandps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x54,0xc9]      
vandps %ymm9, %ymm9, %ymm9 

// CHECK: vblendpd $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096, %xmm15, %xmm15 

// CHECK: vblendpd $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0d,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096, %xmm6, %xmm6 

// CHECK: vblendpd $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096, %ymm7, %ymm7 

// CHECK: vblendpd $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendpd $0, 485498096, %ymm9, %ymm9 

// CHECK: vblendpd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0d,0x7c,0x82,0xc0,0x00]     
vblendpd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendpd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0d,0x7c,0x82,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendpd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0d,0x74,0x82,0xc0,0x00]     
vblendpd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendpd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0d,0x74,0x82,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendpd $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0d,0x7c,0x82,0xc0,0x00]     
vblendpd $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendpd $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0d,0x7c,0x82,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendpd $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0d,0x4c,0x82,0xc0,0x00]     
vblendpd $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendpd $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0d,0x4c,0x82,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendpd $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0d,0x7c,0x02,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vblendpd $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0d,0x74,0x02,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vblendpd $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0d,0x7c,0x02,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vblendpd $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0d,0x4c,0x02,0x40,0x00]     
vblendpd $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vblendpd $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0d,0x7a,0x40,0x00]     
vblendpd $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vblendpd $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0d,0x72,0x40,0x00]     
vblendpd $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vblendpd $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0d,0x7a,0x40,0x00]     
vblendpd $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vblendpd $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0d,0x4a,0x40,0x00]     
vblendpd $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vblendpd $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0d,0x3a,0x00]     
vblendpd $0, (%rdx), %xmm15, %xmm15 

// CHECK: vblendpd $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0d,0x32,0x00]     
vblendpd $0, (%rdx), %xmm6, %xmm6 

// CHECK: vblendpd $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0d,0x3a,0x00]     
vblendpd $0, (%rdx), %ymm7, %ymm7 

// CHECK: vblendpd $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0d,0x0a,0x00]     
vblendpd $0, (%rdx), %ymm9, %ymm9 

// CHECK: vblendpd $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x0d,0xff,0x00]     
vblendpd $0, %xmm15, %xmm15, %xmm15 

// CHECK: vblendpd $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0d,0xf6,0x00]     
vblendpd $0, %xmm6, %xmm6, %xmm6 

// CHECK: vblendpd $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0d,0xff,0x00]     
vblendpd $0, %ymm7, %ymm7, %ymm7 

// CHECK: vblendpd $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x0d,0xc9,0x00]     
vblendpd $0, %ymm9, %ymm9, %ymm9 

// CHECK: vblendps $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096, %xmm15, %xmm15 

// CHECK: vblendps $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0c,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096, %xmm6, %xmm6 

// CHECK: vblendps $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096, %ymm7, %ymm7 

// CHECK: vblendps $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vblendps $0, 485498096, %ymm9, %ymm9 

// CHECK: vblendps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0c,0x7c,0x82,0xc0,0x00]     
vblendps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0c,0x7c,0x82,0x40,0x00]     
vblendps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0c,0x74,0x82,0xc0,0x00]     
vblendps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0c,0x74,0x82,0x40,0x00]     
vblendps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendps $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0c,0x7c,0x82,0xc0,0x00]     
vblendps $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendps $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0c,0x7c,0x82,0x40,0x00]     
vblendps $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendps $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0c,0x4c,0x82,0xc0,0x00]     
vblendps $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendps $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0c,0x4c,0x82,0x40,0x00]     
vblendps $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendps $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0c,0x7c,0x02,0x40,0x00]     
vblendps $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vblendps $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0c,0x74,0x02,0x40,0x00]     
vblendps $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vblendps $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0c,0x7c,0x02,0x40,0x00]     
vblendps $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vblendps $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0c,0x4c,0x02,0x40,0x00]     
vblendps $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vblendps $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0c,0x7a,0x40,0x00]     
vblendps $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vblendps $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0c,0x72,0x40,0x00]     
vblendps $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vblendps $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0c,0x7a,0x40,0x00]     
vblendps $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vblendps $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0c,0x4a,0x40,0x00]     
vblendps $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vblendps $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0c,0x3a,0x00]     
vblendps $0, (%rdx), %xmm15, %xmm15 

// CHECK: vblendps $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0c,0x32,0x00]     
vblendps $0, (%rdx), %xmm6, %xmm6 

// CHECK: vblendps $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0c,0x3a,0x00]     
vblendps $0, (%rdx), %ymm7, %ymm7 

// CHECK: vblendps $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0c,0x0a,0x00]     
vblendps $0, (%rdx), %ymm9, %ymm9 

// CHECK: vblendps $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x0c,0xff,0x00]     
vblendps $0, %xmm15, %xmm15, %xmm15 

// CHECK: vblendps $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0c,0xf6,0x00]     
vblendps $0, %xmm6, %xmm6, %xmm6 

// CHECK: vblendps $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0c,0xff,0x00]     
vblendps $0, %ymm7, %ymm7, %ymm7 

// CHECK: vblendps $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x0c,0xc9,0x00]     
vblendps $0, %ymm9, %ymm9, %ymm9 

// CHECK: vblendvpd %xmm15, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0xf0]     
vblendvpd %xmm15, 485498096, %xmm15, %xmm15 

// CHECK: vblendvpd %xmm15, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4b,0x7c,0x82,0xc0,0xf0]     
vblendvpd %xmm15, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendvpd %xmm15, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4b,0x7c,0x82,0x40,0xf0]     
vblendvpd %xmm15, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendvpd %xmm15, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4b,0x7c,0x02,0x40,0xf0]     
vblendvpd %xmm15, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vblendvpd %xmm15, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4b,0x7a,0x40,0xf0]     
vblendvpd %xmm15, 64(%rdx), %xmm15, %xmm15 

// CHECK: vblendvpd %xmm15, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4b,0x3a,0xf0]     
vblendvpd %xmm15, (%rdx), %xmm15, %xmm15 

// CHECK: vblendvpd %xmm15, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x4b,0xff,0xf0]     
vblendvpd %xmm15, %xmm15, %xmm15, %xmm15 

// CHECK: vblendvpd %xmm6, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4b,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x60]     
vblendvpd %xmm6, 485498096, %xmm6, %xmm6 

// CHECK: vblendvpd %xmm6, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4b,0x74,0x82,0xc0,0x60]     
vblendvpd %xmm6, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendvpd %xmm6, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4b,0x74,0x82,0x40,0x60]     
vblendvpd %xmm6, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendvpd %xmm6, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4b,0x74,0x02,0x40,0x60]     
vblendvpd %xmm6, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vblendvpd %xmm6, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4b,0x72,0x40,0x60]     
vblendvpd %xmm6, 64(%rdx), %xmm6, %xmm6 

// CHECK: vblendvpd %xmm6, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4b,0x32,0x60]     
vblendvpd %xmm6, (%rdx), %xmm6, %xmm6 

// CHECK: vblendvpd %xmm6, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4b,0xf6,0x60]     
vblendvpd %xmm6, %xmm6, %xmm6, %xmm6 

// CHECK: vblendvpd %ymm7, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x70]     
vblendvpd %ymm7, 485498096, %ymm7, %ymm7 

// CHECK: vblendvpd %ymm7, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4b,0x7c,0x82,0xc0,0x70]     
vblendvpd %ymm7, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendvpd %ymm7, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4b,0x7c,0x82,0x40,0x70]     
vblendvpd %ymm7, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendvpd %ymm7, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4b,0x7c,0x02,0x40,0x70]     
vblendvpd %ymm7, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vblendvpd %ymm7, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4b,0x7a,0x40,0x70]     
vblendvpd %ymm7, 64(%rdx), %ymm7, %ymm7 

// CHECK: vblendvpd %ymm7, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4b,0x3a,0x70]     
vblendvpd %ymm7, (%rdx), %ymm7, %ymm7 

// CHECK: vblendvpd %ymm7, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4b,0xff,0x70]     
vblendvpd %ymm7, %ymm7, %ymm7, %ymm7 

// CHECK: vblendvpd %ymm9, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x90]     
vblendvpd %ymm9, 485498096, %ymm9, %ymm9 

// CHECK: vblendvpd %ymm9, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4b,0x4c,0x82,0xc0,0x90]     
vblendvpd %ymm9, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendvpd %ymm9, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4b,0x4c,0x82,0x40,0x90]     
vblendvpd %ymm9, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendvpd %ymm9, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4b,0x4c,0x02,0x40,0x90]     
vblendvpd %ymm9, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vblendvpd %ymm9, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4b,0x4a,0x40,0x90]     
vblendvpd %ymm9, 64(%rdx), %ymm9, %ymm9 

// CHECK: vblendvpd %ymm9, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4b,0x0a,0x90]     
vblendvpd %ymm9, (%rdx), %ymm9, %ymm9 

// CHECK: vblendvpd %ymm9, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x4b,0xc9,0x90]     
vblendvpd %ymm9, %ymm9, %ymm9, %ymm9 

// CHECK: vblendvps %xmm15, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0xf0]     
vblendvps %xmm15, 485498096, %xmm15, %xmm15 

// CHECK: vblendvps %xmm15, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4a,0x7c,0x82,0xc0,0xf0]     
vblendvps %xmm15, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendvps %xmm15, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4a,0x7c,0x82,0x40,0xf0]     
vblendvps %xmm15, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vblendvps %xmm15, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4a,0x7c,0x02,0x40,0xf0]     
vblendvps %xmm15, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vblendvps %xmm15, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4a,0x7a,0x40,0xf0]     
vblendvps %xmm15, 64(%rdx), %xmm15, %xmm15 

// CHECK: vblendvps %xmm15, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4a,0x3a,0xf0]     
vblendvps %xmm15, (%rdx), %xmm15, %xmm15 

// CHECK: vblendvps %xmm15, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x4a,0xff,0xf0]     
vblendvps %xmm15, %xmm15, %xmm15, %xmm15 

// CHECK: vblendvps %xmm6, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4a,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x60]     
vblendvps %xmm6, 485498096, %xmm6, %xmm6 

// CHECK: vblendvps %xmm6, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4a,0x74,0x82,0xc0,0x60]     
vblendvps %xmm6, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendvps %xmm6, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4a,0x74,0x82,0x40,0x60]     
vblendvps %xmm6, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vblendvps %xmm6, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4a,0x74,0x02,0x40,0x60]     
vblendvps %xmm6, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vblendvps %xmm6, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4a,0x72,0x40,0x60]     
vblendvps %xmm6, 64(%rdx), %xmm6, %xmm6 

// CHECK: vblendvps %xmm6, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4a,0x32,0x60]     
vblendvps %xmm6, (%rdx), %xmm6, %xmm6 

// CHECK: vblendvps %xmm6, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4a,0xf6,0x60]     
vblendvps %xmm6, %xmm6, %xmm6, %xmm6 

// CHECK: vblendvps %ymm7, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x70]     
vblendvps %ymm7, 485498096, %ymm7, %ymm7 

// CHECK: vblendvps %ymm7, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4a,0x7c,0x82,0xc0,0x70]     
vblendvps %ymm7, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendvps %ymm7, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4a,0x7c,0x82,0x40,0x70]     
vblendvps %ymm7, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vblendvps %ymm7, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4a,0x7c,0x02,0x40,0x70]     
vblendvps %ymm7, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vblendvps %ymm7, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4a,0x7a,0x40,0x70]     
vblendvps %ymm7, 64(%rdx), %ymm7, %ymm7 

// CHECK: vblendvps %ymm7, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4a,0x3a,0x70]     
vblendvps %ymm7, (%rdx), %ymm7, %ymm7 

// CHECK: vblendvps %ymm7, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4a,0xff,0x70]     
vblendvps %ymm7, %ymm7, %ymm7, %ymm7 

// CHECK: vblendvps %ymm9, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x90]     
vblendvps %ymm9, 485498096, %ymm9, %ymm9 

// CHECK: vblendvps %ymm9, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4a,0x4c,0x82,0xc0,0x90]     
vblendvps %ymm9, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendvps %ymm9, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4a,0x4c,0x82,0x40,0x90]     
vblendvps %ymm9, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vblendvps %ymm9, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4a,0x4c,0x02,0x40,0x90]     
vblendvps %ymm9, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vblendvps %ymm9, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4a,0x4a,0x40,0x90]     
vblendvps %ymm9, 64(%rdx), %ymm9, %ymm9 

// CHECK: vblendvps %ymm9, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4a,0x0a,0x90]     
vblendvps %ymm9, (%rdx), %ymm9, %ymm9 

// CHECK: vblendvps %ymm9, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x4a,0xc9,0x90]     
vblendvps %ymm9, %ymm9, %ymm9, %ymm9 

// CHECK: vbroadcastf128 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastf128 485498096, %ymm7 

// CHECK: vbroadcastf128 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastf128 485498096, %ymm9 

// CHECK: vbroadcastf128 -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x7c,0x82,0xc0]       
vbroadcastf128 -64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcastf128 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x7c,0x82,0x40]       
vbroadcastf128 64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcastf128 -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1a,0x4c,0x82,0xc0]       
vbroadcastf128 -64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcastf128 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1a,0x4c,0x82,0x40]       
vbroadcastf128 64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcastf128 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x7c,0x02,0x40]       
vbroadcastf128 64(%rdx,%rax), %ymm7 

// CHECK: vbroadcastf128 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1a,0x4c,0x02,0x40]       
vbroadcastf128 64(%rdx,%rax), %ymm9 

// CHECK: vbroadcastf128 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x7a,0x40]       
vbroadcastf128 64(%rdx), %ymm7 

// CHECK: vbroadcastf128 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1a,0x4a,0x40]       
vbroadcastf128 64(%rdx), %ymm9 

// CHECK: vbroadcastf128 (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x3a]       
vbroadcastf128 (%rdx), %ymm7 

// CHECK: vbroadcastf128 (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1a,0x0a]       
vbroadcastf128 (%rdx), %ymm9 

// CHECK: vbroadcastsd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastsd 485498096, %ymm7 

// CHECK: vbroadcastsd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x19,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastsd 485498096, %ymm9 

// CHECK: vbroadcastsd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x7c,0x82,0xc0]       
vbroadcastsd -64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcastsd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x7c,0x82,0x40]       
vbroadcastsd 64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcastsd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x19,0x4c,0x82,0xc0]       
vbroadcastsd -64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcastsd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x19,0x4c,0x82,0x40]       
vbroadcastsd 64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcastsd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x7c,0x02,0x40]       
vbroadcastsd 64(%rdx,%rax), %ymm7 

// CHECK: vbroadcastsd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x19,0x4c,0x02,0x40]       
vbroadcastsd 64(%rdx,%rax), %ymm9 

// CHECK: vbroadcastsd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x7a,0x40]       
vbroadcastsd 64(%rdx), %ymm7 

// CHECK: vbroadcastsd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x19,0x4a,0x40]       
vbroadcastsd 64(%rdx), %ymm9 

// CHECK: vbroadcastsd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x3a]       
vbroadcastsd (%rdx), %ymm7 

// CHECK: vbroadcastsd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x19,0x0a]       
vbroadcastsd (%rdx), %ymm9 

// CHECK: vbroadcastss 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x18,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096, %xmm15 

// CHECK: vbroadcastss 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096, %xmm6 

// CHECK: vbroadcastss 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096, %ymm7 

// CHECK: vbroadcastss 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x18,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcastss 485498096, %ymm9 

// CHECK: vbroadcastss -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x18,0x7c,0x82,0xc0]       
vbroadcastss -64(%rdx,%rax,4), %xmm15 

// CHECK: vbroadcastss 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x18,0x7c,0x82,0x40]       
vbroadcastss 64(%rdx,%rax,4), %xmm15 

// CHECK: vbroadcastss -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x74,0x82,0xc0]       
vbroadcastss -64(%rdx,%rax,4), %xmm6 

// CHECK: vbroadcastss 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x74,0x82,0x40]       
vbroadcastss 64(%rdx,%rax,4), %xmm6 

// CHECK: vbroadcastss -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x7c,0x82,0xc0]       
vbroadcastss -64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcastss 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x7c,0x82,0x40]       
vbroadcastss 64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcastss -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x18,0x4c,0x82,0xc0]       
vbroadcastss -64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcastss 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x18,0x4c,0x82,0x40]       
vbroadcastss 64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcastss 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x18,0x7c,0x02,0x40]       
vbroadcastss 64(%rdx,%rax), %xmm15 

// CHECK: vbroadcastss 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x74,0x02,0x40]       
vbroadcastss 64(%rdx,%rax), %xmm6 

// CHECK: vbroadcastss 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x7c,0x02,0x40]       
vbroadcastss 64(%rdx,%rax), %ymm7 

// CHECK: vbroadcastss 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x18,0x4c,0x02,0x40]       
vbroadcastss 64(%rdx,%rax), %ymm9 

// CHECK: vbroadcastss 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x18,0x7a,0x40]       
vbroadcastss 64(%rdx), %xmm15 

// CHECK: vbroadcastss 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x72,0x40]       
vbroadcastss 64(%rdx), %xmm6 

// CHECK: vbroadcastss 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x7a,0x40]       
vbroadcastss 64(%rdx), %ymm7 

// CHECK: vbroadcastss 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x18,0x4a,0x40]       
vbroadcastss 64(%rdx), %ymm9 

// CHECK: vbroadcastss (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x18,0x3a]       
vbroadcastss (%rdx), %xmm15 

// CHECK: vbroadcastss (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x32]       
vbroadcastss (%rdx), %xmm6 

// CHECK: vbroadcastss (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x3a]       
vbroadcastss (%rdx), %ymm7 

// CHECK: vbroadcastss (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x18,0x0a]       
vbroadcastss (%rdx), %ymm9 

// CHECK: vcmpeqpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096, %xmm15, %xmm15 

// CHECK: vcmpeqpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc2,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096, %xmm6, %xmm6 

// CHECK: vcmpeqpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096, %ymm7, %ymm7 

// CHECK: vcmpeqpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc2,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqpd 485498096, %ymm9, %ymm9 

// CHECK: vcmpeqpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc2,0x7c,0x82,0xc0,0x00]      
vcmpeqpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc2,0x7c,0x82,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc2,0x74,0x82,0xc0,0x00]      
vcmpeqpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc2,0x74,0x82,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc2,0x7c,0x82,0xc0,0x00]      
vcmpeqpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vcmpeqpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc2,0x7c,0x82,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vcmpeqpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc2,0x4c,0x82,0xc0,0x00]      
vcmpeqpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vcmpeqpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc2,0x4c,0x82,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vcmpeqpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc2,0x7c,0x02,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcmpeqpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc2,0x74,0x02,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcmpeqpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc2,0x7c,0x02,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vcmpeqpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc2,0x4c,0x02,0x40,0x00]      
vcmpeqpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vcmpeqpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc2,0x7a,0x40,0x00]      
vcmpeqpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc2,0x72,0x40,0x00]      
vcmpeqpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc2,0x7a,0x40,0x00]      
vcmpeqpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vcmpeqpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc2,0x4a,0x40,0x00]      
vcmpeqpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vcmpeqpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc2,0x3a,0x00]      
vcmpeqpd (%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc2,0x32,0x00]      
vcmpeqpd (%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc2,0x3a,0x00]      
vcmpeqpd (%rdx), %ymm7, %ymm7 

// CHECK: vcmpeqpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc2,0x0a,0x00]      
vcmpeqpd (%rdx), %ymm9, %ymm9 

// CHECK: vcmpeqpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xc2,0xff,0x00]      
vcmpeqpd %xmm15, %xmm15, %xmm15 

// CHECK: vcmpeqpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc2,0xf6,0x00]      
vcmpeqpd %xmm6, %xmm6, %xmm6 

// CHECK: vcmpeqpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc2,0xff,0x00]      
vcmpeqpd %ymm7, %ymm7, %ymm7 

// CHECK: vcmpeqpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xc2,0xc9,0x00]      
vcmpeqpd %ymm9, %ymm9, %ymm9 

// CHECK: vcmpeqps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096, %xmm15, %xmm15 

// CHECK: vcmpeqps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc2,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096, %xmm6, %xmm6 

// CHECK: vcmpeqps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096, %ymm7, %ymm7 

// CHECK: vcmpeqps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc2,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqps 485498096, %ymm9, %ymm9 

// CHECK: vcmpeqps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc2,0x7c,0x82,0xc0,0x00]      
vcmpeqps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc2,0x7c,0x82,0x40,0x00]      
vcmpeqps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc2,0x74,0x82,0xc0,0x00]      
vcmpeqps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc2,0x74,0x82,0x40,0x00]      
vcmpeqps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc2,0x7c,0x82,0xc0,0x00]      
vcmpeqps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vcmpeqps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc2,0x7c,0x82,0x40,0x00]      
vcmpeqps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vcmpeqps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc2,0x4c,0x82,0xc0,0x00]      
vcmpeqps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vcmpeqps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc2,0x4c,0x82,0x40,0x00]      
vcmpeqps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vcmpeqps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc2,0x7c,0x02,0x40,0x00]      
vcmpeqps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcmpeqps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc2,0x74,0x02,0x40,0x00]      
vcmpeqps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcmpeqps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc2,0x7c,0x02,0x40,0x00]      
vcmpeqps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vcmpeqps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc2,0x4c,0x02,0x40,0x00]      
vcmpeqps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vcmpeqps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc2,0x7a,0x40,0x00]      
vcmpeqps 64(%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc2,0x72,0x40,0x00]      
vcmpeqps 64(%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc2,0x7a,0x40,0x00]      
vcmpeqps 64(%rdx), %ymm7, %ymm7 

// CHECK: vcmpeqps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc2,0x4a,0x40,0x00]      
vcmpeqps 64(%rdx), %ymm9, %ymm9 

// CHECK: vcmpeqps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc2,0x3a,0x00]      
vcmpeqps (%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc2,0x32,0x00]      
vcmpeqps (%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc2,0x3a,0x00]      
vcmpeqps (%rdx), %ymm7, %ymm7 

// CHECK: vcmpeqps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc2,0x0a,0x00]      
vcmpeqps (%rdx), %ymm9, %ymm9 

// CHECK: vcmpeqps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0xc2,0xff,0x00]      
vcmpeqps %xmm15, %xmm15, %xmm15 

// CHECK: vcmpeqps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc2,0xf6,0x00]      
vcmpeqps %xmm6, %xmm6, %xmm6 

// CHECK: vcmpeqps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc2,0xff,0x00]      
vcmpeqps %ymm7, %ymm7, %ymm7 

// CHECK: vcmpeqps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0xc2,0xc9,0x00]      
vcmpeqps %ymm9, %ymm9, %ymm9 

// CHECK: vcmpeqsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xc2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqsd 485498096, %xmm15, %xmm15 

// CHECK: vcmpeqsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xc2,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqsd 485498096, %xmm6, %xmm6 

// CHECK: vcmpeqsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xc2,0x7c,0x82,0xc0,0x00]      
vcmpeqsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xc2,0x7c,0x82,0x40,0x00]      
vcmpeqsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xc2,0x74,0x82,0xc0,0x00]      
vcmpeqsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xc2,0x74,0x82,0x40,0x00]      
vcmpeqsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xc2,0x7c,0x02,0x40,0x00]      
vcmpeqsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcmpeqsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xc2,0x74,0x02,0x40,0x00]      
vcmpeqsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcmpeqsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xc2,0x7a,0x40,0x00]      
vcmpeqsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xc2,0x72,0x40,0x00]      
vcmpeqsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0xc2,0x3a,0x00]      
vcmpeqsd (%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xc2,0x32,0x00]      
vcmpeqsd (%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0xc2,0xff,0x00]      
vcmpeqsd %xmm15, %xmm15, %xmm15 

// CHECK: vcmpeqsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0xc2,0xf6,0x00]      
vcmpeqsd %xmm6, %xmm6, %xmm6 

// CHECK: vcmpeqss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0xc2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqss 485498096, %xmm15, %xmm15 

// CHECK: vcmpeqss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0xc2,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vcmpeqss 485498096, %xmm6, %xmm6 

// CHECK: vcmpeqss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0xc2,0x7c,0x82,0xc0,0x00]      
vcmpeqss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0xc2,0x7c,0x82,0x40,0x00]      
vcmpeqss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcmpeqss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0xc2,0x74,0x82,0xc0,0x00]      
vcmpeqss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0xc2,0x74,0x82,0x40,0x00]      
vcmpeqss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcmpeqss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0xc2,0x7c,0x02,0x40,0x00]      
vcmpeqss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcmpeqss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0xc2,0x74,0x02,0x40,0x00]      
vcmpeqss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcmpeqss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0xc2,0x7a,0x40,0x00]      
vcmpeqss 64(%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0xc2,0x72,0x40,0x00]      
vcmpeqss 64(%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0xc2,0x3a,0x00]      
vcmpeqss (%rdx), %xmm15, %xmm15 

// CHECK: vcmpeqss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0xc2,0x32,0x00]      
vcmpeqss (%rdx), %xmm6, %xmm6 

// CHECK: vcmpeqss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0xc2,0xff,0x00]      
vcmpeqss %xmm15, %xmm15, %xmm15 

// CHECK: vcmpeqss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0xc2,0xf6,0x00]      
vcmpeqss %xmm6, %xmm6, %xmm6 

// CHECK: vcomisd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcomisd 485498096, %xmm15 

// CHECK: vcomisd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcomisd 485498096, %xmm6 

// CHECK: vcomisd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2f,0x7c,0x82,0xc0]       
vcomisd -64(%rdx,%rax,4), %xmm15 

// CHECK: vcomisd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2f,0x7c,0x82,0x40]       
vcomisd 64(%rdx,%rax,4), %xmm15 

// CHECK: vcomisd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x74,0x82,0xc0]       
vcomisd -64(%rdx,%rax,4), %xmm6 

// CHECK: vcomisd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x74,0x82,0x40]       
vcomisd 64(%rdx,%rax,4), %xmm6 

// CHECK: vcomisd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2f,0x7c,0x02,0x40]       
vcomisd 64(%rdx,%rax), %xmm15 

// CHECK: vcomisd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x74,0x02,0x40]       
vcomisd 64(%rdx,%rax), %xmm6 

// CHECK: vcomisd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2f,0x7a,0x40]       
vcomisd 64(%rdx), %xmm15 

// CHECK: vcomisd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x72,0x40]       
vcomisd 64(%rdx), %xmm6 

// CHECK: vcomisd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2f,0x3a]       
vcomisd (%rdx), %xmm15 

// CHECK: vcomisd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2f,0x32]       
vcomisd (%rdx), %xmm6 

// CHECK: vcomisd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x2f,0xff]       
vcomisd %xmm15, %xmm15 

// CHECK: vcomisd %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2f,0xf6]       
vcomisd %xmm6, %xmm6 

// CHECK: vcomiss 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcomiss 485498096, %xmm15 

// CHECK: vcomiss 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcomiss 485498096, %xmm6 

// CHECK: vcomiss -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2f,0x7c,0x82,0xc0]       
vcomiss -64(%rdx,%rax,4), %xmm15 

// CHECK: vcomiss 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2f,0x7c,0x82,0x40]       
vcomiss 64(%rdx,%rax,4), %xmm15 

// CHECK: vcomiss -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x74,0x82,0xc0]       
vcomiss -64(%rdx,%rax,4), %xmm6 

// CHECK: vcomiss 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x74,0x82,0x40]       
vcomiss 64(%rdx,%rax,4), %xmm6 

// CHECK: vcomiss 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2f,0x7c,0x02,0x40]       
vcomiss 64(%rdx,%rax), %xmm15 

// CHECK: vcomiss 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x74,0x02,0x40]       
vcomiss 64(%rdx,%rax), %xmm6 

// CHECK: vcomiss 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2f,0x7a,0x40]       
vcomiss 64(%rdx), %xmm15 

// CHECK: vcomiss 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x72,0x40]       
vcomiss 64(%rdx), %xmm6 

// CHECK: vcomiss (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2f,0x3a]       
vcomiss (%rdx), %xmm15 

// CHECK: vcomiss (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2f,0x32]       
vcomiss (%rdx), %xmm6 

// CHECK: vcomiss %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x2f,0xff]       
vcomiss %xmm15, %xmm15 

// CHECK: vcomiss %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2f,0xf6]       
vcomiss %xmm6, %xmm6 

// CHECK: vcvtdq2pd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0xe6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096, %xmm15 

// CHECK: vcvtdq2pd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096, %xmm6 

// CHECK: vcvtdq2pd 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096, %ymm7 

// CHECK: vcvtdq2pd 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7e,0xe6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2pd 485498096, %ymm9 

// CHECK: vcvtdq2pd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0xe6,0x7c,0x82,0xc0]       
vcvtdq2pd -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtdq2pd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0xe6,0x7c,0x82,0x40]       
vcvtdq2pd 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtdq2pd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x74,0x82,0xc0]       
vcvtdq2pd -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtdq2pd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x74,0x82,0x40]       
vcvtdq2pd 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtdq2pd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x7c,0x82,0xc0]       
vcvtdq2pd -64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtdq2pd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x7c,0x82,0x40]       
vcvtdq2pd 64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtdq2pd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0xe6,0x4c,0x82,0xc0]       
vcvtdq2pd -64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtdq2pd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0xe6,0x4c,0x82,0x40]       
vcvtdq2pd 64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtdq2pd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0xe6,0x7c,0x02,0x40]       
vcvtdq2pd 64(%rdx,%rax), %xmm15 

// CHECK: vcvtdq2pd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x74,0x02,0x40]       
vcvtdq2pd 64(%rdx,%rax), %xmm6 

// CHECK: vcvtdq2pd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x7c,0x02,0x40]       
vcvtdq2pd 64(%rdx,%rax), %ymm7 

// CHECK: vcvtdq2pd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0xe6,0x4c,0x02,0x40]       
vcvtdq2pd 64(%rdx,%rax), %ymm9 

// CHECK: vcvtdq2pd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0xe6,0x7a,0x40]       
vcvtdq2pd 64(%rdx), %xmm15 

// CHECK: vcvtdq2pd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x72,0x40]       
vcvtdq2pd 64(%rdx), %xmm6 

// CHECK: vcvtdq2pd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x7a,0x40]       
vcvtdq2pd 64(%rdx), %ymm7 

// CHECK: vcvtdq2pd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0xe6,0x4a,0x40]       
vcvtdq2pd 64(%rdx), %ymm9 

// CHECK: vcvtdq2pd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0xe6,0x3a]       
vcvtdq2pd (%rdx), %xmm15 

// CHECK: vcvtdq2pd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0xe6,0x32]       
vcvtdq2pd (%rdx), %xmm6 

// CHECK: vcvtdq2pd (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0xe6,0x3a]       
vcvtdq2pd (%rdx), %ymm7 

// CHECK: vcvtdq2pd (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0xe6,0x0a]       
vcvtdq2pd (%rdx), %ymm9 

// CHECK: vcvtdq2pd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7a,0xe6,0xff]       
vcvtdq2pd %xmm15, %xmm15 

// CHECK: vcvtdq2pd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7e,0xe6,0xcf]       
vcvtdq2pd %xmm15, %ymm9 

// CHECK: vcvtdq2pd %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0xe6,0xf6]       
vcvtdq2pd %xmm6, %xmm6 

// CHECK: vcvtdq2pd %xmm6, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0xe6,0xfe]       
vcvtdq2pd %xmm6, %ymm7 

// CHECK: vcvtdq2ps 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096, %xmm15 

// CHECK: vcvtdq2ps 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096, %xmm6 

// CHECK: vcvtdq2ps 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096, %ymm7 

// CHECK: vcvtdq2ps 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtdq2ps 485498096, %ymm9 

// CHECK: vcvtdq2ps -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5b,0x7c,0x82,0xc0]       
vcvtdq2ps -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtdq2ps 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5b,0x7c,0x82,0x40]       
vcvtdq2ps 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtdq2ps -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x74,0x82,0xc0]       
vcvtdq2ps -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtdq2ps 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x74,0x82,0x40]       
vcvtdq2ps 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtdq2ps -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x7c,0x82,0xc0]       
vcvtdq2ps -64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtdq2ps 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x7c,0x82,0x40]       
vcvtdq2ps 64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtdq2ps -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5b,0x4c,0x82,0xc0]       
vcvtdq2ps -64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtdq2ps 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5b,0x4c,0x82,0x40]       
vcvtdq2ps 64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtdq2ps 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5b,0x7c,0x02,0x40]       
vcvtdq2ps 64(%rdx,%rax), %xmm15 

// CHECK: vcvtdq2ps 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x74,0x02,0x40]       
vcvtdq2ps 64(%rdx,%rax), %xmm6 

// CHECK: vcvtdq2ps 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x7c,0x02,0x40]       
vcvtdq2ps 64(%rdx,%rax), %ymm7 

// CHECK: vcvtdq2ps 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5b,0x4c,0x02,0x40]       
vcvtdq2ps 64(%rdx,%rax), %ymm9 

// CHECK: vcvtdq2ps 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5b,0x7a,0x40]       
vcvtdq2ps 64(%rdx), %xmm15 

// CHECK: vcvtdq2ps 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x72,0x40]       
vcvtdq2ps 64(%rdx), %xmm6 

// CHECK: vcvtdq2ps 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x7a,0x40]       
vcvtdq2ps 64(%rdx), %ymm7 

// CHECK: vcvtdq2ps 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5b,0x4a,0x40]       
vcvtdq2ps 64(%rdx), %ymm9 

// CHECK: vcvtdq2ps (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5b,0x3a]       
vcvtdq2ps (%rdx), %xmm15 

// CHECK: vcvtdq2ps (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5b,0x32]       
vcvtdq2ps (%rdx), %xmm6 

// CHECK: vcvtdq2ps (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5b,0x3a]       
vcvtdq2ps (%rdx), %ymm7 

// CHECK: vcvtdq2ps (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5b,0x0a]       
vcvtdq2ps (%rdx), %ymm9 

// CHECK: vcvtdq2ps %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x5b,0xff]       
vcvtdq2ps %xmm15, %xmm15 

// CHECK: vcvtdq2ps %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5b,0xf6]       
vcvtdq2ps %xmm6, %xmm6 

// CHECK: vcvtdq2ps %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5b,0xff]       
vcvtdq2ps %ymm7, %ymm7 

// CHECK: vcvtdq2ps %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7c,0x5b,0xc9]       
vcvtdq2ps %ymm9, %ymm9 

// CHECK: vcvtpd2dqx 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xe6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqx 485498096, %xmm15 

// CHECK: vcvtpd2dqx 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqx 485498096, %xmm6 

// CHECK: vcvtpd2dqx -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xe6,0x7c,0x82,0xc0]       
vcvtpd2dqx -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2dqx 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xe6,0x7c,0x82,0x40]       
vcvtpd2dqx 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2dqx -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x74,0x82,0xc0]       
vcvtpd2dqx -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2dqx 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x74,0x82,0x40]       
vcvtpd2dqx 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2dqx 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xe6,0x7c,0x02,0x40]       
vcvtpd2dqx 64(%rdx,%rax), %xmm15 

// CHECK: vcvtpd2dqx 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x74,0x02,0x40]       
vcvtpd2dqx 64(%rdx,%rax), %xmm6 

// CHECK: vcvtpd2dqx 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xe6,0x7a,0x40]       
vcvtpd2dqx 64(%rdx), %xmm15 

// CHECK: vcvtpd2dqx 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x72,0x40]       
vcvtpd2dqx 64(%rdx), %xmm6 

// CHECK: vcvtpd2dq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7b,0xe6,0xff]       
vcvtpd2dq %xmm15, %xmm15 

// CHECK: vcvtpd2dq %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xe6,0xf6]       
vcvtpd2dq %xmm6, %xmm6 

// CHECK: vcvtpd2dqx (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xe6,0x3a]       
vcvtpd2dqx (%rdx), %xmm15 

// CHECK: vcvtpd2dqx (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xe6,0x32]       
vcvtpd2dqx (%rdx), %xmm6 

// CHECK: vcvtpd2dqy 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7f,0xe6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqy 485498096, %xmm15 

// CHECK: vcvtpd2dqy 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xff,0xe6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2dqy 485498096, %xmm6 

// CHECK: vcvtpd2dqy -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7f,0xe6,0x7c,0x82,0xc0]       
vcvtpd2dqy -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2dqy 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7f,0xe6,0x7c,0x82,0x40]       
vcvtpd2dqy 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2dqy -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xff,0xe6,0x74,0x82,0xc0]       
vcvtpd2dqy -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2dqy 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xff,0xe6,0x74,0x82,0x40]       
vcvtpd2dqy 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2dqy 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7f,0xe6,0x7c,0x02,0x40]       
vcvtpd2dqy 64(%rdx,%rax), %xmm15 

// CHECK: vcvtpd2dqy 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xff,0xe6,0x74,0x02,0x40]       
vcvtpd2dqy 64(%rdx,%rax), %xmm6 

// CHECK: vcvtpd2dqy 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7f,0xe6,0x7a,0x40]       
vcvtpd2dqy 64(%rdx), %xmm15 

// CHECK: vcvtpd2dqy 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xff,0xe6,0x72,0x40]       
vcvtpd2dqy 64(%rdx), %xmm6 

// CHECK: vcvtpd2dq %ymm7, %xmm6 
// CHECK: encoding: [0xc5,0xff,0xe6,0xf7]       
vcvtpd2dq %ymm7, %xmm6 

// CHECK: vcvtpd2dq %ymm9, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7f,0xe6,0xf9]       
vcvtpd2dq %ymm9, %xmm15 

// CHECK: vcvtpd2dqy (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7f,0xe6,0x3a]       
vcvtpd2dqy (%rdx), %xmm15 

// CHECK: vcvtpd2dqy (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xff,0xe6,0x32]       
vcvtpd2dqy (%rdx), %xmm6 

// CHECK: vcvtpd2psx 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psx 485498096, %xmm15 

// CHECK: vcvtpd2psx 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psx 485498096, %xmm6 

// CHECK: vcvtpd2psx -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5a,0x7c,0x82,0xc0]       
vcvtpd2psx -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2psx 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5a,0x7c,0x82,0x40]       
vcvtpd2psx 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2psx -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x74,0x82,0xc0]       
vcvtpd2psx -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2psx 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x74,0x82,0x40]       
vcvtpd2psx 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2psx 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5a,0x7c,0x02,0x40]       
vcvtpd2psx 64(%rdx,%rax), %xmm15 

// CHECK: vcvtpd2psx 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x74,0x02,0x40]       
vcvtpd2psx 64(%rdx,%rax), %xmm6 

// CHECK: vcvtpd2psx 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5a,0x7a,0x40]       
vcvtpd2psx 64(%rdx), %xmm15 

// CHECK: vcvtpd2psx 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x72,0x40]       
vcvtpd2psx 64(%rdx), %xmm6 

// CHECK: vcvtpd2ps %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x5a,0xff]       
vcvtpd2ps %xmm15, %xmm15 

// CHECK: vcvtpd2ps %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5a,0xf6]       
vcvtpd2ps %xmm6, %xmm6 

// CHECK: vcvtpd2psx (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5a,0x3a]       
vcvtpd2psx (%rdx), %xmm15 

// CHECK: vcvtpd2psx (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5a,0x32]       
vcvtpd2psx (%rdx), %xmm6 

// CHECK: vcvtpd2psy 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7d,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psy 485498096, %xmm15 

// CHECK: vcvtpd2psy 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtpd2psy 485498096, %xmm6 

// CHECK: vcvtpd2psy -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0x5a,0x7c,0x82,0xc0]       
vcvtpd2psy -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2psy 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0x5a,0x7c,0x82,0x40]       
vcvtpd2psy 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtpd2psy -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x74,0x82,0xc0]       
vcvtpd2psy -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2psy 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x74,0x82,0x40]       
vcvtpd2psy 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtpd2psy 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0x5a,0x7c,0x02,0x40]       
vcvtpd2psy 64(%rdx,%rax), %xmm15 

// CHECK: vcvtpd2psy 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x74,0x02,0x40]       
vcvtpd2psy 64(%rdx,%rax), %xmm6 

// CHECK: vcvtpd2psy 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0x5a,0x7a,0x40]       
vcvtpd2psy 64(%rdx), %xmm15 

// CHECK: vcvtpd2psy 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x72,0x40]       
vcvtpd2psy 64(%rdx), %xmm6 

// CHECK: vcvtpd2ps %ymm7, %xmm6 
// CHECK: encoding: [0xc5,0xfd,0x5a,0xf7]       
vcvtpd2ps %ymm7, %xmm6 

// CHECK: vcvtpd2ps %ymm9, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7d,0x5a,0xf9]       
vcvtpd2ps %ymm9, %xmm15 

// CHECK: vcvtpd2psy (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0x5a,0x3a]       
vcvtpd2psy (%rdx), %xmm15 

// CHECK: vcvtpd2psy (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0x5a,0x32]       
vcvtpd2psy (%rdx), %xmm6 

// CHECK: vcvtps2dq 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096, %xmm15 

// CHECK: vcvtps2dq 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096, %xmm6 

// CHECK: vcvtps2dq 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096, %ymm7 

// CHECK: vcvtps2dq 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x5b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2dq 485498096, %ymm9 

// CHECK: vcvtps2dq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5b,0x7c,0x82,0xc0]       
vcvtps2dq -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtps2dq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5b,0x7c,0x82,0x40]       
vcvtps2dq 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtps2dq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x74,0x82,0xc0]       
vcvtps2dq -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtps2dq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x74,0x82,0x40]       
vcvtps2dq 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtps2dq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x7c,0x82,0xc0]       
vcvtps2dq -64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtps2dq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x7c,0x82,0x40]       
vcvtps2dq 64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtps2dq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x5b,0x4c,0x82,0xc0]       
vcvtps2dq -64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtps2dq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x5b,0x4c,0x82,0x40]       
vcvtps2dq 64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtps2dq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5b,0x7c,0x02,0x40]       
vcvtps2dq 64(%rdx,%rax), %xmm15 

// CHECK: vcvtps2dq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x74,0x02,0x40]       
vcvtps2dq 64(%rdx,%rax), %xmm6 

// CHECK: vcvtps2dq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x7c,0x02,0x40]       
vcvtps2dq 64(%rdx,%rax), %ymm7 

// CHECK: vcvtps2dq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x5b,0x4c,0x02,0x40]       
vcvtps2dq 64(%rdx,%rax), %ymm9 

// CHECK: vcvtps2dq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5b,0x7a,0x40]       
vcvtps2dq 64(%rdx), %xmm15 

// CHECK: vcvtps2dq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x72,0x40]       
vcvtps2dq 64(%rdx), %xmm6 

// CHECK: vcvtps2dq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x7a,0x40]       
vcvtps2dq 64(%rdx), %ymm7 

// CHECK: vcvtps2dq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x5b,0x4a,0x40]       
vcvtps2dq 64(%rdx), %ymm9 

// CHECK: vcvtps2dq (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x5b,0x3a]       
vcvtps2dq (%rdx), %xmm15 

// CHECK: vcvtps2dq (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5b,0x32]       
vcvtps2dq (%rdx), %xmm6 

// CHECK: vcvtps2dq (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x5b,0x3a]       
vcvtps2dq (%rdx), %ymm7 

// CHECK: vcvtps2dq (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x5b,0x0a]       
vcvtps2dq (%rdx), %ymm9 

// CHECK: vcvtps2dq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x5b,0xff]       
vcvtps2dq %xmm15, %xmm15 

// CHECK: vcvtps2dq %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x5b,0xf6]       
vcvtps2dq %xmm6, %xmm6 

// CHECK: vcvtps2dq %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x5b,0xff]       
vcvtps2dq %ymm7, %ymm7 

// CHECK: vcvtps2dq %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7d,0x5b,0xc9]       
vcvtps2dq %ymm9, %ymm9 

// CHECK: vcvtps2pd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096, %xmm15 

// CHECK: vcvtps2pd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096, %xmm6 

// CHECK: vcvtps2pd 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096, %ymm7 

// CHECK: vcvtps2pd 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtps2pd 485498096, %ymm9 

// CHECK: vcvtps2pd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5a,0x7c,0x82,0xc0]       
vcvtps2pd -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtps2pd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5a,0x7c,0x82,0x40]       
vcvtps2pd 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvtps2pd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x74,0x82,0xc0]       
vcvtps2pd -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtps2pd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x74,0x82,0x40]       
vcvtps2pd 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvtps2pd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x7c,0x82,0xc0]       
vcvtps2pd -64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtps2pd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x7c,0x82,0x40]       
vcvtps2pd 64(%rdx,%rax,4), %ymm7 

// CHECK: vcvtps2pd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5a,0x4c,0x82,0xc0]       
vcvtps2pd -64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtps2pd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5a,0x4c,0x82,0x40]       
vcvtps2pd 64(%rdx,%rax,4), %ymm9 

// CHECK: vcvtps2pd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5a,0x7c,0x02,0x40]       
vcvtps2pd 64(%rdx,%rax), %xmm15 

// CHECK: vcvtps2pd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x74,0x02,0x40]       
vcvtps2pd 64(%rdx,%rax), %xmm6 

// CHECK: vcvtps2pd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x7c,0x02,0x40]       
vcvtps2pd 64(%rdx,%rax), %ymm7 

// CHECK: vcvtps2pd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5a,0x4c,0x02,0x40]       
vcvtps2pd 64(%rdx,%rax), %ymm9 

// CHECK: vcvtps2pd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5a,0x7a,0x40]       
vcvtps2pd 64(%rdx), %xmm15 

// CHECK: vcvtps2pd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x72,0x40]       
vcvtps2pd 64(%rdx), %xmm6 

// CHECK: vcvtps2pd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x7a,0x40]       
vcvtps2pd 64(%rdx), %ymm7 

// CHECK: vcvtps2pd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5a,0x4a,0x40]       
vcvtps2pd 64(%rdx), %ymm9 

// CHECK: vcvtps2pd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x5a,0x3a]       
vcvtps2pd (%rdx), %xmm15 

// CHECK: vcvtps2pd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5a,0x32]       
vcvtps2pd (%rdx), %xmm6 

// CHECK: vcvtps2pd (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5a,0x3a]       
vcvtps2pd (%rdx), %ymm7 

// CHECK: vcvtps2pd (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x5a,0x0a]       
vcvtps2pd (%rdx), %ymm9 

// CHECK: vcvtps2pd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x5a,0xff]       
vcvtps2pd %xmm15, %xmm15 

// CHECK: vcvtps2pd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7c,0x5a,0xcf]       
vcvtps2pd %xmm15, %ymm9 

// CHECK: vcvtps2pd %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x5a,0xf6]       
vcvtps2pd %xmm6, %xmm6 

// CHECK: vcvtps2pd %xmm6, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x5a,0xfe]       
vcvtps2pd %xmm6, %ymm7 

// CHECK: vcvtsd2si 485498096, %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2d,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtsd2si 485498096, %r13d 

// CHECK: vcvtsd2si 485498096, %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtsd2si 485498096, %r15 

// CHECK: vcvtsd2si 64(%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2d,0x6a,0x40]       
vcvtsd2si 64(%rdx), %r13d 

// CHECK: vcvtsd2si 64(%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x7a,0x40]       
vcvtsd2si 64(%rdx), %r15 

// CHECK: vcvtsd2si -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2d,0x6c,0x82,0xc0]       
vcvtsd2si -64(%rdx,%rax,4), %r13d 

// CHECK: vcvtsd2si 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2d,0x6c,0x82,0x40]       
vcvtsd2si 64(%rdx,%rax,4), %r13d 

// CHECK: vcvtsd2si -64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x7c,0x82,0xc0]       
vcvtsd2si -64(%rdx,%rax,4), %r15 

// CHECK: vcvtsd2si 64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x7c,0x82,0x40]       
vcvtsd2si 64(%rdx,%rax,4), %r15 

// CHECK: vcvtsd2si 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2d,0x6c,0x02,0x40]       
vcvtsd2si 64(%rdx,%rax), %r13d 

// CHECK: vcvtsd2si 64(%rdx,%rax), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x7c,0x02,0x40]       
vcvtsd2si 64(%rdx,%rax), %r15 

// CHECK: vcvtsd2si (%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2d,0x2a]       
vcvtsd2si (%rdx), %r13d 

// CHECK: vcvtsd2si (%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x3a]       
vcvtsd2si (%rdx), %r15 

// CHECK: vcvtsd2si %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x7b,0x2d,0xef]       
vcvtsd2si %xmm15, %r13d 

// CHECK: vcvtsd2si %xmm15, %r15 
// CHECK: encoding: [0xc4,0x41,0xfb,0x2d,0xff]       
vcvtsd2si %xmm15, %r15 

// CHECK: vcvtsd2si %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2d,0xee]       
vcvtsd2si %xmm6, %r13d 

// CHECK: vcvtsd2si %xmm6, %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0xfe]       
vcvtsd2si %xmm6, %r15 

// CHECK: vcvtsd2ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsd2ss 485498096, %xmm15, %xmm15 

// CHECK: vcvtsd2ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsd2ss 485498096, %xmm6, %xmm6 

// CHECK: vcvtsd2ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5a,0x7c,0x82,0xc0]      
vcvtsd2ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsd2ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5a,0x7c,0x82,0x40]      
vcvtsd2ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsd2ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5a,0x74,0x82,0xc0]      
vcvtsd2ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsd2ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5a,0x74,0x82,0x40]      
vcvtsd2ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsd2ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5a,0x7c,0x02,0x40]      
vcvtsd2ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcvtsd2ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5a,0x74,0x02,0x40]      
vcvtsd2ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcvtsd2ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5a,0x7a,0x40]      
vcvtsd2ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vcvtsd2ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5a,0x72,0x40]      
vcvtsd2ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vcvtsd2ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5a,0x3a]      
vcvtsd2ss (%rdx), %xmm15, %xmm15 

// CHECK: vcvtsd2ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5a,0x32]      
vcvtsd2ss (%rdx), %xmm6, %xmm6 

// CHECK: vcvtsd2ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x5a,0xff]      
vcvtsd2ss %xmm15, %xmm15, %xmm15 

// CHECK: vcvtsd2ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5a,0xf6]      
vcvtsd2ss %xmm6, %xmm6, %xmm6 

// CHECK: vcvtsi2sdl 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2sdl 485498096, %xmm15, %xmm15 

// CHECK: vcvtsi2sdl 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2sdl 485498096, %xmm6, %xmm6 

// CHECK: vcvtsi2sdl -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x2a,0x7c,0x82,0xc0]      
vcvtsi2sdl -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2sdl 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x2a,0x7c,0x82,0x40]      
vcvtsi2sdl 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2sdl -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x2a,0x74,0x82,0xc0]      
vcvtsi2sdl -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2sdl 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x2a,0x74,0x82,0x40]      
vcvtsi2sdl 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2sdl 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x2a,0x7c,0x02,0x40]      
vcvtsi2sdl 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcvtsi2sdl 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x2a,0x74,0x02,0x40]      
vcvtsi2sdl 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcvtsi2sdl 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x2a,0x7a,0x40]      
vcvtsi2sdl 64(%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2sdl 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x2a,0x72,0x40]      
vcvtsi2sdl 64(%rdx), %xmm6, %xmm6 

// CHECK: vcvtsi2sdl %r13d, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x2a,0xfd]      
vcvtsi2sdl %r13d, %xmm15, %xmm15 

// CHECK: vcvtsi2sdl %r13d, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc1,0x4b,0x2a,0xf5]      
vcvtsi2sdl %r13d, %xmm6, %xmm6 

// CHECK: vcvtsi2sdl (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x2a,0x3a]      
vcvtsi2sdl (%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2sdl (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x2a,0x32]      
vcvtsi2sdl (%rdx), %xmm6, %xmm6 

// CHECK: vcvtsi2sdq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2sdq 485498096, %xmm15, %xmm15 

// CHECK: vcvtsi2sdq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xcb,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2sdq 485498096, %xmm6, %xmm6 

// CHECK: vcvtsi2sdq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x7c,0x82,0xc0]      
vcvtsi2sdq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2sdq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x7c,0x82,0x40]      
vcvtsi2sdq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2sdq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xcb,0x2a,0x74,0x82,0xc0]      
vcvtsi2sdq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2sdq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xcb,0x2a,0x74,0x82,0x40]      
vcvtsi2sdq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2sdq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x7c,0x02,0x40]      
vcvtsi2sdq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcvtsi2sdq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xcb,0x2a,0x74,0x02,0x40]      
vcvtsi2sdq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcvtsi2sdq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x7a,0x40]      
vcvtsi2sdq 64(%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2sdq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xcb,0x2a,0x72,0x40]      
vcvtsi2sdq 64(%rdx), %xmm6, %xmm6 

// CHECK: vcvtsi2sdq %r15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x83,0x2a,0xff]      
vcvtsi2sdq %r15, %xmm15, %xmm15 

// CHECK: vcvtsi2sdq %r15, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc1,0xcb,0x2a,0xf7]      
vcvtsi2sdq %r15, %xmm6, %xmm6 

// CHECK: vcvtsi2sdq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x3a]      
vcvtsi2sdq (%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2sdq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xcb,0x2a,0x32]      
vcvtsi2sdq (%rdx), %xmm6, %xmm6 

// CHECK: vcvtsi2ssl 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2ssl 485498096, %xmm15, %xmm15 

// CHECK: vcvtsi2ssl 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2ssl 485498096, %xmm6, %xmm6 

// CHECK: vcvtsi2ssl -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x2a,0x7c,0x82,0xc0]      
vcvtsi2ssl -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2ssl 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x2a,0x7c,0x82,0x40]      
vcvtsi2ssl 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2ssl -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x2a,0x74,0x82,0xc0]      
vcvtsi2ssl -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2ssl 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x2a,0x74,0x82,0x40]      
vcvtsi2ssl 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2ssl 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x2a,0x7c,0x02,0x40]      
vcvtsi2ssl 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcvtsi2ssl 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x2a,0x74,0x02,0x40]      
vcvtsi2ssl 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcvtsi2ssl 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x2a,0x7a,0x40]      
vcvtsi2ssl 64(%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2ssl 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x2a,0x72,0x40]      
vcvtsi2ssl 64(%rdx), %xmm6, %xmm6 

// CHECK: vcvtsi2ssl %r13d, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x2a,0xfd]      
vcvtsi2ssl %r13d, %xmm15, %xmm15 

// CHECK: vcvtsi2ssl %r13d, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc1,0x4a,0x2a,0xf5]      
vcvtsi2ssl %r13d, %xmm6, %xmm6 

// CHECK: vcvtsi2ssl (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x2a,0x3a]      
vcvtsi2ssl (%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2ssl (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x2a,0x32]      
vcvtsi2ssl (%rdx), %xmm6, %xmm6 

// CHECK: vcvtsi2ssq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2ssq 485498096, %xmm15, %xmm15 

// CHECK: vcvtsi2ssq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xca,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtsi2ssq 485498096, %xmm6, %xmm6 

// CHECK: vcvtsi2ssq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x7c,0x82,0xc0]      
vcvtsi2ssq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2ssq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x7c,0x82,0x40]      
vcvtsi2ssq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtsi2ssq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xca,0x2a,0x74,0x82,0xc0]      
vcvtsi2ssq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2ssq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xca,0x2a,0x74,0x82,0x40]      
vcvtsi2ssq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtsi2ssq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x7c,0x02,0x40]      
vcvtsi2ssq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcvtsi2ssq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xca,0x2a,0x74,0x02,0x40]      
vcvtsi2ssq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcvtsi2ssq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x7a,0x40]      
vcvtsi2ssq 64(%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2ssq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xca,0x2a,0x72,0x40]      
vcvtsi2ssq 64(%rdx), %xmm6, %xmm6 

// CHECK: vcvtsi2ssq %r15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x82,0x2a,0xff]      
vcvtsi2ssq %r15, %xmm15, %xmm15 

// CHECK: vcvtsi2ssq %r15, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc1,0xca,0x2a,0xf7]      
vcvtsi2ssq %r15, %xmm6, %xmm6 

// CHECK: vcvtsi2ssq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x3a]      
vcvtsi2ssq (%rdx), %xmm15, %xmm15 

// CHECK: vcvtsi2ssq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe1,0xca,0x2a,0x32]      
vcvtsi2ssq (%rdx), %xmm6, %xmm6 

// CHECK: vcvtss2sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtss2sd 485498096, %xmm15, %xmm15 

// CHECK: vcvtss2sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vcvtss2sd 485498096, %xmm6, %xmm6 

// CHECK: vcvtss2sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5a,0x7c,0x82,0xc0]      
vcvtss2sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtss2sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5a,0x7c,0x82,0x40]      
vcvtss2sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vcvtss2sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5a,0x74,0x82,0xc0]      
vcvtss2sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtss2sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5a,0x74,0x82,0x40]      
vcvtss2sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vcvtss2sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5a,0x7c,0x02,0x40]      
vcvtss2sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vcvtss2sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5a,0x74,0x02,0x40]      
vcvtss2sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vcvtss2sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5a,0x7a,0x40]      
vcvtss2sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vcvtss2sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5a,0x72,0x40]      
vcvtss2sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vcvtss2sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5a,0x3a]      
vcvtss2sd (%rdx), %xmm15, %xmm15 

// CHECK: vcvtss2sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5a,0x32]      
vcvtss2sd (%rdx), %xmm6, %xmm6 

// CHECK: vcvtss2sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x5a,0xff]      
vcvtss2sd %xmm15, %xmm15, %xmm15 

// CHECK: vcvtss2sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5a,0xf6]      
vcvtss2sd %xmm6, %xmm6, %xmm6 

// CHECK: vcvtss2si 485498096, %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2d,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtss2si 485498096, %r13d 

// CHECK: vcvtss2si 485498096, %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvtss2si 485498096, %r15 

// CHECK: vcvtss2si 64(%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2d,0x6a,0x40]       
vcvtss2si 64(%rdx), %r13d 

// CHECK: vcvtss2si 64(%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x7a,0x40]       
vcvtss2si 64(%rdx), %r15 

// CHECK: vcvtss2si -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2d,0x6c,0x82,0xc0]       
vcvtss2si -64(%rdx,%rax,4), %r13d 

// CHECK: vcvtss2si 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2d,0x6c,0x82,0x40]       
vcvtss2si 64(%rdx,%rax,4), %r13d 

// CHECK: vcvtss2si -64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x7c,0x82,0xc0]       
vcvtss2si -64(%rdx,%rax,4), %r15 

// CHECK: vcvtss2si 64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x7c,0x82,0x40]       
vcvtss2si 64(%rdx,%rax,4), %r15 

// CHECK: vcvtss2si 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2d,0x6c,0x02,0x40]       
vcvtss2si 64(%rdx,%rax), %r13d 

// CHECK: vcvtss2si 64(%rdx,%rax), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x7c,0x02,0x40]       
vcvtss2si 64(%rdx,%rax), %r15 

// CHECK: vcvtss2si (%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2d,0x2a]       
vcvtss2si (%rdx), %r13d 

// CHECK: vcvtss2si (%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x3a]       
vcvtss2si (%rdx), %r15 

// CHECK: vcvtss2si %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x7a,0x2d,0xef]       
vcvtss2si %xmm15, %r13d 

// CHECK: vcvtss2si %xmm15, %r15 
// CHECK: encoding: [0xc4,0x41,0xfa,0x2d,0xff]       
vcvtss2si %xmm15, %r15 

// CHECK: vcvtss2si %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2d,0xee]       
vcvtss2si %xmm6, %r13d 

// CHECK: vcvtss2si %xmm6, %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0xfe]       
vcvtss2si %xmm6, %r15 

// CHECK: vcvttpd2dqx 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0xe6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqx 485498096, %xmm15 

// CHECK: vcvttpd2dqx 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqx 485498096, %xmm6 

// CHECK: vcvttpd2dqx -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0xe6,0x7c,0x82,0xc0]       
vcvttpd2dqx -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvttpd2dqx 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0xe6,0x7c,0x82,0x40]       
vcvttpd2dqx 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvttpd2dqx -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x74,0x82,0xc0]       
vcvttpd2dqx -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvttpd2dqx 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x74,0x82,0x40]       
vcvttpd2dqx 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvttpd2dqx 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0xe6,0x7c,0x02,0x40]       
vcvttpd2dqx 64(%rdx,%rax), %xmm15 

// CHECK: vcvttpd2dqx 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x74,0x02,0x40]       
vcvttpd2dqx 64(%rdx,%rax), %xmm6 

// CHECK: vcvttpd2dqx 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0xe6,0x7a,0x40]       
vcvttpd2dqx 64(%rdx), %xmm15 

// CHECK: vcvttpd2dqx 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x72,0x40]       
vcvttpd2dqx 64(%rdx), %xmm6 

// CHECK: vcvttpd2dq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0xe6,0xff]       
vcvttpd2dq %xmm15, %xmm15 

// CHECK: vcvttpd2dq %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xe6,0xf6]       
vcvttpd2dq %xmm6, %xmm6 

// CHECK: vcvttpd2dqx (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0xe6,0x3a]       
vcvttpd2dqx (%rdx), %xmm15 

// CHECK: vcvttpd2dqx (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xe6,0x32]       
vcvttpd2dqx (%rdx), %xmm6 

// CHECK: vcvttpd2dqy 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7d,0xe6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqy 485498096, %xmm15 

// CHECK: vcvttpd2dqy 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttpd2dqy 485498096, %xmm6 

// CHECK: vcvttpd2dqy -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0xe6,0x7c,0x82,0xc0]       
vcvttpd2dqy -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvttpd2dqy 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0xe6,0x7c,0x82,0x40]       
vcvttpd2dqy 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvttpd2dqy -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x74,0x82,0xc0]       
vcvttpd2dqy -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvttpd2dqy 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x74,0x82,0x40]       
vcvttpd2dqy 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvttpd2dqy 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0xe6,0x7c,0x02,0x40]       
vcvttpd2dqy 64(%rdx,%rax), %xmm15 

// CHECK: vcvttpd2dqy 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x74,0x02,0x40]       
vcvttpd2dqy 64(%rdx,%rax), %xmm6 

// CHECK: vcvttpd2dqy 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0xe6,0x7a,0x40]       
vcvttpd2dqy 64(%rdx), %xmm15 

// CHECK: vcvttpd2dqy 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x72,0x40]       
vcvttpd2dqy 64(%rdx), %xmm6 

// CHECK: vcvttpd2dq %ymm7, %xmm6 
// CHECK: encoding: [0xc5,0xfd,0xe6,0xf7]       
vcvttpd2dq %ymm7, %xmm6 

// CHECK: vcvttpd2dq %ymm9, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7d,0xe6,0xf9]       
vcvttpd2dq %ymm9, %xmm15 

// CHECK: vcvttpd2dqy (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7d,0xe6,0x3a]       
vcvttpd2dqy (%rdx), %xmm15 

// CHECK: vcvttpd2dqy (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfd,0xe6,0x32]       
vcvttpd2dqy (%rdx), %xmm6 

// CHECK: vcvttps2dq 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x5b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096, %xmm15 

// CHECK: vcvttps2dq 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096, %xmm6 

// CHECK: vcvttps2dq 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096, %ymm7 

// CHECK: vcvttps2dq 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x5b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttps2dq 485498096, %ymm9 

// CHECK: vcvttps2dq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x5b,0x7c,0x82,0xc0]       
vcvttps2dq -64(%rdx,%rax,4), %xmm15 

// CHECK: vcvttps2dq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x5b,0x7c,0x82,0x40]       
vcvttps2dq 64(%rdx,%rax,4), %xmm15 

// CHECK: vcvttps2dq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x74,0x82,0xc0]       
vcvttps2dq -64(%rdx,%rax,4), %xmm6 

// CHECK: vcvttps2dq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x74,0x82,0x40]       
vcvttps2dq 64(%rdx,%rax,4), %xmm6 

// CHECK: vcvttps2dq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x7c,0x82,0xc0]       
vcvttps2dq -64(%rdx,%rax,4), %ymm7 

// CHECK: vcvttps2dq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x7c,0x82,0x40]       
vcvttps2dq 64(%rdx,%rax,4), %ymm7 

// CHECK: vcvttps2dq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x5b,0x4c,0x82,0xc0]       
vcvttps2dq -64(%rdx,%rax,4), %ymm9 

// CHECK: vcvttps2dq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x5b,0x4c,0x82,0x40]       
vcvttps2dq 64(%rdx,%rax,4), %ymm9 

// CHECK: vcvttps2dq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x5b,0x7c,0x02,0x40]       
vcvttps2dq 64(%rdx,%rax), %xmm15 

// CHECK: vcvttps2dq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x74,0x02,0x40]       
vcvttps2dq 64(%rdx,%rax), %xmm6 

// CHECK: vcvttps2dq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x7c,0x02,0x40]       
vcvttps2dq 64(%rdx,%rax), %ymm7 

// CHECK: vcvttps2dq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x5b,0x4c,0x02,0x40]       
vcvttps2dq 64(%rdx,%rax), %ymm9 

// CHECK: vcvttps2dq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x5b,0x7a,0x40]       
vcvttps2dq 64(%rdx), %xmm15 

// CHECK: vcvttps2dq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x72,0x40]       
vcvttps2dq 64(%rdx), %xmm6 

// CHECK: vcvttps2dq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x7a,0x40]       
vcvttps2dq 64(%rdx), %ymm7 

// CHECK: vcvttps2dq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x5b,0x4a,0x40]       
vcvttps2dq 64(%rdx), %ymm9 

// CHECK: vcvttps2dq (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x5b,0x3a]       
vcvttps2dq (%rdx), %xmm15 

// CHECK: vcvttps2dq (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x5b,0x32]       
vcvttps2dq (%rdx), %xmm6 

// CHECK: vcvttps2dq (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x5b,0x3a]       
vcvttps2dq (%rdx), %ymm7 

// CHECK: vcvttps2dq (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x5b,0x0a]       
vcvttps2dq (%rdx), %ymm9 

// CHECK: vcvttps2dq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7a,0x5b,0xff]       
vcvttps2dq %xmm15, %xmm15 

// CHECK: vcvttps2dq %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x5b,0xf6]       
vcvttps2dq %xmm6, %xmm6 

// CHECK: vcvttps2dq %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x5b,0xff]       
vcvttps2dq %ymm7, %ymm7 

// CHECK: vcvttps2dq %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7e,0x5b,0xc9]       
vcvttps2dq %ymm9, %ymm9 

// CHECK: vcvttsd2si 485498096, %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2c,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttsd2si 485498096, %r13d 

// CHECK: vcvttsd2si 485498096, %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttsd2si 485498096, %r15 

// CHECK: vcvttsd2si 64(%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2c,0x6a,0x40]       
vcvttsd2si 64(%rdx), %r13d 

// CHECK: vcvttsd2si 64(%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x7a,0x40]       
vcvttsd2si 64(%rdx), %r15 

// CHECK: vcvttsd2si -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2c,0x6c,0x82,0xc0]       
vcvttsd2si -64(%rdx,%rax,4), %r13d 

// CHECK: vcvttsd2si 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2c,0x6c,0x82,0x40]       
vcvttsd2si 64(%rdx,%rax,4), %r13d 

// CHECK: vcvttsd2si -64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x7c,0x82,0xc0]       
vcvttsd2si -64(%rdx,%rax,4), %r15 

// CHECK: vcvttsd2si 64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x7c,0x82,0x40]       
vcvttsd2si 64(%rdx,%rax,4), %r15 

// CHECK: vcvttsd2si 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2c,0x6c,0x02,0x40]       
vcvttsd2si 64(%rdx,%rax), %r13d 

// CHECK: vcvttsd2si 64(%rdx,%rax), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x7c,0x02,0x40]       
vcvttsd2si 64(%rdx,%rax), %r15 

// CHECK: vcvttsd2si (%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2c,0x2a]       
vcvttsd2si (%rdx), %r13d 

// CHECK: vcvttsd2si (%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x3a]       
vcvttsd2si (%rdx), %r15 

// CHECK: vcvttsd2si %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x7b,0x2c,0xef]       
vcvttsd2si %xmm15, %r13d 

// CHECK: vcvttsd2si %xmm15, %r15 
// CHECK: encoding: [0xc4,0x41,0xfb,0x2c,0xff]       
vcvttsd2si %xmm15, %r15 

// CHECK: vcvttsd2si %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x7b,0x2c,0xee]       
vcvttsd2si %xmm6, %r13d 

// CHECK: vcvttsd2si %xmm6, %r15 
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0xfe]       
vcvttsd2si %xmm6, %r15 

// CHECK: vcvttss2si 485498096, %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2c,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttss2si 485498096, %r13d 

// CHECK: vcvttss2si 485498096, %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vcvttss2si 485498096, %r15 

// CHECK: vcvttss2si 64(%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2c,0x6a,0x40]       
vcvttss2si 64(%rdx), %r13d 

// CHECK: vcvttss2si 64(%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x7a,0x40]       
vcvttss2si 64(%rdx), %r15 

// CHECK: vcvttss2si -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2c,0x6c,0x82,0xc0]       
vcvttss2si -64(%rdx,%rax,4), %r13d 

// CHECK: vcvttss2si 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2c,0x6c,0x82,0x40]       
vcvttss2si 64(%rdx,%rax,4), %r13d 

// CHECK: vcvttss2si -64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x7c,0x82,0xc0]       
vcvttss2si -64(%rdx,%rax,4), %r15 

// CHECK: vcvttss2si 64(%rdx,%rax,4), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x7c,0x82,0x40]       
vcvttss2si 64(%rdx,%rax,4), %r15 

// CHECK: vcvttss2si 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2c,0x6c,0x02,0x40]       
vcvttss2si 64(%rdx,%rax), %r13d 

// CHECK: vcvttss2si 64(%rdx,%rax), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x7c,0x02,0x40]       
vcvttss2si 64(%rdx,%rax), %r15 

// CHECK: vcvttss2si (%rdx), %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2c,0x2a]       
vcvttss2si (%rdx), %r13d 

// CHECK: vcvttss2si (%rdx), %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x3a]       
vcvttss2si (%rdx), %r15 

// CHECK: vcvttss2si %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x7a,0x2c,0xef]       
vcvttss2si %xmm15, %r13d 

// CHECK: vcvttss2si %xmm15, %r15 
// CHECK: encoding: [0xc4,0x41,0xfa,0x2c,0xff]       
vcvttss2si %xmm15, %r15 

// CHECK: vcvttss2si %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x7a,0x2c,0xee]       
vcvttss2si %xmm6, %r13d 

// CHECK: vcvttss2si %xmm6, %r15 
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0xfe]       
vcvttss2si %xmm6, %r15 

// CHECK: vdivpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096, %xmm15, %xmm15 

// CHECK: vdivpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096, %xmm6, %xmm6 

// CHECK: vdivpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096, %ymm7, %ymm7 

// CHECK: vdivpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivpd 485498096, %ymm9, %ymm9 

// CHECK: vdivpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5e,0x7c,0x82,0xc0]      
vdivpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5e,0x7c,0x82,0x40]      
vdivpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5e,0x74,0x82,0xc0]      
vdivpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5e,0x74,0x82,0x40]      
vdivpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5e,0x7c,0x82,0xc0]      
vdivpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vdivpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5e,0x7c,0x82,0x40]      
vdivpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vdivpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5e,0x4c,0x82,0xc0]      
vdivpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vdivpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5e,0x4c,0x82,0x40]      
vdivpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vdivpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5e,0x7c,0x02,0x40]      
vdivpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vdivpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5e,0x74,0x02,0x40]      
vdivpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vdivpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5e,0x7c,0x02,0x40]      
vdivpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vdivpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5e,0x4c,0x02,0x40]      
vdivpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vdivpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5e,0x7a,0x40]      
vdivpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vdivpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5e,0x72,0x40]      
vdivpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vdivpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5e,0x7a,0x40]      
vdivpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vdivpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5e,0x4a,0x40]      
vdivpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vdivpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5e,0x3a]      
vdivpd (%rdx), %xmm15, %xmm15 

// CHECK: vdivpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5e,0x32]      
vdivpd (%rdx), %xmm6, %xmm6 

// CHECK: vdivpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5e,0x3a]      
vdivpd (%rdx), %ymm7, %ymm7 

// CHECK: vdivpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5e,0x0a]      
vdivpd (%rdx), %ymm9, %ymm9 

// CHECK: vdivpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x5e,0xff]      
vdivpd %xmm15, %xmm15, %xmm15 

// CHECK: vdivpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5e,0xf6]      
vdivpd %xmm6, %xmm6, %xmm6 

// CHECK: vdivpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5e,0xff]      
vdivpd %ymm7, %ymm7, %ymm7 

// CHECK: vdivpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x5e,0xc9]      
vdivpd %ymm9, %ymm9, %ymm9 

// CHECK: vdivps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096, %xmm15, %xmm15 

// CHECK: vdivps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096, %xmm6, %xmm6 

// CHECK: vdivps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096, %ymm7, %ymm7 

// CHECK: vdivps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivps 485498096, %ymm9, %ymm9 

// CHECK: vdivps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5e,0x7c,0x82,0xc0]      
vdivps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5e,0x7c,0x82,0x40]      
vdivps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5e,0x74,0x82,0xc0]      
vdivps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5e,0x74,0x82,0x40]      
vdivps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5e,0x7c,0x82,0xc0]      
vdivps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vdivps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5e,0x7c,0x82,0x40]      
vdivps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vdivps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5e,0x4c,0x82,0xc0]      
vdivps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vdivps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5e,0x4c,0x82,0x40]      
vdivps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vdivps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5e,0x7c,0x02,0x40]      
vdivps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vdivps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5e,0x74,0x02,0x40]      
vdivps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vdivps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5e,0x7c,0x02,0x40]      
vdivps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vdivps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5e,0x4c,0x02,0x40]      
vdivps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vdivps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5e,0x7a,0x40]      
vdivps 64(%rdx), %xmm15, %xmm15 

// CHECK: vdivps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5e,0x72,0x40]      
vdivps 64(%rdx), %xmm6, %xmm6 

// CHECK: vdivps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5e,0x7a,0x40]      
vdivps 64(%rdx), %ymm7, %ymm7 

// CHECK: vdivps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5e,0x4a,0x40]      
vdivps 64(%rdx), %ymm9, %ymm9 

// CHECK: vdivps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5e,0x3a]      
vdivps (%rdx), %xmm15, %xmm15 

// CHECK: vdivps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5e,0x32]      
vdivps (%rdx), %xmm6, %xmm6 

// CHECK: vdivps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5e,0x3a]      
vdivps (%rdx), %ymm7, %ymm7 

// CHECK: vdivps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5e,0x0a]      
vdivps (%rdx), %ymm9, %ymm9 

// CHECK: vdivps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x5e,0xff]      
vdivps %xmm15, %xmm15, %xmm15 

// CHECK: vdivps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5e,0xf6]      
vdivps %xmm6, %xmm6, %xmm6 

// CHECK: vdivps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5e,0xff]      
vdivps %ymm7, %ymm7, %ymm7 

// CHECK: vdivps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x5e,0xc9]      
vdivps %ymm9, %ymm9, %ymm9 

// CHECK: vdivsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivsd 485498096, %xmm15, %xmm15 

// CHECK: vdivsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivsd 485498096, %xmm6, %xmm6 

// CHECK: vdivsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5e,0x7c,0x82,0xc0]      
vdivsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5e,0x7c,0x82,0x40]      
vdivsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5e,0x74,0x82,0xc0]      
vdivsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5e,0x74,0x82,0x40]      
vdivsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5e,0x7c,0x02,0x40]      
vdivsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vdivsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5e,0x74,0x02,0x40]      
vdivsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vdivsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5e,0x7a,0x40]      
vdivsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vdivsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5e,0x72,0x40]      
vdivsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vdivsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5e,0x3a]      
vdivsd (%rdx), %xmm15, %xmm15 

// CHECK: vdivsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5e,0x32]      
vdivsd (%rdx), %xmm6, %xmm6 

// CHECK: vdivsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x5e,0xff]      
vdivsd %xmm15, %xmm15, %xmm15 

// CHECK: vdivsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5e,0xf6]      
vdivsd %xmm6, %xmm6, %xmm6 

// CHECK: vdivss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivss 485498096, %xmm15, %xmm15 

// CHECK: vdivss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vdivss 485498096, %xmm6, %xmm6 

// CHECK: vdivss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5e,0x7c,0x82,0xc0]      
vdivss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5e,0x7c,0x82,0x40]      
vdivss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdivss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5e,0x74,0x82,0xc0]      
vdivss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5e,0x74,0x82,0x40]      
vdivss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdivss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5e,0x7c,0x02,0x40]      
vdivss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vdivss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5e,0x74,0x02,0x40]      
vdivss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vdivss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5e,0x7a,0x40]      
vdivss 64(%rdx), %xmm15, %xmm15 

// CHECK: vdivss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5e,0x72,0x40]      
vdivss 64(%rdx), %xmm6, %xmm6 

// CHECK: vdivss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5e,0x3a]      
vdivss (%rdx), %xmm15, %xmm15 

// CHECK: vdivss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5e,0x32]      
vdivss (%rdx), %xmm6, %xmm6 

// CHECK: vdivss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x5e,0xff]      
vdivss %xmm15, %xmm15, %xmm15 

// CHECK: vdivss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5e,0xf6]      
vdivss %xmm6, %xmm6, %xmm6 

// CHECK: vdppd $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x41,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vdppd $0, 485498096, %xmm15, %xmm15 

// CHECK: vdppd $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x41,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vdppd $0, 485498096, %xmm6, %xmm6 

// CHECK: vdppd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x41,0x7c,0x82,0xc0,0x00]     
vdppd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdppd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x41,0x7c,0x82,0x40,0x00]     
vdppd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdppd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x41,0x74,0x82,0xc0,0x00]     
vdppd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdppd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x41,0x74,0x82,0x40,0x00]     
vdppd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdppd $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x41,0x7c,0x02,0x40,0x00]     
vdppd $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vdppd $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x41,0x74,0x02,0x40,0x00]     
vdppd $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vdppd $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x41,0x7a,0x40,0x00]     
vdppd $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vdppd $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x41,0x72,0x40,0x00]     
vdppd $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vdppd $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x41,0x3a,0x00]     
vdppd $0, (%rdx), %xmm15, %xmm15 

// CHECK: vdppd $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x41,0x32,0x00]     
vdppd $0, (%rdx), %xmm6, %xmm6 

// CHECK: vdppd $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x41,0xff,0x00]     
vdppd $0, %xmm15, %xmm15, %xmm15 

// CHECK: vdppd $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x41,0xf6,0x00]     
vdppd $0, %xmm6, %xmm6, %xmm6 

// CHECK: vdpps $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x40,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096, %xmm15, %xmm15 

// CHECK: vdpps $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x40,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096, %xmm6, %xmm6 

// CHECK: vdpps $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x40,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096, %ymm7, %ymm7 

// CHECK: vdpps $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x40,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vdpps $0, 485498096, %ymm9, %ymm9 

// CHECK: vdpps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x40,0x7c,0x82,0xc0,0x00]     
vdpps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdpps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x40,0x7c,0x82,0x40,0x00]     
vdpps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vdpps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x40,0x74,0x82,0xc0,0x00]     
vdpps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdpps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x40,0x74,0x82,0x40,0x00]     
vdpps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vdpps $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x40,0x7c,0x82,0xc0,0x00]     
vdpps $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vdpps $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x40,0x7c,0x82,0x40,0x00]     
vdpps $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vdpps $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x40,0x4c,0x82,0xc0,0x00]     
vdpps $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vdpps $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x40,0x4c,0x82,0x40,0x00]     
vdpps $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vdpps $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x40,0x7c,0x02,0x40,0x00]     
vdpps $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vdpps $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x40,0x74,0x02,0x40,0x00]     
vdpps $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vdpps $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x40,0x7c,0x02,0x40,0x00]     
vdpps $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vdpps $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x40,0x4c,0x02,0x40,0x00]     
vdpps $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vdpps $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x40,0x7a,0x40,0x00]     
vdpps $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vdpps $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x40,0x72,0x40,0x00]     
vdpps $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vdpps $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x40,0x7a,0x40,0x00]     
vdpps $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vdpps $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x40,0x4a,0x40,0x00]     
vdpps $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vdpps $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x40,0x3a,0x00]     
vdpps $0, (%rdx), %xmm15, %xmm15 

// CHECK: vdpps $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x40,0x32,0x00]     
vdpps $0, (%rdx), %xmm6, %xmm6 

// CHECK: vdpps $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x40,0x3a,0x00]     
vdpps $0, (%rdx), %ymm7, %ymm7 

// CHECK: vdpps $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x40,0x0a,0x00]     
vdpps $0, (%rdx), %ymm9, %ymm9 

// CHECK: vdpps $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x40,0xff,0x00]     
vdpps $0, %xmm15, %xmm15, %xmm15 

// CHECK: vdpps $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x40,0xf6,0x00]     
vdpps $0, %xmm6, %xmm6, %xmm6 

// CHECK: vdpps $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x40,0xff,0x00]     
vdpps $0, %ymm7, %ymm7, %ymm7 

// CHECK: vdpps $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x40,0xc9,0x00]     
vdpps $0, %ymm9, %ymm9, %ymm9 

// CHECK: vextractf128 $0, %ymm7, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractf128 $0, %ymm7, 485498096 

// CHECK: vextractf128 $0, %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x7a,0x40,0x00]      
vextractf128 $0, %ymm7, 64(%rdx) 

// CHECK: vextractf128 $0, %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x7c,0x02,0x40,0x00]      
vextractf128 $0, %ymm7, 64(%rdx,%rax) 

// CHECK: vextractf128 $0, %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x7c,0x82,0xc0,0x00]      
vextractf128 $0, %ymm7, -64(%rdx,%rax,4) 

// CHECK: vextractf128 $0, %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x7c,0x82,0x40,0x00]      
vextractf128 $0, %ymm7, 64(%rdx,%rax,4) 

// CHECK: vextractf128 $0, %ymm7, (%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x3a,0x00]      
vextractf128 $0, %ymm7, (%rdx) 

// CHECK: vextractf128 $0, %ymm7, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0xfe,0x00]      
vextractf128 $0, %ymm7, %xmm6 

// CHECK: vextractf128 $0, %ymm9, 485498096 
// CHECK: encoding: [0xc4,0x63,0x7d,0x19,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractf128 $0, %ymm9, 485498096 

// CHECK: vextractf128 $0, %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x19,0x4a,0x40,0x00]      
vextractf128 $0, %ymm9, 64(%rdx) 

// CHECK: vextractf128 $0, %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x19,0x4c,0x02,0x40,0x00]      
vextractf128 $0, %ymm9, 64(%rdx,%rax) 

// CHECK: vextractf128 $0, %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x19,0x4c,0x82,0xc0,0x00]      
vextractf128 $0, %ymm9, -64(%rdx,%rax,4) 

// CHECK: vextractf128 $0, %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x19,0x4c,0x82,0x40,0x00]      
vextractf128 $0, %ymm9, 64(%rdx,%rax,4) 

// CHECK: vextractf128 $0, %ymm9, (%rdx) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x19,0x0a,0x00]      
vextractf128 $0, %ymm9, (%rdx) 

// CHECK: vextractf128 $0, %ymm9, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x7d,0x19,0xcf,0x00]      
vextractf128 $0, %ymm9, %xmm15 

// CHECK: vextractps $0, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractps $0, %xmm15, 485498096 

// CHECK: vextractps $0, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0x7a,0x40,0x00]      
vextractps $0, %xmm15, 64(%rdx) 

// CHECK: vextractps $0, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0x7c,0x02,0x40,0x00]      
vextractps $0, %xmm15, 64(%rdx,%rax) 

// CHECK: vextractps $0, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0x7c,0x82,0xc0,0x00]      
vextractps $0, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vextractps $0, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0x7c,0x82,0x40,0x00]      
vextractps $0, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vextractps $0, %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x43,0x79,0x17,0xfd,0x00]      
vextractps $0, %xmm15, %r13d 

// CHECK: vextractps $0, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0x3a,0x00]      
vextractps $0, %xmm15, (%rdx) 

// CHECK: vextractps $0, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextractps $0, %xmm6, 485498096 

// CHECK: vextractps $0, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x72,0x40,0x00]      
vextractps $0, %xmm6, 64(%rdx) 

// CHECK: vextractps $0, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x74,0x02,0x40,0x00]      
vextractps $0, %xmm6, 64(%rdx,%rax) 

// CHECK: vextractps $0, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x74,0x82,0xc0,0x00]      
vextractps $0, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vextractps $0, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x74,0x82,0x40,0x00]      
vextractps $0, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vextractps $0, %xmm6, %r13d 
// CHECK: encoding: [0xc4,0xc3,0x79,0x17,0xf5,0x00]      
vextractps $0, %xmm6, %r13d 

// CHECK: vextractps $0, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x32,0x00]      
vextractps $0, %xmm6, (%rdx) 

// CHECK: vhaddpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096, %xmm15, %xmm15 

// CHECK: vhaddpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096, %xmm6, %xmm6 

// CHECK: vhaddpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096, %ymm7, %ymm7 

// CHECK: vhaddpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddpd 485498096, %ymm9, %ymm9 

// CHECK: vhaddpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7c,0x7c,0x82,0xc0]      
vhaddpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhaddpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7c,0x7c,0x82,0x40]      
vhaddpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhaddpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7c,0x74,0x82,0xc0]      
vhaddpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhaddpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7c,0x74,0x82,0x40]      
vhaddpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhaddpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7c,0x7c,0x82,0xc0]      
vhaddpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhaddpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7c,0x7c,0x82,0x40]      
vhaddpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhaddpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7c,0x4c,0x82,0xc0]      
vhaddpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhaddpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7c,0x4c,0x82,0x40]      
vhaddpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhaddpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7c,0x7c,0x02,0x40]      
vhaddpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vhaddpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7c,0x74,0x02,0x40]      
vhaddpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vhaddpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7c,0x7c,0x02,0x40]      
vhaddpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vhaddpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7c,0x4c,0x02,0x40]      
vhaddpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vhaddpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7c,0x7a,0x40]      
vhaddpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vhaddpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7c,0x72,0x40]      
vhaddpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vhaddpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7c,0x7a,0x40]      
vhaddpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vhaddpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7c,0x4a,0x40]      
vhaddpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vhaddpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7c,0x3a]      
vhaddpd (%rdx), %xmm15, %xmm15 

// CHECK: vhaddpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7c,0x32]      
vhaddpd (%rdx), %xmm6, %xmm6 

// CHECK: vhaddpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7c,0x3a]      
vhaddpd (%rdx), %ymm7, %ymm7 

// CHECK: vhaddpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7c,0x0a]      
vhaddpd (%rdx), %ymm9, %ymm9 

// CHECK: vhaddpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x7c,0xff]      
vhaddpd %xmm15, %xmm15, %xmm15 

// CHECK: vhaddpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7c,0xf6]      
vhaddpd %xmm6, %xmm6, %xmm6 

// CHECK: vhaddpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7c,0xff]      
vhaddpd %ymm7, %ymm7, %ymm7 

// CHECK: vhaddpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x7c,0xc9]      
vhaddpd %ymm9, %ymm9, %ymm9 

// CHECK: vhaddps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096, %xmm15, %xmm15 

// CHECK: vhaddps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096, %xmm6, %xmm6 

// CHECK: vhaddps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096, %ymm7, %ymm7 

// CHECK: vhaddps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhaddps 485498096, %ymm9, %ymm9 

// CHECK: vhaddps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7c,0x7c,0x82,0xc0]      
vhaddps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhaddps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7c,0x7c,0x82,0x40]      
vhaddps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhaddps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7c,0x74,0x82,0xc0]      
vhaddps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhaddps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7c,0x74,0x82,0x40]      
vhaddps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhaddps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7c,0x7c,0x82,0xc0]      
vhaddps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhaddps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7c,0x7c,0x82,0x40]      
vhaddps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhaddps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7c,0x4c,0x82,0xc0]      
vhaddps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhaddps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7c,0x4c,0x82,0x40]      
vhaddps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhaddps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7c,0x7c,0x02,0x40]      
vhaddps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vhaddps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7c,0x74,0x02,0x40]      
vhaddps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vhaddps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7c,0x7c,0x02,0x40]      
vhaddps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vhaddps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7c,0x4c,0x02,0x40]      
vhaddps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vhaddps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7c,0x7a,0x40]      
vhaddps 64(%rdx), %xmm15, %xmm15 

// CHECK: vhaddps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7c,0x72,0x40]      
vhaddps 64(%rdx), %xmm6, %xmm6 

// CHECK: vhaddps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7c,0x7a,0x40]      
vhaddps 64(%rdx), %ymm7, %ymm7 

// CHECK: vhaddps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7c,0x4a,0x40]      
vhaddps 64(%rdx), %ymm9, %ymm9 

// CHECK: vhaddps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7c,0x3a]      
vhaddps (%rdx), %xmm15, %xmm15 

// CHECK: vhaddps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7c,0x32]      
vhaddps (%rdx), %xmm6, %xmm6 

// CHECK: vhaddps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7c,0x3a]      
vhaddps (%rdx), %ymm7, %ymm7 

// CHECK: vhaddps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7c,0x0a]      
vhaddps (%rdx), %ymm9, %ymm9 

// CHECK: vhaddps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x7c,0xff]      
vhaddps %xmm15, %xmm15, %xmm15 

// CHECK: vhaddps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7c,0xf6]      
vhaddps %xmm6, %xmm6, %xmm6 

// CHECK: vhaddps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7c,0xff]      
vhaddps %ymm7, %ymm7, %ymm7 

// CHECK: vhaddps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x37,0x7c,0xc9]      
vhaddps %ymm9, %ymm9, %ymm9 

// CHECK: vhsubpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096, %xmm15, %xmm15 

// CHECK: vhsubpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096, %xmm6, %xmm6 

// CHECK: vhsubpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096, %ymm7, %ymm7 

// CHECK: vhsubpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubpd 485498096, %ymm9, %ymm9 

// CHECK: vhsubpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7d,0x7c,0x82,0xc0]      
vhsubpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhsubpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7d,0x7c,0x82,0x40]      
vhsubpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhsubpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7d,0x74,0x82,0xc0]      
vhsubpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhsubpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7d,0x74,0x82,0x40]      
vhsubpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhsubpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7d,0x7c,0x82,0xc0]      
vhsubpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhsubpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7d,0x7c,0x82,0x40]      
vhsubpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhsubpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7d,0x4c,0x82,0xc0]      
vhsubpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhsubpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7d,0x4c,0x82,0x40]      
vhsubpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhsubpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7d,0x7c,0x02,0x40]      
vhsubpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vhsubpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7d,0x74,0x02,0x40]      
vhsubpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vhsubpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7d,0x7c,0x02,0x40]      
vhsubpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vhsubpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7d,0x4c,0x02,0x40]      
vhsubpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vhsubpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7d,0x7a,0x40]      
vhsubpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vhsubpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7d,0x72,0x40]      
vhsubpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vhsubpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7d,0x7a,0x40]      
vhsubpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vhsubpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7d,0x4a,0x40]      
vhsubpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vhsubpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x7d,0x3a]      
vhsubpd (%rdx), %xmm15, %xmm15 

// CHECK: vhsubpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7d,0x32]      
vhsubpd (%rdx), %xmm6, %xmm6 

// CHECK: vhsubpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7d,0x3a]      
vhsubpd (%rdx), %ymm7, %ymm7 

// CHECK: vhsubpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x7d,0x0a]      
vhsubpd (%rdx), %ymm9, %ymm9 

// CHECK: vhsubpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x7d,0xff]      
vhsubpd %xmm15, %xmm15, %xmm15 

// CHECK: vhsubpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x7d,0xf6]      
vhsubpd %xmm6, %xmm6, %xmm6 

// CHECK: vhsubpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x7d,0xff]      
vhsubpd %ymm7, %ymm7, %ymm7 

// CHECK: vhsubpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x7d,0xc9]      
vhsubpd %ymm9, %ymm9, %ymm9 

// CHECK: vhsubps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096, %xmm15, %xmm15 

// CHECK: vhsubps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096, %xmm6, %xmm6 

// CHECK: vhsubps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096, %ymm7, %ymm7 

// CHECK: vhsubps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vhsubps 485498096, %ymm9, %ymm9 

// CHECK: vhsubps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7d,0x7c,0x82,0xc0]      
vhsubps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhsubps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7d,0x7c,0x82,0x40]      
vhsubps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vhsubps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7d,0x74,0x82,0xc0]      
vhsubps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhsubps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7d,0x74,0x82,0x40]      
vhsubps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vhsubps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7d,0x7c,0x82,0xc0]      
vhsubps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhsubps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7d,0x7c,0x82,0x40]      
vhsubps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vhsubps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7d,0x4c,0x82,0xc0]      
vhsubps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhsubps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7d,0x4c,0x82,0x40]      
vhsubps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vhsubps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7d,0x7c,0x02,0x40]      
vhsubps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vhsubps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7d,0x74,0x02,0x40]      
vhsubps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vhsubps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7d,0x7c,0x02,0x40]      
vhsubps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vhsubps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7d,0x4c,0x02,0x40]      
vhsubps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vhsubps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7d,0x7a,0x40]      
vhsubps 64(%rdx), %xmm15, %xmm15 

// CHECK: vhsubps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7d,0x72,0x40]      
vhsubps 64(%rdx), %xmm6, %xmm6 

// CHECK: vhsubps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7d,0x7a,0x40]      
vhsubps 64(%rdx), %ymm7, %ymm7 

// CHECK: vhsubps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7d,0x4a,0x40]      
vhsubps 64(%rdx), %ymm9, %ymm9 

// CHECK: vhsubps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x7d,0x3a]      
vhsubps (%rdx), %xmm15, %xmm15 

// CHECK: vhsubps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7d,0x32]      
vhsubps (%rdx), %xmm6, %xmm6 

// CHECK: vhsubps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7d,0x3a]      
vhsubps (%rdx), %ymm7, %ymm7 

// CHECK: vhsubps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x37,0x7d,0x0a]      
vhsubps (%rdx), %ymm9, %ymm9 

// CHECK: vhsubps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x7d,0xff]      
vhsubps %xmm15, %xmm15, %xmm15 

// CHECK: vhsubps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x7d,0xf6]      
vhsubps %xmm6, %xmm6, %xmm6 

// CHECK: vhsubps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc7,0x7d,0xff]      
vhsubps %ymm7, %ymm7, %ymm7 

// CHECK: vhsubps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x37,0x7d,0xc9]      
vhsubps %ymm9, %ymm9, %ymm9 

// CHECK: vinsertf128 $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x18,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertf128 $0, 485498096, %ymm7, %ymm7 

// CHECK: vinsertf128 $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x18,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertf128 $0, 485498096, %ymm9, %ymm9 

// CHECK: vinsertf128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x18,0x7c,0x82,0xc0,0x00]     
vinsertf128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vinsertf128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x18,0x7c,0x82,0x40,0x00]     
vinsertf128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vinsertf128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x18,0x4c,0x82,0xc0,0x00]     
vinsertf128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vinsertf128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x18,0x4c,0x82,0x40,0x00]     
vinsertf128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vinsertf128 $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x18,0x7c,0x02,0x40,0x00]     
vinsertf128 $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vinsertf128 $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x18,0x4c,0x02,0x40,0x00]     
vinsertf128 $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vinsertf128 $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x18,0x7a,0x40,0x00]     
vinsertf128 $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vinsertf128 $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x18,0x4a,0x40,0x00]     
vinsertf128 $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vinsertf128 $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x18,0x3a,0x00]     
vinsertf128 $0, (%rdx), %ymm7, %ymm7 

// CHECK: vinsertf128 $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x18,0x0a,0x00]     
vinsertf128 $0, (%rdx), %ymm9, %ymm9 

// CHECK: vinsertf128 $0, %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x18,0xcf,0x00]     
vinsertf128 $0, %xmm15, %ymm9, %ymm9 

// CHECK: vinsertf128 $0, %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x18,0xfe,0x00]     
vinsertf128 $0, %xmm6, %ymm7, %ymm7 

// CHECK: vinsertps $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x21,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertps $0, 485498096, %xmm15, %xmm15 

// CHECK: vinsertps $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x21,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinsertps $0, 485498096, %xmm6, %xmm6 

// CHECK: vinsertps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x21,0x7c,0x82,0xc0,0x00]     
vinsertps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vinsertps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x21,0x7c,0x82,0x40,0x00]     
vinsertps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vinsertps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x21,0x74,0x82,0xc0,0x00]     
vinsertps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vinsertps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x21,0x74,0x82,0x40,0x00]     
vinsertps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vinsertps $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x21,0x7c,0x02,0x40,0x00]     
vinsertps $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vinsertps $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x21,0x74,0x02,0x40,0x00]     
vinsertps $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vinsertps $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x21,0x7a,0x40,0x00]     
vinsertps $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vinsertps $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x21,0x72,0x40,0x00]     
vinsertps $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vinsertps $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x21,0x3a,0x00]     
vinsertps $0, (%rdx), %xmm15, %xmm15 

// CHECK: vinsertps $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x21,0x32,0x00]     
vinsertps $0, (%rdx), %xmm6, %xmm6 

// CHECK: vinsertps $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x21,0xff,0x00]     
vinsertps $0, %xmm15, %xmm15, %xmm15 

// CHECK: vinsertps $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x21,0xf6,0x00]     
vinsertps $0, %xmm6, %xmm6, %xmm6 

// CHECK: vlddqu 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xf0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096, %xmm15 

// CHECK: vlddqu 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096, %xmm6 

// CHECK: vlddqu 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xff,0xf0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096, %ymm7 

// CHECK: vlddqu 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7f,0xf0,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vlddqu 485498096, %ymm9 

// CHECK: vlddqu -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xf0,0x7c,0x82,0xc0]       
vlddqu -64(%rdx,%rax,4), %xmm15 

// CHECK: vlddqu 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xf0,0x7c,0x82,0x40]       
vlddqu 64(%rdx,%rax,4), %xmm15 

// CHECK: vlddqu -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x74,0x82,0xc0]       
vlddqu -64(%rdx,%rax,4), %xmm6 

// CHECK: vlddqu 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x74,0x82,0x40]       
vlddqu 64(%rdx,%rax,4), %xmm6 

// CHECK: vlddqu -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xff,0xf0,0x7c,0x82,0xc0]       
vlddqu -64(%rdx,%rax,4), %ymm7 

// CHECK: vlddqu 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xff,0xf0,0x7c,0x82,0x40]       
vlddqu 64(%rdx,%rax,4), %ymm7 

// CHECK: vlddqu -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0xf0,0x4c,0x82,0xc0]       
vlddqu -64(%rdx,%rax,4), %ymm9 

// CHECK: vlddqu 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0xf0,0x4c,0x82,0x40]       
vlddqu 64(%rdx,%rax,4), %ymm9 

// CHECK: vlddqu 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xf0,0x7c,0x02,0x40]       
vlddqu 64(%rdx,%rax), %xmm15 

// CHECK: vlddqu 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x74,0x02,0x40]       
vlddqu 64(%rdx,%rax), %xmm6 

// CHECK: vlddqu 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xff,0xf0,0x7c,0x02,0x40]       
vlddqu 64(%rdx,%rax), %ymm7 

// CHECK: vlddqu 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0xf0,0x4c,0x02,0x40]       
vlddqu 64(%rdx,%rax), %ymm9 

// CHECK: vlddqu 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xf0,0x7a,0x40]       
vlddqu 64(%rdx), %xmm15 

// CHECK: vlddqu 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x72,0x40]       
vlddqu 64(%rdx), %xmm6 

// CHECK: vlddqu 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xff,0xf0,0x7a,0x40]       
vlddqu 64(%rdx), %ymm7 

// CHECK: vlddqu 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0xf0,0x4a,0x40]       
vlddqu 64(%rdx), %ymm9 

// CHECK: vlddqu (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0xf0,0x3a]       
vlddqu (%rdx), %xmm15 

// CHECK: vlddqu (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0xf0,0x32]       
vlddqu (%rdx), %xmm6 

// CHECK: vlddqu (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xff,0xf0,0x3a]       
vlddqu (%rdx), %ymm7 

// CHECK: vlddqu (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0xf0,0x0a]       
vlddqu (%rdx), %ymm9 

// CHECK: vldmxcsr 485498096 
// CHECK: encoding: [0xc5,0xf8,0xae,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
vldmxcsr 485498096 

// CHECK: vldmxcsr 64(%rdx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x52,0x40]        
vldmxcsr 64(%rdx) 

// CHECK: vldmxcsr -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x54,0x82,0xc0]        
vldmxcsr -64(%rdx,%rax,4) 

// CHECK: vldmxcsr 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x54,0x82,0x40]        
vldmxcsr 64(%rdx,%rax,4) 

// CHECK: vldmxcsr 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x54,0x02,0x40]        
vldmxcsr 64(%rdx,%rax) 

// CHECK: vldmxcsr (%rdx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x12]        
vldmxcsr (%rdx) 

// CHECK: vmaskmovdqu %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0xf7,0xff]       
vmaskmovdqu %xmm15, %xmm15 

// CHECK: vmaskmovdqu %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0xf7,0xf6]       
vmaskmovdqu %xmm6, %xmm6 

// CHECK: vmaskmovpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096, %xmm15, %xmm15 

// CHECK: vmaskmovpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096, %xmm6, %xmm6 

// CHECK: vmaskmovpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096, %ymm7, %ymm7 

// CHECK: vmaskmovpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd 485498096, %ymm9, %ymm9 

// CHECK: vmaskmovpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2d,0x7c,0x82,0xc0]      
vmaskmovpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaskmovpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2d,0x7c,0x82,0x40]      
vmaskmovpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaskmovpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2d,0x74,0x82,0xc0]      
vmaskmovpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaskmovpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2d,0x74,0x82,0x40]      
vmaskmovpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaskmovpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2d,0x7c,0x82,0xc0]      
vmaskmovpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaskmovpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2d,0x7c,0x82,0x40]      
vmaskmovpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaskmovpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2d,0x4c,0x82,0xc0]      
vmaskmovpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaskmovpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2d,0x4c,0x82,0x40]      
vmaskmovpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaskmovpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2d,0x7c,0x02,0x40]      
vmaskmovpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmaskmovpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2d,0x74,0x02,0x40]      
vmaskmovpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmaskmovpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2d,0x7c,0x02,0x40]      
vmaskmovpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vmaskmovpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2d,0x4c,0x02,0x40]      
vmaskmovpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vmaskmovpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2d,0x7a,0x40]      
vmaskmovpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vmaskmovpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2d,0x72,0x40]      
vmaskmovpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vmaskmovpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2d,0x7a,0x40]      
vmaskmovpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vmaskmovpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2d,0x4a,0x40]      
vmaskmovpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vmaskmovpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2d,0x3a]      
vmaskmovpd (%rdx), %xmm15, %xmm15 

// CHECK: vmaskmovpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2d,0x32]      
vmaskmovpd (%rdx), %xmm6, %xmm6 

// CHECK: vmaskmovpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2d,0x3a]      
vmaskmovpd (%rdx), %ymm7, %ymm7 

// CHECK: vmaskmovpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2d,0x0a]      
vmaskmovpd (%rdx), %ymm9, %ymm9 

// CHECK: vmaskmovpd %xmm15, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x62,0x01,0x2f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %xmm15, %xmm15, 485498096 

// CHECK: vmaskmovpd %xmm15, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2f,0x7a,0x40]      
vmaskmovpd %xmm15, %xmm15, 64(%rdx) 

// CHECK: vmaskmovpd %xmm15, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2f,0x7c,0x02,0x40]      
vmaskmovpd %xmm15, %xmm15, 64(%rdx,%rax) 

// CHECK: vmaskmovpd %xmm15, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2f,0x7c,0x82,0xc0]      
vmaskmovpd %xmm15, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %xmm15, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2f,0x7c,0x82,0x40]      
vmaskmovpd %xmm15, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %xmm15, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2f,0x3a]      
vmaskmovpd %xmm15, %xmm15, (%rdx) 

// CHECK: vmaskmovpd %xmm6, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %xmm6, %xmm6, 485498096 

// CHECK: vmaskmovpd %xmm6, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2f,0x72,0x40]      
vmaskmovpd %xmm6, %xmm6, 64(%rdx) 

// CHECK: vmaskmovpd %xmm6, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2f,0x74,0x02,0x40]      
vmaskmovpd %xmm6, %xmm6, 64(%rdx,%rax) 

// CHECK: vmaskmovpd %xmm6, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2f,0x74,0x82,0xc0]      
vmaskmovpd %xmm6, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %xmm6, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2f,0x74,0x82,0x40]      
vmaskmovpd %xmm6, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %xmm6, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2f,0x32]      
vmaskmovpd %xmm6, %xmm6, (%rdx) 

// CHECK: vmaskmovpd %ymm7, %ymm7, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %ymm7, %ymm7, 485498096 

// CHECK: vmaskmovpd %ymm7, %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2f,0x7a,0x40]      
vmaskmovpd %ymm7, %ymm7, 64(%rdx) 

// CHECK: vmaskmovpd %ymm7, %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2f,0x7c,0x02,0x40]      
vmaskmovpd %ymm7, %ymm7, 64(%rdx,%rax) 

// CHECK: vmaskmovpd %ymm7, %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2f,0x7c,0x82,0xc0]      
vmaskmovpd %ymm7, %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %ymm7, %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2f,0x7c,0x82,0x40]      
vmaskmovpd %ymm7, %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %ymm7, %ymm7, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2f,0x3a]      
vmaskmovpd %ymm7, %ymm7, (%rdx) 

// CHECK: vmaskmovpd %ymm9, %ymm9, 485498096 
// CHECK: encoding: [0xc4,0x62,0x35,0x2f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovpd %ymm9, %ymm9, 485498096 

// CHECK: vmaskmovpd %ymm9, %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2f,0x4a,0x40]      
vmaskmovpd %ymm9, %ymm9, 64(%rdx) 

// CHECK: vmaskmovpd %ymm9, %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2f,0x4c,0x02,0x40]      
vmaskmovpd %ymm9, %ymm9, 64(%rdx,%rax) 

// CHECK: vmaskmovpd %ymm9, %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2f,0x4c,0x82,0xc0]      
vmaskmovpd %ymm9, %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %ymm9, %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2f,0x4c,0x82,0x40]      
vmaskmovpd %ymm9, %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmaskmovpd %ymm9, %ymm9, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2f,0x0a]      
vmaskmovpd %ymm9, %ymm9, (%rdx) 

// CHECK: vmaskmovps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096, %xmm15, %xmm15 

// CHECK: vmaskmovps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096, %xmm6, %xmm6 

// CHECK: vmaskmovps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096, %ymm7, %ymm7 

// CHECK: vmaskmovps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps 485498096, %ymm9, %ymm9 

// CHECK: vmaskmovps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2c,0x7c,0x82,0xc0]      
vmaskmovps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaskmovps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2c,0x7c,0x82,0x40]      
vmaskmovps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaskmovps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2c,0x74,0x82,0xc0]      
vmaskmovps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaskmovps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2c,0x74,0x82,0x40]      
vmaskmovps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaskmovps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2c,0x7c,0x82,0xc0]      
vmaskmovps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaskmovps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2c,0x7c,0x82,0x40]      
vmaskmovps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaskmovps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2c,0x4c,0x82,0xc0]      
vmaskmovps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaskmovps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2c,0x4c,0x82,0x40]      
vmaskmovps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaskmovps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2c,0x7c,0x02,0x40]      
vmaskmovps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmaskmovps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2c,0x74,0x02,0x40]      
vmaskmovps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmaskmovps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2c,0x7c,0x02,0x40]      
vmaskmovps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vmaskmovps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2c,0x4c,0x02,0x40]      
vmaskmovps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vmaskmovps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2c,0x7a,0x40]      
vmaskmovps 64(%rdx), %xmm15, %xmm15 

// CHECK: vmaskmovps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2c,0x72,0x40]      
vmaskmovps 64(%rdx), %xmm6, %xmm6 

// CHECK: vmaskmovps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2c,0x7a,0x40]      
vmaskmovps 64(%rdx), %ymm7, %ymm7 

// CHECK: vmaskmovps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2c,0x4a,0x40]      
vmaskmovps 64(%rdx), %ymm9, %ymm9 

// CHECK: vmaskmovps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2c,0x3a]      
vmaskmovps (%rdx), %xmm15, %xmm15 

// CHECK: vmaskmovps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2c,0x32]      
vmaskmovps (%rdx), %xmm6, %xmm6 

// CHECK: vmaskmovps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2c,0x3a]      
vmaskmovps (%rdx), %ymm7, %ymm7 

// CHECK: vmaskmovps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2c,0x0a]      
vmaskmovps (%rdx), %ymm9, %ymm9 

// CHECK: vmaskmovps %xmm15, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x62,0x01,0x2e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %xmm15, %xmm15, 485498096 

// CHECK: vmaskmovps %xmm15, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2e,0x7a,0x40]      
vmaskmovps %xmm15, %xmm15, 64(%rdx) 

// CHECK: vmaskmovps %xmm15, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2e,0x7c,0x02,0x40]      
vmaskmovps %xmm15, %xmm15, 64(%rdx,%rax) 

// CHECK: vmaskmovps %xmm15, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2e,0x7c,0x82,0xc0]      
vmaskmovps %xmm15, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmaskmovps %xmm15, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2e,0x7c,0x82,0x40]      
vmaskmovps %xmm15, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmaskmovps %xmm15, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0x01,0x2e,0x3a]      
vmaskmovps %xmm15, %xmm15, (%rdx) 

// CHECK: vmaskmovps %xmm6, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %xmm6, %xmm6, 485498096 

// CHECK: vmaskmovps %xmm6, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2e,0x72,0x40]      
vmaskmovps %xmm6, %xmm6, 64(%rdx) 

// CHECK: vmaskmovps %xmm6, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2e,0x74,0x02,0x40]      
vmaskmovps %xmm6, %xmm6, 64(%rdx,%rax) 

// CHECK: vmaskmovps %xmm6, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2e,0x74,0x82,0xc0]      
vmaskmovps %xmm6, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmaskmovps %xmm6, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2e,0x74,0x82,0x40]      
vmaskmovps %xmm6, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmaskmovps %xmm6, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2e,0x32]      
vmaskmovps %xmm6, %xmm6, (%rdx) 

// CHECK: vmaskmovps %ymm7, %ymm7, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %ymm7, %ymm7, 485498096 

// CHECK: vmaskmovps %ymm7, %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2e,0x7a,0x40]      
vmaskmovps %ymm7, %ymm7, 64(%rdx) 

// CHECK: vmaskmovps %ymm7, %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2e,0x7c,0x02,0x40]      
vmaskmovps %ymm7, %ymm7, 64(%rdx,%rax) 

// CHECK: vmaskmovps %ymm7, %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2e,0x7c,0x82,0xc0]      
vmaskmovps %ymm7, %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmaskmovps %ymm7, %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2e,0x7c,0x82,0x40]      
vmaskmovps %ymm7, %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmaskmovps %ymm7, %ymm7, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2e,0x3a]      
vmaskmovps %ymm7, %ymm7, (%rdx) 

// CHECK: vmaskmovps %ymm9, %ymm9, 485498096 
// CHECK: encoding: [0xc4,0x62,0x35,0x2e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaskmovps %ymm9, %ymm9, 485498096 

// CHECK: vmaskmovps %ymm9, %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2e,0x4a,0x40]      
vmaskmovps %ymm9, %ymm9, 64(%rdx) 

// CHECK: vmaskmovps %ymm9, %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2e,0x4c,0x02,0x40]      
vmaskmovps %ymm9, %ymm9, 64(%rdx,%rax) 

// CHECK: vmaskmovps %ymm9, %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2e,0x4c,0x82,0xc0]      
vmaskmovps %ymm9, %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmaskmovps %ymm9, %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2e,0x4c,0x82,0x40]      
vmaskmovps %ymm9, %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmaskmovps %ymm9, %ymm9, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0x35,0x2e,0x0a]      
vmaskmovps %ymm9, %ymm9, (%rdx) 

// CHECK: vmaxpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096, %xmm15, %xmm15 

// CHECK: vmaxpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096, %xmm6, %xmm6 

// CHECK: vmaxpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096, %ymm7, %ymm7 

// CHECK: vmaxpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxpd 485498096, %ymm9, %ymm9 

// CHECK: vmaxpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5f,0x7c,0x82,0xc0]      
vmaxpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5f,0x7c,0x82,0x40]      
vmaxpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5f,0x74,0x82,0xc0]      
vmaxpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5f,0x74,0x82,0x40]      
vmaxpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5f,0x7c,0x82,0xc0]      
vmaxpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaxpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5f,0x7c,0x82,0x40]      
vmaxpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaxpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5f,0x4c,0x82,0xc0]      
vmaxpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaxpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5f,0x4c,0x82,0x40]      
vmaxpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaxpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5f,0x7c,0x02,0x40]      
vmaxpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmaxpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5f,0x74,0x02,0x40]      
vmaxpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmaxpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5f,0x7c,0x02,0x40]      
vmaxpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vmaxpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5f,0x4c,0x02,0x40]      
vmaxpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vmaxpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5f,0x7a,0x40]      
vmaxpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vmaxpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5f,0x72,0x40]      
vmaxpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vmaxpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5f,0x7a,0x40]      
vmaxpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vmaxpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5f,0x4a,0x40]      
vmaxpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vmaxpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5f,0x3a]      
vmaxpd (%rdx), %xmm15, %xmm15 

// CHECK: vmaxpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5f,0x32]      
vmaxpd (%rdx), %xmm6, %xmm6 

// CHECK: vmaxpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5f,0x3a]      
vmaxpd (%rdx), %ymm7, %ymm7 

// CHECK: vmaxpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5f,0x0a]      
vmaxpd (%rdx), %ymm9, %ymm9 

// CHECK: vmaxpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x5f,0xff]      
vmaxpd %xmm15, %xmm15, %xmm15 

// CHECK: vmaxpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5f,0xf6]      
vmaxpd %xmm6, %xmm6, %xmm6 

// CHECK: vmaxpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5f,0xff]      
vmaxpd %ymm7, %ymm7, %ymm7 

// CHECK: vmaxpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x5f,0xc9]      
vmaxpd %ymm9, %ymm9, %ymm9 

// CHECK: vmaxps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096, %xmm15, %xmm15 

// CHECK: vmaxps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096, %xmm6, %xmm6 

// CHECK: vmaxps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096, %ymm7, %ymm7 

// CHECK: vmaxps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxps 485498096, %ymm9, %ymm9 

// CHECK: vmaxps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5f,0x7c,0x82,0xc0]      
vmaxps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5f,0x7c,0x82,0x40]      
vmaxps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5f,0x74,0x82,0xc0]      
vmaxps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5f,0x74,0x82,0x40]      
vmaxps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5f,0x7c,0x82,0xc0]      
vmaxps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaxps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5f,0x7c,0x82,0x40]      
vmaxps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmaxps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5f,0x4c,0x82,0xc0]      
vmaxps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaxps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5f,0x4c,0x82,0x40]      
vmaxps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmaxps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5f,0x7c,0x02,0x40]      
vmaxps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmaxps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5f,0x74,0x02,0x40]      
vmaxps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmaxps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5f,0x7c,0x02,0x40]      
vmaxps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vmaxps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5f,0x4c,0x02,0x40]      
vmaxps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vmaxps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5f,0x7a,0x40]      
vmaxps 64(%rdx), %xmm15, %xmm15 

// CHECK: vmaxps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5f,0x72,0x40]      
vmaxps 64(%rdx), %xmm6, %xmm6 

// CHECK: vmaxps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5f,0x7a,0x40]      
vmaxps 64(%rdx), %ymm7, %ymm7 

// CHECK: vmaxps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5f,0x4a,0x40]      
vmaxps 64(%rdx), %ymm9, %ymm9 

// CHECK: vmaxps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5f,0x3a]      
vmaxps (%rdx), %xmm15, %xmm15 

// CHECK: vmaxps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5f,0x32]      
vmaxps (%rdx), %xmm6, %xmm6 

// CHECK: vmaxps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5f,0x3a]      
vmaxps (%rdx), %ymm7, %ymm7 

// CHECK: vmaxps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5f,0x0a]      
vmaxps (%rdx), %ymm9, %ymm9 

// CHECK: vmaxps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x5f,0xff]      
vmaxps %xmm15, %xmm15, %xmm15 

// CHECK: vmaxps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5f,0xf6]      
vmaxps %xmm6, %xmm6, %xmm6 

// CHECK: vmaxps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5f,0xff]      
vmaxps %ymm7, %ymm7, %ymm7 

// CHECK: vmaxps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x5f,0xc9]      
vmaxps %ymm9, %ymm9, %ymm9 

// CHECK: vmaxsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxsd 485498096, %xmm15, %xmm15 

// CHECK: vmaxsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxsd 485498096, %xmm6, %xmm6 

// CHECK: vmaxsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5f,0x7c,0x82,0xc0]      
vmaxsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5f,0x7c,0x82,0x40]      
vmaxsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5f,0x74,0x82,0xc0]      
vmaxsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5f,0x74,0x82,0x40]      
vmaxsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5f,0x7c,0x02,0x40]      
vmaxsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmaxsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5f,0x74,0x02,0x40]      
vmaxsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmaxsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5f,0x7a,0x40]      
vmaxsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vmaxsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5f,0x72,0x40]      
vmaxsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vmaxsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5f,0x3a]      
vmaxsd (%rdx), %xmm15, %xmm15 

// CHECK: vmaxsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5f,0x32]      
vmaxsd (%rdx), %xmm6, %xmm6 

// CHECK: vmaxsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x5f,0xff]      
vmaxsd %xmm15, %xmm15, %xmm15 

// CHECK: vmaxsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5f,0xf6]      
vmaxsd %xmm6, %xmm6, %xmm6 

// CHECK: vmaxss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxss 485498096, %xmm15, %xmm15 

// CHECK: vmaxss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmaxss 485498096, %xmm6, %xmm6 

// CHECK: vmaxss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5f,0x7c,0x82,0xc0]      
vmaxss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5f,0x7c,0x82,0x40]      
vmaxss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmaxss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5f,0x74,0x82,0xc0]      
vmaxss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5f,0x74,0x82,0x40]      
vmaxss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmaxss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5f,0x7c,0x02,0x40]      
vmaxss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmaxss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5f,0x74,0x02,0x40]      
vmaxss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmaxss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5f,0x7a,0x40]      
vmaxss 64(%rdx), %xmm15, %xmm15 

// CHECK: vmaxss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5f,0x72,0x40]      
vmaxss 64(%rdx), %xmm6, %xmm6 

// CHECK: vmaxss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5f,0x3a]      
vmaxss (%rdx), %xmm15, %xmm15 

// CHECK: vmaxss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5f,0x32]      
vmaxss (%rdx), %xmm6, %xmm6 

// CHECK: vmaxss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x5f,0xff]      
vmaxss %xmm15, %xmm15, %xmm15 

// CHECK: vmaxss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5f,0xf6]      
vmaxss %xmm6, %xmm6, %xmm6 

// CHECK: vminpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096, %xmm15, %xmm15 

// CHECK: vminpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096, %xmm6, %xmm6 

// CHECK: vminpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096, %ymm7, %ymm7 

// CHECK: vminpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminpd 485498096, %ymm9, %ymm9 

// CHECK: vminpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5d,0x7c,0x82,0xc0]      
vminpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5d,0x7c,0x82,0x40]      
vminpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5d,0x74,0x82,0xc0]      
vminpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5d,0x74,0x82,0x40]      
vminpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5d,0x7c,0x82,0xc0]      
vminpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vminpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5d,0x7c,0x82,0x40]      
vminpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vminpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5d,0x4c,0x82,0xc0]      
vminpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vminpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5d,0x4c,0x82,0x40]      
vminpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vminpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5d,0x7c,0x02,0x40]      
vminpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vminpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5d,0x74,0x02,0x40]      
vminpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vminpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5d,0x7c,0x02,0x40]      
vminpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vminpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5d,0x4c,0x02,0x40]      
vminpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vminpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5d,0x7a,0x40]      
vminpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vminpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5d,0x72,0x40]      
vminpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vminpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5d,0x7a,0x40]      
vminpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vminpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5d,0x4a,0x40]      
vminpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vminpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5d,0x3a]      
vminpd (%rdx), %xmm15, %xmm15 

// CHECK: vminpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5d,0x32]      
vminpd (%rdx), %xmm6, %xmm6 

// CHECK: vminpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5d,0x3a]      
vminpd (%rdx), %ymm7, %ymm7 

// CHECK: vminpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5d,0x0a]      
vminpd (%rdx), %ymm9, %ymm9 

// CHECK: vminpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x5d,0xff]      
vminpd %xmm15, %xmm15, %xmm15 

// CHECK: vminpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5d,0xf6]      
vminpd %xmm6, %xmm6, %xmm6 

// CHECK: vminpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5d,0xff]      
vminpd %ymm7, %ymm7, %ymm7 

// CHECK: vminpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x5d,0xc9]      
vminpd %ymm9, %ymm9, %ymm9 

// CHECK: vminps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096, %xmm15, %xmm15 

// CHECK: vminps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096, %xmm6, %xmm6 

// CHECK: vminps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096, %ymm7, %ymm7 

// CHECK: vminps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminps 485498096, %ymm9, %ymm9 

// CHECK: vminps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5d,0x7c,0x82,0xc0]      
vminps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5d,0x7c,0x82,0x40]      
vminps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5d,0x74,0x82,0xc0]      
vminps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5d,0x74,0x82,0x40]      
vminps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5d,0x7c,0x82,0xc0]      
vminps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vminps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5d,0x7c,0x82,0x40]      
vminps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vminps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5d,0x4c,0x82,0xc0]      
vminps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vminps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5d,0x4c,0x82,0x40]      
vminps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vminps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5d,0x7c,0x02,0x40]      
vminps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vminps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5d,0x74,0x02,0x40]      
vminps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vminps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5d,0x7c,0x02,0x40]      
vminps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vminps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5d,0x4c,0x02,0x40]      
vminps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vminps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5d,0x7a,0x40]      
vminps 64(%rdx), %xmm15, %xmm15 

// CHECK: vminps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5d,0x72,0x40]      
vminps 64(%rdx), %xmm6, %xmm6 

// CHECK: vminps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5d,0x7a,0x40]      
vminps 64(%rdx), %ymm7, %ymm7 

// CHECK: vminps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5d,0x4a,0x40]      
vminps 64(%rdx), %ymm9, %ymm9 

// CHECK: vminps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5d,0x3a]      
vminps (%rdx), %xmm15, %xmm15 

// CHECK: vminps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5d,0x32]      
vminps (%rdx), %xmm6, %xmm6 

// CHECK: vminps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5d,0x3a]      
vminps (%rdx), %ymm7, %ymm7 

// CHECK: vminps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5d,0x0a]      
vminps (%rdx), %ymm9, %ymm9 

// CHECK: vminps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x5d,0xff]      
vminps %xmm15, %xmm15, %xmm15 

// CHECK: vminps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5d,0xf6]      
vminps %xmm6, %xmm6, %xmm6 

// CHECK: vminps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5d,0xff]      
vminps %ymm7, %ymm7, %ymm7 

// CHECK: vminps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x5d,0xc9]      
vminps %ymm9, %ymm9, %ymm9 

// CHECK: vminsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminsd 485498096, %xmm15, %xmm15 

// CHECK: vminsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vminsd 485498096, %xmm6, %xmm6 

// CHECK: vminsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5d,0x7c,0x82,0xc0]      
vminsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5d,0x7c,0x82,0x40]      
vminsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5d,0x74,0x82,0xc0]      
vminsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5d,0x74,0x82,0x40]      
vminsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5d,0x7c,0x02,0x40]      
vminsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vminsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5d,0x74,0x02,0x40]      
vminsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vminsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5d,0x7a,0x40]      
vminsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vminsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5d,0x72,0x40]      
vminsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vminsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5d,0x3a]      
vminsd (%rdx), %xmm15, %xmm15 

// CHECK: vminsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5d,0x32]      
vminsd (%rdx), %xmm6, %xmm6 

// CHECK: vminsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x5d,0xff]      
vminsd %xmm15, %xmm15, %xmm15 

// CHECK: vminsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5d,0xf6]      
vminsd %xmm6, %xmm6, %xmm6 

// CHECK: vminss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vminss 485498096, %xmm15, %xmm15 

// CHECK: vminss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vminss 485498096, %xmm6, %xmm6 

// CHECK: vminss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5d,0x7c,0x82,0xc0]      
vminss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5d,0x7c,0x82,0x40]      
vminss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vminss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5d,0x74,0x82,0xc0]      
vminss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5d,0x74,0x82,0x40]      
vminss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vminss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5d,0x7c,0x02,0x40]      
vminss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vminss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5d,0x74,0x02,0x40]      
vminss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vminss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5d,0x7a,0x40]      
vminss 64(%rdx), %xmm15, %xmm15 

// CHECK: vminss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5d,0x72,0x40]      
vminss 64(%rdx), %xmm6, %xmm6 

// CHECK: vminss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5d,0x3a]      
vminss (%rdx), %xmm15, %xmm15 

// CHECK: vminss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5d,0x32]      
vminss (%rdx), %xmm6, %xmm6 

// CHECK: vminss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x5d,0xff]      
vminss %xmm15, %xmm15, %xmm15 

// CHECK: vminss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5d,0xf6]      
vminss %xmm6, %xmm6, %xmm6 

// CHECK: vmovapd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x28,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096, %xmm15 

// CHECK: vmovapd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x28,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096, %xmm6 

// CHECK: vmovapd 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x28,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096, %ymm7 

// CHECK: vmovapd 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x28,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd 485498096, %ymm9 

// CHECK: vmovapd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x28,0x7c,0x82,0xc0]       
vmovapd -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovapd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x28,0x7c,0x82,0x40]       
vmovapd 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovapd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x28,0x74,0x82,0xc0]       
vmovapd -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovapd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x28,0x74,0x82,0x40]       
vmovapd 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovapd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x28,0x7c,0x82,0xc0]       
vmovapd -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovapd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x28,0x7c,0x82,0x40]       
vmovapd 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovapd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x28,0x4c,0x82,0xc0]       
vmovapd -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovapd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x28,0x4c,0x82,0x40]       
vmovapd 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovapd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x28,0x7c,0x02,0x40]       
vmovapd 64(%rdx,%rax), %xmm15 

// CHECK: vmovapd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x28,0x74,0x02,0x40]       
vmovapd 64(%rdx,%rax), %xmm6 

// CHECK: vmovapd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x28,0x7c,0x02,0x40]       
vmovapd 64(%rdx,%rax), %ymm7 

// CHECK: vmovapd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x28,0x4c,0x02,0x40]       
vmovapd 64(%rdx,%rax), %ymm9 

// CHECK: vmovapd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x28,0x7a,0x40]       
vmovapd 64(%rdx), %xmm15 

// CHECK: vmovapd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x28,0x72,0x40]       
vmovapd 64(%rdx), %xmm6 

// CHECK: vmovapd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x28,0x7a,0x40]       
vmovapd 64(%rdx), %ymm7 

// CHECK: vmovapd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x28,0x4a,0x40]       
vmovapd 64(%rdx), %ymm9 

// CHECK: vmovapd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x28,0x3a]       
vmovapd (%rdx), %xmm15 

// CHECK: vmovapd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x28,0x32]       
vmovapd (%rdx), %xmm6 

// CHECK: vmovapd (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x28,0x3a]       
vmovapd (%rdx), %ymm7 

// CHECK: vmovapd (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x28,0x0a]       
vmovapd (%rdx), %ymm9 

// CHECK: vmovapd %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0x29,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd %xmm15, 485498096 

// CHECK: vmovapd %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0x29,0x7a,0x40]       
vmovapd %xmm15, 64(%rdx) 

// CHECK: vmovapd %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0x29,0x7c,0x02,0x40]       
vmovapd %xmm15, 64(%rdx,%rax) 

// CHECK: vmovapd %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x29,0x7c,0x82,0xc0]       
vmovapd %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovapd %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x29,0x7c,0x82,0x40]       
vmovapd %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovapd %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0x29,0x3a]       
vmovapd %xmm15, (%rdx) 

// CHECK: vmovapd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x28,0xff]       
vmovapd %xmm15, %xmm15 

// CHECK: vmovapd %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x29,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd %xmm6, 485498096 

// CHECK: vmovapd %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x72,0x40]       
vmovapd %xmm6, 64(%rdx) 

// CHECK: vmovapd %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x74,0x02,0x40]       
vmovapd %xmm6, 64(%rdx,%rax) 

// CHECK: vmovapd %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x74,0x82,0xc0]       
vmovapd %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovapd %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x74,0x82,0x40]       
vmovapd %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovapd %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x29,0x32]       
vmovapd %xmm6, (%rdx) 

// CHECK: vmovapd %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x28,0xf6]       
vmovapd %xmm6, %xmm6 

// CHECK: vmovapd %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x29,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd %ymm7, 485498096 

// CHECK: vmovapd %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x29,0x7a,0x40]       
vmovapd %ymm7, 64(%rdx) 

// CHECK: vmovapd %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfd,0x29,0x7c,0x02,0x40]       
vmovapd %ymm7, 64(%rdx,%rax) 

// CHECK: vmovapd %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x29,0x7c,0x82,0xc0]       
vmovapd %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovapd %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x29,0x7c,0x82,0x40]       
vmovapd %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovapd %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x29,0x3a]       
vmovapd %ymm7, (%rdx) 

// CHECK: vmovapd %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x28,0xff]       
vmovapd %ymm7, %ymm7 

// CHECK: vmovapd %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7d,0x29,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovapd %ymm9, 485498096 

// CHECK: vmovapd %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x29,0x4a,0x40]       
vmovapd %ymm9, 64(%rdx) 

// CHECK: vmovapd %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7d,0x29,0x4c,0x02,0x40]       
vmovapd %ymm9, 64(%rdx,%rax) 

// CHECK: vmovapd %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x29,0x4c,0x82,0xc0]       
vmovapd %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovapd %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x29,0x4c,0x82,0x40]       
vmovapd %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovapd %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x29,0x0a]       
vmovapd %ymm9, (%rdx) 

// CHECK: vmovapd %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7d,0x28,0xc9]       
vmovapd %ymm9, %ymm9 

// CHECK: vmovaps 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x28,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096, %xmm15 

// CHECK: vmovaps 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x28,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096, %xmm6 

// CHECK: vmovaps 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x28,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096, %ymm7 

// CHECK: vmovaps 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x28,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps 485498096, %ymm9 

// CHECK: vmovaps -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x28,0x7c,0x82,0xc0]       
vmovaps -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovaps 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x28,0x7c,0x82,0x40]       
vmovaps 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovaps -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x28,0x74,0x82,0xc0]       
vmovaps -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovaps 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x28,0x74,0x82,0x40]       
vmovaps 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovaps -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x28,0x7c,0x82,0xc0]       
vmovaps -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovaps 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x28,0x7c,0x82,0x40]       
vmovaps 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovaps -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x28,0x4c,0x82,0xc0]       
vmovaps -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovaps 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x28,0x4c,0x82,0x40]       
vmovaps 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovaps 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x28,0x7c,0x02,0x40]       
vmovaps 64(%rdx,%rax), %xmm15 

// CHECK: vmovaps 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x28,0x74,0x02,0x40]       
vmovaps 64(%rdx,%rax), %xmm6 

// CHECK: vmovaps 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x28,0x7c,0x02,0x40]       
vmovaps 64(%rdx,%rax), %ymm7 

// CHECK: vmovaps 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x28,0x4c,0x02,0x40]       
vmovaps 64(%rdx,%rax), %ymm9 

// CHECK: vmovaps 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x28,0x7a,0x40]       
vmovaps 64(%rdx), %xmm15 

// CHECK: vmovaps 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x28,0x72,0x40]       
vmovaps 64(%rdx), %xmm6 

// CHECK: vmovaps 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x28,0x7a,0x40]       
vmovaps 64(%rdx), %ymm7 

// CHECK: vmovaps 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x28,0x4a,0x40]       
vmovaps 64(%rdx), %ymm9 

// CHECK: vmovaps (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x28,0x3a]       
vmovaps (%rdx), %xmm15 

// CHECK: vmovaps (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x28,0x32]       
vmovaps (%rdx), %xmm6 

// CHECK: vmovaps (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x28,0x3a]       
vmovaps (%rdx), %ymm7 

// CHECK: vmovaps (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x28,0x0a]       
vmovaps (%rdx), %ymm9 

// CHECK: vmovaps %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x78,0x29,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps %xmm15, 485498096 

// CHECK: vmovaps %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x78,0x29,0x7a,0x40]       
vmovaps %xmm15, 64(%rdx) 

// CHECK: vmovaps %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x78,0x29,0x7c,0x02,0x40]       
vmovaps %xmm15, 64(%rdx,%rax) 

// CHECK: vmovaps %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x29,0x7c,0x82,0xc0]       
vmovaps %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovaps %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x29,0x7c,0x82,0x40]       
vmovaps %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovaps %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x78,0x29,0x3a]       
vmovaps %xmm15, (%rdx) 

// CHECK: vmovaps %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x28,0xff]       
vmovaps %xmm15, %xmm15 

// CHECK: vmovaps %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x29,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps %xmm6, 485498096 

// CHECK: vmovaps %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x72,0x40]       
vmovaps %xmm6, 64(%rdx) 

// CHECK: vmovaps %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x74,0x02,0x40]       
vmovaps %xmm6, 64(%rdx,%rax) 

// CHECK: vmovaps %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x74,0x82,0xc0]       
vmovaps %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovaps %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x74,0x82,0x40]       
vmovaps %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovaps %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x29,0x32]       
vmovaps %xmm6, (%rdx) 

// CHECK: vmovaps %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x28,0xf6]       
vmovaps %xmm6, %xmm6 

// CHECK: vmovaps %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfc,0x29,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps %ymm7, 485498096 

// CHECK: vmovaps %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfc,0x29,0x7a,0x40]       
vmovaps %ymm7, 64(%rdx) 

// CHECK: vmovaps %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfc,0x29,0x7c,0x02,0x40]       
vmovaps %ymm7, 64(%rdx,%rax) 

// CHECK: vmovaps %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfc,0x29,0x7c,0x82,0xc0]       
vmovaps %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovaps %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfc,0x29,0x7c,0x82,0x40]       
vmovaps %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovaps %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfc,0x29,0x3a]       
vmovaps %ymm7, (%rdx) 

// CHECK: vmovaps %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x28,0xff]       
vmovaps %ymm7, %ymm7 

// CHECK: vmovaps %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7c,0x29,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovaps %ymm9, 485498096 

// CHECK: vmovaps %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7c,0x29,0x4a,0x40]       
vmovaps %ymm9, 64(%rdx) 

// CHECK: vmovaps %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7c,0x29,0x4c,0x02,0x40]       
vmovaps %ymm9, 64(%rdx,%rax) 

// CHECK: vmovaps %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7c,0x29,0x4c,0x82,0xc0]       
vmovaps %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovaps %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7c,0x29,0x4c,0x82,0x40]       
vmovaps %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovaps %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7c,0x29,0x0a]       
vmovaps %ymm9, (%rdx) 

// CHECK: vmovaps %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7c,0x28,0xc9]       
vmovaps %ymm9, %ymm9 

// CHECK: vmovd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovd 485498096, %xmm15 

// CHECK: vmovd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovd 485498096, %xmm6 

// CHECK: vmovd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6e,0x7c,0x82,0xc0]       
vmovd -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6e,0x7c,0x82,0x40]       
vmovd 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x74,0x82,0xc0]       
vmovd -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x74,0x82,0x40]       
vmovd 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6e,0x7c,0x02,0x40]       
vmovd 64(%rdx,%rax), %xmm15 

// CHECK: vmovd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x74,0x02,0x40]       
vmovd 64(%rdx,%rax), %xmm6 

// CHECK: vmovd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6e,0x7a,0x40]       
vmovd 64(%rdx), %xmm15 

// CHECK: vmovd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x72,0x40]       
vmovd 64(%rdx), %xmm6 

// CHECK: vmovddup 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x12,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096, %xmm15 

// CHECK: vmovddup 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096, %xmm6 

// CHECK: vmovddup 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xff,0x12,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096, %ymm7 

// CHECK: vmovddup 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x12,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovddup 485498096, %ymm9 

// CHECK: vmovddup -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x12,0x7c,0x82,0xc0]       
vmovddup -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovddup 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x12,0x7c,0x82,0x40]       
vmovddup 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovddup -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x12,0x74,0x82,0xc0]       
vmovddup -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovddup 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x12,0x74,0x82,0x40]       
vmovddup 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovddup -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x12,0x7c,0x82,0xc0]       
vmovddup -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovddup 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x12,0x7c,0x82,0x40]       
vmovddup 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovddup -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x12,0x4c,0x82,0xc0]       
vmovddup -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovddup 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x12,0x4c,0x82,0x40]       
vmovddup 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovddup 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x12,0x7c,0x02,0x40]       
vmovddup 64(%rdx,%rax), %xmm15 

// CHECK: vmovddup 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x12,0x74,0x02,0x40]       
vmovddup 64(%rdx,%rax), %xmm6 

// CHECK: vmovddup 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x12,0x7c,0x02,0x40]       
vmovddup 64(%rdx,%rax), %ymm7 

// CHECK: vmovddup 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x12,0x4c,0x02,0x40]       
vmovddup 64(%rdx,%rax), %ymm9 

// CHECK: vmovddup 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x12,0x7a,0x40]       
vmovddup 64(%rdx), %xmm15 

// CHECK: vmovddup 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x12,0x72,0x40]       
vmovddup 64(%rdx), %xmm6 

// CHECK: vmovddup 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x12,0x7a,0x40]       
vmovddup 64(%rdx), %ymm7 

// CHECK: vmovddup 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x12,0x4a,0x40]       
vmovddup 64(%rdx), %ymm9 

// CHECK: vmovddup (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x12,0x3a]       
vmovddup (%rdx), %xmm15 

// CHECK: vmovddup (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x12,0x32]       
vmovddup (%rdx), %xmm6 

// CHECK: vmovddup (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x12,0x3a]       
vmovddup (%rdx), %ymm7 

// CHECK: vmovddup (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x12,0x0a]       
vmovddup (%rdx), %ymm9 

// CHECK: vmovddup %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7b,0x12,0xff]       
vmovddup %xmm15, %xmm15 

// CHECK: vmovddup %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x12,0xf6]       
vmovddup %xmm6, %xmm6 

// CHECK: vmovddup %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xff,0x12,0xff]       
vmovddup %ymm7, %ymm7 

// CHECK: vmovddup %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7f,0x12,0xc9]       
vmovddup %ymm9, %ymm9 

// CHECK: vmovdqa 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096, %xmm15 

// CHECK: vmovdqa 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096, %xmm6 

// CHECK: vmovdqa 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096, %ymm7 

// CHECK: vmovdqa 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x6f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa 485498096, %ymm9 

// CHECK: vmovdqa -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6f,0x7c,0x82,0xc0]       
vmovdqa -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovdqa 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6f,0x7c,0x82,0x40]       
vmovdqa 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovdqa -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x74,0x82,0xc0]       
vmovdqa -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovdqa 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x74,0x82,0x40]       
vmovdqa 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovdqa -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x7c,0x82,0xc0]       
vmovdqa -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovdqa 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x7c,0x82,0x40]       
vmovdqa 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovdqa -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x6f,0x4c,0x82,0xc0]       
vmovdqa -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovdqa 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x6f,0x4c,0x82,0x40]       
vmovdqa 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovdqa 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6f,0x7c,0x02,0x40]       
vmovdqa 64(%rdx,%rax), %xmm15 

// CHECK: vmovdqa 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x74,0x02,0x40]       
vmovdqa 64(%rdx,%rax), %xmm6 

// CHECK: vmovdqa 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x7c,0x02,0x40]       
vmovdqa 64(%rdx,%rax), %ymm7 

// CHECK: vmovdqa 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x6f,0x4c,0x02,0x40]       
vmovdqa 64(%rdx,%rax), %ymm9 

// CHECK: vmovdqa 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6f,0x7a,0x40]       
vmovdqa 64(%rdx), %xmm15 

// CHECK: vmovdqa 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x72,0x40]       
vmovdqa 64(%rdx), %xmm6 

// CHECK: vmovdqa 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x7a,0x40]       
vmovdqa 64(%rdx), %ymm7 

// CHECK: vmovdqa 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x6f,0x4a,0x40]       
vmovdqa 64(%rdx), %ymm9 

// CHECK: vmovdqa (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6f,0x3a]       
vmovdqa (%rdx), %xmm15 

// CHECK: vmovdqa (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6f,0x32]       
vmovdqa (%rdx), %xmm6 

// CHECK: vmovdqa (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x6f,0x3a]       
vmovdqa (%rdx), %ymm7 

// CHECK: vmovdqa (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x6f,0x0a]       
vmovdqa (%rdx), %ymm9 

// CHECK: vmovdqa %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0x7f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %xmm15, 485498096 

// CHECK: vmovdqa %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0x7f,0x7a,0x40]       
vmovdqa %xmm15, 64(%rdx) 

// CHECK: vmovdqa %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0x7f,0x7c,0x02,0x40]       
vmovdqa %xmm15, 64(%rdx,%rax) 

// CHECK: vmovdqa %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x7f,0x7c,0x82,0xc0]       
vmovdqa %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovdqa %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x7f,0x7c,0x82,0x40]       
vmovdqa %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovdqa %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0x7f,0x3a]       
vmovdqa %xmm15, (%rdx) 

// CHECK: vmovdqa %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x6f,0xff]       
vmovdqa %xmm15, %xmm15 

// CHECK: vmovdqa %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %xmm6, 485498096 

// CHECK: vmovdqa %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x72,0x40]       
vmovdqa %xmm6, 64(%rdx) 

// CHECK: vmovdqa %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x74,0x02,0x40]       
vmovdqa %xmm6, 64(%rdx,%rax) 

// CHECK: vmovdqa %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x74,0x82,0xc0]       
vmovdqa %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovdqa %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x74,0x82,0x40]       
vmovdqa %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovdqa %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x7f,0x32]       
vmovdqa %xmm6, (%rdx) 

// CHECK: vmovdqa %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6f,0xf6]       
vmovdqa %xmm6, %xmm6 

// CHECK: vmovdqa %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %ymm7, 485498096 

// CHECK: vmovdqa %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x7a,0x40]       
vmovdqa %ymm7, 64(%rdx) 

// CHECK: vmovdqa %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x7c,0x02,0x40]       
vmovdqa %ymm7, 64(%rdx,%rax) 

// CHECK: vmovdqa %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x7c,0x82,0xc0]       
vmovdqa %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovdqa %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x7c,0x82,0x40]       
vmovdqa %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovdqa %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x7f,0x3a]       
vmovdqa %ymm7, (%rdx) 

// CHECK: vmovdqa %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x6f,0xff]       
vmovdqa %ymm7, %ymm7 

// CHECK: vmovdqa %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7d,0x7f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqa %ymm9, 485498096 

// CHECK: vmovdqa %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x7f,0x4a,0x40]       
vmovdqa %ymm9, 64(%rdx) 

// CHECK: vmovdqa %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7d,0x7f,0x4c,0x02,0x40]       
vmovdqa %ymm9, 64(%rdx,%rax) 

// CHECK: vmovdqa %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x7f,0x4c,0x82,0xc0]       
vmovdqa %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovdqa %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x7f,0x4c,0x82,0x40]       
vmovdqa %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovdqa %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x7f,0x0a]       
vmovdqa %ymm9, (%rdx) 

// CHECK: vmovdqa %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7d,0x6f,0xc9]       
vmovdqa %ymm9, %ymm9 

// CHECK: vmovdqu 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x6f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096, %xmm15 

// CHECK: vmovdqu 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096, %xmm6 

// CHECK: vmovdqu 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096, %ymm7 

// CHECK: vmovdqu 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x6f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu 485498096, %ymm9 

// CHECK: vmovdqu -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x6f,0x7c,0x82,0xc0]       
vmovdqu -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovdqu 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x6f,0x7c,0x82,0x40]       
vmovdqu 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovdqu -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x74,0x82,0xc0]       
vmovdqu -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovdqu 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x74,0x82,0x40]       
vmovdqu 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovdqu -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x7c,0x82,0xc0]       
vmovdqu -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovdqu 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x7c,0x82,0x40]       
vmovdqu 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovdqu -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x6f,0x4c,0x82,0xc0]       
vmovdqu -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovdqu 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x6f,0x4c,0x82,0x40]       
vmovdqu 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovdqu 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x6f,0x7c,0x02,0x40]       
vmovdqu 64(%rdx,%rax), %xmm15 

// CHECK: vmovdqu 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x74,0x02,0x40]       
vmovdqu 64(%rdx,%rax), %xmm6 

// CHECK: vmovdqu 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x7c,0x02,0x40]       
vmovdqu 64(%rdx,%rax), %ymm7 

// CHECK: vmovdqu 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x6f,0x4c,0x02,0x40]       
vmovdqu 64(%rdx,%rax), %ymm9 

// CHECK: vmovdqu 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x6f,0x7a,0x40]       
vmovdqu 64(%rdx), %xmm15 

// CHECK: vmovdqu 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x72,0x40]       
vmovdqu 64(%rdx), %xmm6 

// CHECK: vmovdqu 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x7a,0x40]       
vmovdqu 64(%rdx), %ymm7 

// CHECK: vmovdqu 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x6f,0x4a,0x40]       
vmovdqu 64(%rdx), %ymm9 

// CHECK: vmovdqu (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x6f,0x3a]       
vmovdqu (%rdx), %xmm15 

// CHECK: vmovdqu (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x6f,0x32]       
vmovdqu (%rdx), %xmm6 

// CHECK: vmovdqu (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x6f,0x3a]       
vmovdqu (%rdx), %ymm7 

// CHECK: vmovdqu (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x6f,0x0a]       
vmovdqu (%rdx), %ymm9 

// CHECK: vmovdqu %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x7a,0x7f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %xmm15, 485498096 

// CHECK: vmovdqu %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7a,0x7f,0x7a,0x40]       
vmovdqu %xmm15, 64(%rdx) 

// CHECK: vmovdqu %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7a,0x7f,0x7c,0x02,0x40]       
vmovdqu %xmm15, 64(%rdx,%rax) 

// CHECK: vmovdqu %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7a,0x7f,0x7c,0x82,0xc0]       
vmovdqu %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovdqu %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7a,0x7f,0x7c,0x82,0x40]       
vmovdqu %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovdqu %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x7a,0x7f,0x3a]       
vmovdqu %xmm15, (%rdx) 

// CHECK: vmovdqu %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7a,0x6f,0xff]       
vmovdqu %xmm15, %xmm15 

// CHECK: vmovdqu %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %xmm6, 485498096 

// CHECK: vmovdqu %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x72,0x40]       
vmovdqu %xmm6, 64(%rdx) 

// CHECK: vmovdqu %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x74,0x02,0x40]       
vmovdqu %xmm6, 64(%rdx,%rax) 

// CHECK: vmovdqu %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x74,0x82,0xc0]       
vmovdqu %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovdqu %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x74,0x82,0x40]       
vmovdqu %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovdqu %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xfa,0x7f,0x32]       
vmovdqu %xmm6, (%rdx) 

// CHECK: vmovdqu %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x6f,0xf6]       
vmovdqu %xmm6, %xmm6 

// CHECK: vmovdqu %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %ymm7, 485498096 

// CHECK: vmovdqu %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x7a,0x40]       
vmovdqu %ymm7, 64(%rdx) 

// CHECK: vmovdqu %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x7c,0x02,0x40]       
vmovdqu %ymm7, 64(%rdx,%rax) 

// CHECK: vmovdqu %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x7c,0x82,0xc0]       
vmovdqu %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovdqu %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x7c,0x82,0x40]       
vmovdqu %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovdqu %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfe,0x7f,0x3a]       
vmovdqu %ymm7, (%rdx) 

// CHECK: vmovdqu %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x6f,0xff]       
vmovdqu %ymm7, %ymm7 

// CHECK: vmovdqu %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7e,0x7f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovdqu %ymm9, 485498096 

// CHECK: vmovdqu %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7e,0x7f,0x4a,0x40]       
vmovdqu %ymm9, 64(%rdx) 

// CHECK: vmovdqu %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7e,0x7f,0x4c,0x02,0x40]       
vmovdqu %ymm9, 64(%rdx,%rax) 

// CHECK: vmovdqu %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7e,0x7f,0x4c,0x82,0xc0]       
vmovdqu %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovdqu %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7e,0x7f,0x4c,0x82,0x40]       
vmovdqu %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovdqu %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7e,0x7f,0x0a]       
vmovdqu %ymm9, (%rdx) 

// CHECK: vmovdqu %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7e,0x6f,0xc9]       
vmovdqu %ymm9, %ymm9 

// CHECK: vmovd %r13d, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x6e,0xfd]       
vmovd %r13d, %xmm15 

// CHECK: vmovd %r13d, %xmm6 
// CHECK: encoding: [0xc4,0xc1,0x79,0x6e,0xf5]       
vmovd %r13d, %xmm6 

// CHECK: vmovd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x6e,0x3a]       
vmovd (%rdx), %xmm15 

// CHECK: vmovd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x6e,0x32]       
vmovd (%rdx), %xmm6 

// CHECK: vmovd %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0x7e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovd %xmm15, 485498096 

// CHECK: vmovd %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0x7e,0x7a,0x40]       
vmovd %xmm15, 64(%rdx) 

// CHECK: vmovd %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0x7e,0x7c,0x02,0x40]       
vmovd %xmm15, 64(%rdx,%rax) 

// CHECK: vmovd %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x7e,0x7c,0x82,0xc0]       
vmovd %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovd %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x7e,0x7c,0x82,0x40]       
vmovd %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovd %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x79,0x7e,0xfd]       
vmovd %xmm15, %r13d 

// CHECK: vmovd %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0x7e,0x3a]       
vmovd %xmm15, (%rdx) 

// CHECK: vmovd %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovd %xmm6, 485498096 

// CHECK: vmovd %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x72,0x40]       
vmovd %xmm6, 64(%rdx) 

// CHECK: vmovd %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x74,0x02,0x40]       
vmovd %xmm6, 64(%rdx,%rax) 

// CHECK: vmovd %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x74,0x82,0xc0]       
vmovd %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovd %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x74,0x82,0x40]       
vmovd %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovd %xmm6, %r13d 
// CHECK: encoding: [0xc4,0xc1,0x79,0x7e,0xf5]       
vmovd %xmm6, %r13d 

// CHECK: vmovd %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x7e,0x32]       
vmovd %xmm6, (%rdx) 

// CHECK: vmovhlps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x12,0xff]      
vmovhlps %xmm15, %xmm15, %xmm15 

// CHECK: vmovhlps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x12,0xf6]      
vmovhlps %xmm6, %xmm6, %xmm6 

// CHECK: vmovhpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x16,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovhpd 485498096, %xmm15, %xmm15 

// CHECK: vmovhpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x16,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovhpd 485498096, %xmm6, %xmm6 

// CHECK: vmovhpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x16,0x7c,0x82,0xc0]      
vmovhpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovhpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x16,0x7c,0x82,0x40]      
vmovhpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovhpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x16,0x74,0x82,0xc0]      
vmovhpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovhpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x16,0x74,0x82,0x40]      
vmovhpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovhpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x16,0x7c,0x02,0x40]      
vmovhpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmovhpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x16,0x74,0x02,0x40]      
vmovhpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmovhpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x16,0x7a,0x40]      
vmovhpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vmovhpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x16,0x72,0x40]      
vmovhpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vmovhpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x16,0x3a]      
vmovhpd (%rdx), %xmm15, %xmm15 

// CHECK: vmovhpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x16,0x32]      
vmovhpd (%rdx), %xmm6, %xmm6 

// CHECK: vmovhpd %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0x17,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovhpd %xmm15, 485498096 

// CHECK: vmovhpd %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0x17,0x7a,0x40]       
vmovhpd %xmm15, 64(%rdx) 

// CHECK: vmovhpd %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0x17,0x7c,0x02,0x40]       
vmovhpd %xmm15, 64(%rdx,%rax) 

// CHECK: vmovhpd %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x17,0x7c,0x82,0xc0]       
vmovhpd %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovhpd %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x17,0x7c,0x82,0x40]       
vmovhpd %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovhpd %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0x17,0x3a]       
vmovhpd %xmm15, (%rdx) 

// CHECK: vmovhpd %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x17,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovhpd %xmm6, 485498096 

// CHECK: vmovhpd %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x72,0x40]       
vmovhpd %xmm6, 64(%rdx) 

// CHECK: vmovhpd %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x74,0x02,0x40]       
vmovhpd %xmm6, 64(%rdx,%rax) 

// CHECK: vmovhpd %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x74,0x82,0xc0]       
vmovhpd %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovhpd %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x74,0x82,0x40]       
vmovhpd %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovhpd %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x17,0x32]       
vmovhpd %xmm6, (%rdx) 

// CHECK: vmovhps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x16,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovhps 485498096, %xmm15, %xmm15 

// CHECK: vmovhps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x16,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovhps 485498096, %xmm6, %xmm6 

// CHECK: vmovhps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x16,0x7c,0x82,0xc0]      
vmovhps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovhps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x16,0x7c,0x82,0x40]      
vmovhps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovhps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x16,0x74,0x82,0xc0]      
vmovhps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovhps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x16,0x74,0x82,0x40]      
vmovhps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovhps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x16,0x7c,0x02,0x40]      
vmovhps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmovhps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x16,0x74,0x02,0x40]      
vmovhps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmovhps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x16,0x7a,0x40]      
vmovhps 64(%rdx), %xmm15, %xmm15 

// CHECK: vmovhps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x16,0x72,0x40]      
vmovhps 64(%rdx), %xmm6, %xmm6 

// CHECK: vmovhps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x16,0x3a]      
vmovhps (%rdx), %xmm15, %xmm15 

// CHECK: vmovhps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x16,0x32]      
vmovhps (%rdx), %xmm6, %xmm6 

// CHECK: vmovhps %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x78,0x17,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovhps %xmm15, 485498096 

// CHECK: vmovhps %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x78,0x17,0x7a,0x40]       
vmovhps %xmm15, 64(%rdx) 

// CHECK: vmovhps %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x78,0x17,0x7c,0x02,0x40]       
vmovhps %xmm15, 64(%rdx,%rax) 

// CHECK: vmovhps %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x17,0x7c,0x82,0xc0]       
vmovhps %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovhps %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x17,0x7c,0x82,0x40]       
vmovhps %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovhps %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x78,0x17,0x3a]       
vmovhps %xmm15, (%rdx) 

// CHECK: vmovhps %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x17,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovhps %xmm6, 485498096 

// CHECK: vmovhps %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x72,0x40]       
vmovhps %xmm6, 64(%rdx) 

// CHECK: vmovhps %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x74,0x02,0x40]       
vmovhps %xmm6, 64(%rdx,%rax) 

// CHECK: vmovhps %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x74,0x82,0xc0]       
vmovhps %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovhps %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x74,0x82,0x40]       
vmovhps %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovhps %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x17,0x32]       
vmovhps %xmm6, (%rdx) 

// CHECK: vmovlhps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x16,0xff]      
vmovlhps %xmm15, %xmm15, %xmm15 

// CHECK: vmovlhps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x16,0xf6]      
vmovlhps %xmm6, %xmm6, %xmm6 

// CHECK: vmovlpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x12,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovlpd 485498096, %xmm15, %xmm15 

// CHECK: vmovlpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovlpd 485498096, %xmm6, %xmm6 

// CHECK: vmovlpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x12,0x7c,0x82,0xc0]      
vmovlpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovlpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x12,0x7c,0x82,0x40]      
vmovlpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovlpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x12,0x74,0x82,0xc0]      
vmovlpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovlpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x12,0x74,0x82,0x40]      
vmovlpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovlpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x12,0x7c,0x02,0x40]      
vmovlpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmovlpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x12,0x74,0x02,0x40]      
vmovlpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmovlpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x12,0x7a,0x40]      
vmovlpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vmovlpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x12,0x72,0x40]      
vmovlpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vmovlpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x12,0x3a]      
vmovlpd (%rdx), %xmm15, %xmm15 

// CHECK: vmovlpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x12,0x32]      
vmovlpd (%rdx), %xmm6, %xmm6 

// CHECK: vmovlpd %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0x13,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovlpd %xmm15, 485498096 

// CHECK: vmovlpd %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0x13,0x7a,0x40]       
vmovlpd %xmm15, 64(%rdx) 

// CHECK: vmovlpd %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0x13,0x7c,0x02,0x40]       
vmovlpd %xmm15, 64(%rdx,%rax) 

// CHECK: vmovlpd %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x13,0x7c,0x82,0xc0]       
vmovlpd %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovlpd %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x13,0x7c,0x82,0x40]       
vmovlpd %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovlpd %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0x13,0x3a]       
vmovlpd %xmm15, (%rdx) 

// CHECK: vmovlpd %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x13,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovlpd %xmm6, 485498096 

// CHECK: vmovlpd %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x72,0x40]       
vmovlpd %xmm6, 64(%rdx) 

// CHECK: vmovlpd %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x74,0x02,0x40]       
vmovlpd %xmm6, 64(%rdx,%rax) 

// CHECK: vmovlpd %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x74,0x82,0xc0]       
vmovlpd %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovlpd %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x74,0x82,0x40]       
vmovlpd %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovlpd %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x13,0x32]       
vmovlpd %xmm6, (%rdx) 

// CHECK: vmovlps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x12,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovlps 485498096, %xmm15, %xmm15 

// CHECK: vmovlps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmovlps 485498096, %xmm6, %xmm6 

// CHECK: vmovlps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x12,0x7c,0x82,0xc0]      
vmovlps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovlps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x12,0x7c,0x82,0x40]      
vmovlps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmovlps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x12,0x74,0x82,0xc0]      
vmovlps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovlps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x12,0x74,0x82,0x40]      
vmovlps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmovlps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x12,0x7c,0x02,0x40]      
vmovlps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmovlps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x12,0x74,0x02,0x40]      
vmovlps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmovlps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x12,0x7a,0x40]      
vmovlps 64(%rdx), %xmm15, %xmm15 

// CHECK: vmovlps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x12,0x72,0x40]      
vmovlps 64(%rdx), %xmm6, %xmm6 

// CHECK: vmovlps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x12,0x3a]      
vmovlps (%rdx), %xmm15, %xmm15 

// CHECK: vmovlps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x12,0x32]      
vmovlps (%rdx), %xmm6, %xmm6 

// CHECK: vmovlps %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x78,0x13,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovlps %xmm15, 485498096 

// CHECK: vmovlps %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x78,0x13,0x7a,0x40]       
vmovlps %xmm15, 64(%rdx) 

// CHECK: vmovlps %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x78,0x13,0x7c,0x02,0x40]       
vmovlps %xmm15, 64(%rdx,%rax) 

// CHECK: vmovlps %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x13,0x7c,0x82,0xc0]       
vmovlps %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovlps %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x13,0x7c,0x82,0x40]       
vmovlps %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovlps %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x78,0x13,0x3a]       
vmovlps %xmm15, (%rdx) 

// CHECK: vmovlps %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x13,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovlps %xmm6, 485498096 

// CHECK: vmovlps %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x72,0x40]       
vmovlps %xmm6, 64(%rdx) 

// CHECK: vmovlps %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x74,0x02,0x40]       
vmovlps %xmm6, 64(%rdx,%rax) 

// CHECK: vmovlps %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x74,0x82,0xc0]       
vmovlps %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovlps %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x74,0x82,0x40]       
vmovlps %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovlps %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x13,0x32]       
vmovlps %xmm6, (%rdx) 

// CHECK: vmovmskpd %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x79,0x50,0xef]       
vmovmskpd %xmm15, %r13d 

// CHECK: vmovmskpd %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x79,0x50,0xee]       
vmovmskpd %xmm6, %r13d 

// CHECK: vmovmskpd %ymm7, %r13d 
// CHECK: encoding: [0xc5,0x7d,0x50,0xef]       
vmovmskpd %ymm7, %r13d 

// CHECK: vmovmskpd %ymm9, %r13d 
// CHECK: encoding: [0xc4,0x41,0x7d,0x50,0xe9]       
vmovmskpd %ymm9, %r13d 

// CHECK: vmovmskps %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x78,0x50,0xef]       
vmovmskps %xmm15, %r13d 

// CHECK: vmovmskps %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x78,0x50,0xee]       
vmovmskps %xmm6, %r13d 

// CHECK: vmovmskps %ymm7, %r13d 
// CHECK: encoding: [0xc5,0x7c,0x50,0xef]       
vmovmskps %ymm7, %r13d 

// CHECK: vmovmskps %ymm9, %r13d 
// CHECK: encoding: [0xc4,0x41,0x7c,0x50,0xe9]       
vmovmskps %ymm9, %r13d 

// CHECK: vmovntdqa 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096, %xmm15 

// CHECK: vmovntdqa 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096, %xmm6 

// CHECK: vmovntdqa -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x2a,0x7c,0x82,0xc0]       
vmovntdqa -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovntdqa 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x2a,0x7c,0x82,0x40]       
vmovntdqa 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovntdqa -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x74,0x82,0xc0]       
vmovntdqa -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovntdqa 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x74,0x82,0x40]       
vmovntdqa 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovntdqa 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x2a,0x7c,0x02,0x40]       
vmovntdqa 64(%rdx,%rax), %xmm15 

// CHECK: vmovntdqa 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x74,0x02,0x40]       
vmovntdqa 64(%rdx,%rax), %xmm6 

// CHECK: vmovntdqa 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x2a,0x7a,0x40]       
vmovntdqa 64(%rdx), %xmm15 

// CHECK: vmovntdqa 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x72,0x40]       
vmovntdqa 64(%rdx), %xmm6 

// CHECK: vmovntdqa (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x2a,0x3a]       
vmovntdqa (%rdx), %xmm15 

// CHECK: vmovntdqa (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x32]       
vmovntdqa (%rdx), %xmm6 

// CHECK: vmovntdq %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0xe7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %xmm15, 485498096 

// CHECK: vmovntdq %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0xe7,0x7a,0x40]       
vmovntdq %xmm15, 64(%rdx) 

// CHECK: vmovntdq %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0xe7,0x7c,0x02,0x40]       
vmovntdq %xmm15, 64(%rdx,%rax) 

// CHECK: vmovntdq %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0xe7,0x7c,0x82,0xc0]       
vmovntdq %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovntdq %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0xe7,0x7c,0x82,0x40]       
vmovntdq %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovntdq %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0xe7,0x3a]       
vmovntdq %xmm15, (%rdx) 

// CHECK: vmovntdq %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %xmm6, 485498096 

// CHECK: vmovntdq %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x72,0x40]       
vmovntdq %xmm6, 64(%rdx) 

// CHECK: vmovntdq %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x74,0x02,0x40]       
vmovntdq %xmm6, 64(%rdx,%rax) 

// CHECK: vmovntdq %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x74,0x82,0xc0]       
vmovntdq %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovntdq %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x74,0x82,0x40]       
vmovntdq %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovntdq %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0xe7,0x32]       
vmovntdq %xmm6, (%rdx) 

// CHECK: vmovntdq %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %ymm7, 485498096 

// CHECK: vmovntdq %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x7a,0x40]       
vmovntdq %ymm7, 64(%rdx) 

// CHECK: vmovntdq %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x7c,0x02,0x40]       
vmovntdq %ymm7, 64(%rdx,%rax) 

// CHECK: vmovntdq %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x7c,0x82,0xc0]       
vmovntdq %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovntdq %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x7c,0x82,0x40]       
vmovntdq %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovntdq %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfd,0xe7,0x3a]       
vmovntdq %ymm7, (%rdx) 

// CHECK: vmovntdq %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7d,0xe7,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdq %ymm9, 485498096 

// CHECK: vmovntdq %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7d,0xe7,0x4a,0x40]       
vmovntdq %ymm9, 64(%rdx) 

// CHECK: vmovntdq %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7d,0xe7,0x4c,0x02,0x40]       
vmovntdq %ymm9, 64(%rdx,%rax) 

// CHECK: vmovntdq %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0xe7,0x4c,0x82,0xc0]       
vmovntdq %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovntdq %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0xe7,0x4c,0x82,0x40]       
vmovntdq %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovntdq %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7d,0xe7,0x0a]       
vmovntdq %ymm9, (%rdx) 

// CHECK: vmovntpd %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0x2b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %xmm15, 485498096 

// CHECK: vmovntpd %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0x2b,0x7a,0x40]       
vmovntpd %xmm15, 64(%rdx) 

// CHECK: vmovntpd %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0x2b,0x7c,0x02,0x40]       
vmovntpd %xmm15, 64(%rdx,%rax) 

// CHECK: vmovntpd %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x2b,0x7c,0x82,0xc0]       
vmovntpd %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovntpd %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x2b,0x7c,0x82,0x40]       
vmovntpd %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovntpd %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0x2b,0x3a]       
vmovntpd %xmm15, (%rdx) 

// CHECK: vmovntpd %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %xmm6, 485498096 

// CHECK: vmovntpd %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x72,0x40]       
vmovntpd %xmm6, 64(%rdx) 

// CHECK: vmovntpd %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x74,0x02,0x40]       
vmovntpd %xmm6, 64(%rdx,%rax) 

// CHECK: vmovntpd %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x74,0x82,0xc0]       
vmovntpd %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovntpd %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x74,0x82,0x40]       
vmovntpd %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovntpd %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x2b,0x32]       
vmovntpd %xmm6, (%rdx) 

// CHECK: vmovntpd %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %ymm7, 485498096 

// CHECK: vmovntpd %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x7a,0x40]       
vmovntpd %ymm7, 64(%rdx) 

// CHECK: vmovntpd %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x7c,0x02,0x40]       
vmovntpd %ymm7, 64(%rdx,%rax) 

// CHECK: vmovntpd %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x7c,0x82,0xc0]       
vmovntpd %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovntpd %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x7c,0x82,0x40]       
vmovntpd %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovntpd %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x2b,0x3a]       
vmovntpd %ymm7, (%rdx) 

// CHECK: vmovntpd %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7d,0x2b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntpd %ymm9, 485498096 

// CHECK: vmovntpd %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x2b,0x4a,0x40]       
vmovntpd %ymm9, 64(%rdx) 

// CHECK: vmovntpd %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7d,0x2b,0x4c,0x02,0x40]       
vmovntpd %ymm9, 64(%rdx,%rax) 

// CHECK: vmovntpd %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x2b,0x4c,0x82,0xc0]       
vmovntpd %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovntpd %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x2b,0x4c,0x82,0x40]       
vmovntpd %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovntpd %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x2b,0x0a]       
vmovntpd %ymm9, (%rdx) 

// CHECK: vmovntps %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x78,0x2b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntps %xmm15, 485498096 

// CHECK: vmovntps %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x78,0x2b,0x7a,0x40]       
vmovntps %xmm15, 64(%rdx) 

// CHECK: vmovntps %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x78,0x2b,0x7c,0x02,0x40]       
vmovntps %xmm15, 64(%rdx,%rax) 

// CHECK: vmovntps %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x2b,0x7c,0x82,0xc0]       
vmovntps %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovntps %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x2b,0x7c,0x82,0x40]       
vmovntps %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovntps %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x78,0x2b,0x3a]       
vmovntps %xmm15, (%rdx) 

// CHECK: vmovntps %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntps %xmm6, 485498096 

// CHECK: vmovntps %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x72,0x40]       
vmovntps %xmm6, 64(%rdx) 

// CHECK: vmovntps %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x74,0x02,0x40]       
vmovntps %xmm6, 64(%rdx,%rax) 

// CHECK: vmovntps %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x74,0x82,0xc0]       
vmovntps %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovntps %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x74,0x82,0x40]       
vmovntps %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovntps %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x2b,0x32]       
vmovntps %xmm6, (%rdx) 

// CHECK: vmovntps %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntps %ymm7, 485498096 

// CHECK: vmovntps %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x7a,0x40]       
vmovntps %ymm7, 64(%rdx) 

// CHECK: vmovntps %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x7c,0x02,0x40]       
vmovntps %ymm7, 64(%rdx,%rax) 

// CHECK: vmovntps %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x7c,0x82,0xc0]       
vmovntps %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovntps %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x7c,0x82,0x40]       
vmovntps %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovntps %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfc,0x2b,0x3a]       
vmovntps %ymm7, (%rdx) 

// CHECK: vmovntps %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7c,0x2b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntps %ymm9, 485498096 

// CHECK: vmovntps %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7c,0x2b,0x4a,0x40]       
vmovntps %ymm9, 64(%rdx) 

// CHECK: vmovntps %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7c,0x2b,0x4c,0x02,0x40]       
vmovntps %ymm9, 64(%rdx,%rax) 

// CHECK: vmovntps %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7c,0x2b,0x4c,0x82,0xc0]       
vmovntps %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovntps %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7c,0x2b,0x4c,0x82,0x40]       
vmovntps %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovntps %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7c,0x2b,0x0a]       
vmovntps %ymm9, (%rdx) 

// CHECK: vmovq 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x7e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovq 485498096, %xmm15 

// CHECK: vmovq 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovq 485498096, %xmm6 

// CHECK: vmovq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x7e,0x7c,0x82,0xc0]       
vmovq -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x7e,0x7c,0x82,0x40]       
vmovq 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x74,0x82,0xc0]       
vmovq -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x74,0x82,0x40]       
vmovq 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x7e,0x7c,0x02,0x40]       
vmovq 64(%rdx,%rax), %xmm15 

// CHECK: vmovq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x74,0x02,0x40]       
vmovq 64(%rdx,%rax), %xmm6 

// CHECK: vmovq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x7e,0x7a,0x40]       
vmovq 64(%rdx), %xmm15 

// CHECK: vmovq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x72,0x40]       
vmovq 64(%rdx), %xmm6 

// CHECK: vmovq %r15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0xf9,0x6e,0xff]       
vmovq %r15, %xmm15 

// CHECK: vmovq %r15, %xmm6 
// CHECK: encoding: [0xc4,0xc1,0xf9,0x6e,0xf7]       
vmovq %r15, %xmm6 

// CHECK: vmovq (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x7e,0x3a]       
vmovq (%rdx), %xmm15 

// CHECK: vmovq (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x7e,0x32]       
vmovq (%rdx), %xmm6 

// CHECK: vmovq %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0xd6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovq %xmm15, 485498096 

// CHECK: vmovq %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0xd6,0x7a,0x40]       
vmovq %xmm15, 64(%rdx) 

// CHECK: vmovq %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0xd6,0x7c,0x02,0x40]       
vmovq %xmm15, 64(%rdx,%rax) 

// CHECK: vmovq %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0xd6,0x7c,0x82,0xc0]       
vmovq %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovq %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0xd6,0x7c,0x82,0x40]       
vmovq %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovq %xmm15, %r15 
// CHECK: encoding: [0xc4,0x41,0xf9,0x7e,0xff]       
vmovq %xmm15, %r15 

// CHECK: vmovq %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0xd6,0x3a]       
vmovq %xmm15, (%rdx) 

// CHECK: vmovq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7a,0x7e,0xff]       
vmovq %xmm15, %xmm15 

// CHECK: vmovq %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovq %xmm6, 485498096 

// CHECK: vmovq %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x72,0x40]       
vmovq %xmm6, 64(%rdx) 

// CHECK: vmovq %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x74,0x02,0x40]       
vmovq %xmm6, 64(%rdx,%rax) 

// CHECK: vmovq %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x74,0x82,0xc0]       
vmovq %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovq %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x74,0x82,0x40]       
vmovq %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovq %xmm6, %r15 
// CHECK: encoding: [0xc4,0xc1,0xf9,0x7e,0xf7]       
vmovq %xmm6, %r15 

// CHECK: vmovq %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0xd6,0x32]       
vmovq %xmm6, (%rdx) 

// CHECK: vmovq %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x7e,0xf6]       
vmovq %xmm6, %xmm6 

// CHECK: vmovsd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsd 485498096, %xmm15 

// CHECK: vmovsd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x10,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsd 485498096, %xmm6 

// CHECK: vmovsd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x10,0x7c,0x82,0xc0]       
vmovsd -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovsd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x10,0x7c,0x82,0x40]       
vmovsd 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovsd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x10,0x74,0x82,0xc0]       
vmovsd -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovsd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x10,0x74,0x82,0x40]       
vmovsd 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovsd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x10,0x7c,0x02,0x40]       
vmovsd 64(%rdx,%rax), %xmm15 

// CHECK: vmovsd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x10,0x74,0x02,0x40]       
vmovsd 64(%rdx,%rax), %xmm6 

// CHECK: vmovsd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x10,0x7a,0x40]       
vmovsd 64(%rdx), %xmm15 

// CHECK: vmovsd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x10,0x72,0x40]       
vmovsd 64(%rdx), %xmm6 

// CHECK: vmovsd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x10,0x3a]       
vmovsd (%rdx), %xmm15 

// CHECK: vmovsd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x10,0x32]       
vmovsd (%rdx), %xmm6 

// CHECK: vmovsd %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x7b,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsd %xmm15, 485498096 

// CHECK: vmovsd %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7b,0x11,0x7a,0x40]       
vmovsd %xmm15, 64(%rdx) 

// CHECK: vmovsd %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7b,0x11,0x7c,0x02,0x40]       
vmovsd %xmm15, 64(%rdx,%rax) 

// CHECK: vmovsd %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7b,0x11,0x7c,0x82,0xc0]       
vmovsd %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovsd %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7b,0x11,0x7c,0x82,0x40]       
vmovsd %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovsd %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x7b,0x11,0x3a]       
vmovsd %xmm15, (%rdx) 

// CHECK: vmovsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x10,0xff]      
vmovsd %xmm15, %xmm15, %xmm15 

// CHECK: vmovsd %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xfb,0x11,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsd %xmm6, 485498096 

// CHECK: vmovsd %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x72,0x40]       
vmovsd %xmm6, 64(%rdx) 

// CHECK: vmovsd %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x74,0x02,0x40]       
vmovsd %xmm6, 64(%rdx,%rax) 

// CHECK: vmovsd %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x74,0x82,0xc0]       
vmovsd %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovsd %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x74,0x82,0x40]       
vmovsd %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovsd %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xfb,0x11,0x32]       
vmovsd %xmm6, (%rdx) 

// CHECK: vmovsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x10,0xf6]      
vmovsd %xmm6, %xmm6, %xmm6 

// CHECK: vmovshdup 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x16,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096, %xmm15 

// CHECK: vmovshdup 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x16,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096, %xmm6 

// CHECK: vmovshdup 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x16,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096, %ymm7 

// CHECK: vmovshdup 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x16,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovshdup 485498096, %ymm9 

// CHECK: vmovshdup -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x16,0x7c,0x82,0xc0]       
vmovshdup -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovshdup 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x16,0x7c,0x82,0x40]       
vmovshdup 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovshdup -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x16,0x74,0x82,0xc0]       
vmovshdup -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovshdup 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x16,0x74,0x82,0x40]       
vmovshdup 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovshdup -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x16,0x7c,0x82,0xc0]       
vmovshdup -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovshdup 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x16,0x7c,0x82,0x40]       
vmovshdup 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovshdup -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x16,0x4c,0x82,0xc0]       
vmovshdup -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovshdup 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x16,0x4c,0x82,0x40]       
vmovshdup 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovshdup 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x16,0x7c,0x02,0x40]       
vmovshdup 64(%rdx,%rax), %xmm15 

// CHECK: vmovshdup 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x16,0x74,0x02,0x40]       
vmovshdup 64(%rdx,%rax), %xmm6 

// CHECK: vmovshdup 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x16,0x7c,0x02,0x40]       
vmovshdup 64(%rdx,%rax), %ymm7 

// CHECK: vmovshdup 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x16,0x4c,0x02,0x40]       
vmovshdup 64(%rdx,%rax), %ymm9 

// CHECK: vmovshdup 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x16,0x7a,0x40]       
vmovshdup 64(%rdx), %xmm15 

// CHECK: vmovshdup 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x16,0x72,0x40]       
vmovshdup 64(%rdx), %xmm6 

// CHECK: vmovshdup 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x16,0x7a,0x40]       
vmovshdup 64(%rdx), %ymm7 

// CHECK: vmovshdup 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x16,0x4a,0x40]       
vmovshdup 64(%rdx), %ymm9 

// CHECK: vmovshdup (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x16,0x3a]       
vmovshdup (%rdx), %xmm15 

// CHECK: vmovshdup (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x16,0x32]       
vmovshdup (%rdx), %xmm6 

// CHECK: vmovshdup (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x16,0x3a]       
vmovshdup (%rdx), %ymm7 

// CHECK: vmovshdup (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x16,0x0a]       
vmovshdup (%rdx), %ymm9 

// CHECK: vmovshdup %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7a,0x16,0xff]       
vmovshdup %xmm15, %xmm15 

// CHECK: vmovshdup %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x16,0xf6]       
vmovshdup %xmm6, %xmm6 

// CHECK: vmovshdup %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x16,0xff]       
vmovshdup %ymm7, %ymm7 

// CHECK: vmovshdup %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7e,0x16,0xc9]       
vmovshdup %ymm9, %ymm9 

// CHECK: vmovsldup 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x12,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096, %xmm15 

// CHECK: vmovsldup 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096, %xmm6 

// CHECK: vmovsldup 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x12,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096, %ymm7 

// CHECK: vmovsldup 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x12,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovsldup 485498096, %ymm9 

// CHECK: vmovsldup -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x12,0x7c,0x82,0xc0]       
vmovsldup -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovsldup 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x12,0x7c,0x82,0x40]       
vmovsldup 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovsldup -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x12,0x74,0x82,0xc0]       
vmovsldup -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovsldup 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x12,0x74,0x82,0x40]       
vmovsldup 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovsldup -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x12,0x7c,0x82,0xc0]       
vmovsldup -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovsldup 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x12,0x7c,0x82,0x40]       
vmovsldup 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovsldup -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x12,0x4c,0x82,0xc0]       
vmovsldup -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovsldup 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x12,0x4c,0x82,0x40]       
vmovsldup 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovsldup 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x12,0x7c,0x02,0x40]       
vmovsldup 64(%rdx,%rax), %xmm15 

// CHECK: vmovsldup 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x12,0x74,0x02,0x40]       
vmovsldup 64(%rdx,%rax), %xmm6 

// CHECK: vmovsldup 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x12,0x7c,0x02,0x40]       
vmovsldup 64(%rdx,%rax), %ymm7 

// CHECK: vmovsldup 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x12,0x4c,0x02,0x40]       
vmovsldup 64(%rdx,%rax), %ymm9 

// CHECK: vmovsldup 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x12,0x7a,0x40]       
vmovsldup 64(%rdx), %xmm15 

// CHECK: vmovsldup 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x12,0x72,0x40]       
vmovsldup 64(%rdx), %xmm6 

// CHECK: vmovsldup 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x12,0x7a,0x40]       
vmovsldup 64(%rdx), %ymm7 

// CHECK: vmovsldup 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x12,0x4a,0x40]       
vmovsldup 64(%rdx), %ymm9 

// CHECK: vmovsldup (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x12,0x3a]       
vmovsldup (%rdx), %xmm15 

// CHECK: vmovsldup (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x12,0x32]       
vmovsldup (%rdx), %xmm6 

// CHECK: vmovsldup (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x12,0x3a]       
vmovsldup (%rdx), %ymm7 

// CHECK: vmovsldup (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x12,0x0a]       
vmovsldup (%rdx), %ymm9 

// CHECK: vmovsldup %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7a,0x12,0xff]       
vmovsldup %xmm15, %xmm15 

// CHECK: vmovsldup %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x12,0xf6]       
vmovsldup %xmm6, %xmm6 

// CHECK: vmovsldup %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x12,0xff]       
vmovsldup %ymm7, %ymm7 

// CHECK: vmovsldup %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7e,0x12,0xc9]       
vmovsldup %ymm9, %ymm9 

// CHECK: vmovss 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovss 485498096, %xmm15 

// CHECK: vmovss 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x10,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovss 485498096, %xmm6 

// CHECK: vmovss -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x10,0x7c,0x82,0xc0]       
vmovss -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovss 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x10,0x7c,0x82,0x40]       
vmovss 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovss -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x10,0x74,0x82,0xc0]       
vmovss -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovss 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x10,0x74,0x82,0x40]       
vmovss 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovss 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x10,0x7c,0x02,0x40]       
vmovss 64(%rdx,%rax), %xmm15 

// CHECK: vmovss 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x10,0x74,0x02,0x40]       
vmovss 64(%rdx,%rax), %xmm6 

// CHECK: vmovss 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x10,0x7a,0x40]       
vmovss 64(%rdx), %xmm15 

// CHECK: vmovss 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x10,0x72,0x40]       
vmovss 64(%rdx), %xmm6 

// CHECK: vmovss (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x10,0x3a]       
vmovss (%rdx), %xmm15 

// CHECK: vmovss (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x10,0x32]       
vmovss (%rdx), %xmm6 

// CHECK: vmovss %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x7a,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovss %xmm15, 485498096 

// CHECK: vmovss %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7a,0x11,0x7a,0x40]       
vmovss %xmm15, 64(%rdx) 

// CHECK: vmovss %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7a,0x11,0x7c,0x02,0x40]       
vmovss %xmm15, 64(%rdx,%rax) 

// CHECK: vmovss %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7a,0x11,0x7c,0x82,0xc0]       
vmovss %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovss %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7a,0x11,0x7c,0x82,0x40]       
vmovss %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovss %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x7a,0x11,0x3a]       
vmovss %xmm15, (%rdx) 

// CHECK: vmovss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x10,0xff]      
vmovss %xmm15, %xmm15, %xmm15 

// CHECK: vmovss %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xfa,0x11,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovss %xmm6, 485498096 

// CHECK: vmovss %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x72,0x40]       
vmovss %xmm6, 64(%rdx) 

// CHECK: vmovss %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x74,0x02,0x40]       
vmovss %xmm6, 64(%rdx,%rax) 

// CHECK: vmovss %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x74,0x82,0xc0]       
vmovss %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovss %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x74,0x82,0x40]       
vmovss %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovss %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xfa,0x11,0x32]       
vmovss %xmm6, (%rdx) 

// CHECK: vmovss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x10,0xf6]      
vmovss %xmm6, %xmm6, %xmm6 

// CHECK: vmovupd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096, %xmm15 

// CHECK: vmovupd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x10,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096, %xmm6 

// CHECK: vmovupd 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096, %ymm7 

// CHECK: vmovupd 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd 485498096, %ymm9 

// CHECK: vmovupd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x10,0x7c,0x82,0xc0]       
vmovupd -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovupd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x10,0x7c,0x82,0x40]       
vmovupd 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovupd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x10,0x74,0x82,0xc0]       
vmovupd -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovupd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x10,0x74,0x82,0x40]       
vmovupd 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovupd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x10,0x7c,0x82,0xc0]       
vmovupd -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovupd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x10,0x7c,0x82,0x40]       
vmovupd 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovupd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x10,0x4c,0x82,0xc0]       
vmovupd -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovupd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x10,0x4c,0x82,0x40]       
vmovupd 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovupd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x10,0x7c,0x02,0x40]       
vmovupd 64(%rdx,%rax), %xmm15 

// CHECK: vmovupd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x10,0x74,0x02,0x40]       
vmovupd 64(%rdx,%rax), %xmm6 

// CHECK: vmovupd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x10,0x7c,0x02,0x40]       
vmovupd 64(%rdx,%rax), %ymm7 

// CHECK: vmovupd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x10,0x4c,0x02,0x40]       
vmovupd 64(%rdx,%rax), %ymm9 

// CHECK: vmovupd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x10,0x7a,0x40]       
vmovupd 64(%rdx), %xmm15 

// CHECK: vmovupd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x10,0x72,0x40]       
vmovupd 64(%rdx), %xmm6 

// CHECK: vmovupd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x10,0x7a,0x40]       
vmovupd 64(%rdx), %ymm7 

// CHECK: vmovupd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x10,0x4a,0x40]       
vmovupd 64(%rdx), %ymm9 

// CHECK: vmovupd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x10,0x3a]       
vmovupd (%rdx), %xmm15 

// CHECK: vmovupd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x10,0x32]       
vmovupd (%rdx), %xmm6 

// CHECK: vmovupd (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x10,0x3a]       
vmovupd (%rdx), %ymm7 

// CHECK: vmovupd (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x10,0x0a]       
vmovupd (%rdx), %ymm9 

// CHECK: vmovupd %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x79,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd %xmm15, 485498096 

// CHECK: vmovupd %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x79,0x11,0x7a,0x40]       
vmovupd %xmm15, 64(%rdx) 

// CHECK: vmovupd %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x79,0x11,0x7c,0x02,0x40]       
vmovupd %xmm15, 64(%rdx,%rax) 

// CHECK: vmovupd %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x11,0x7c,0x82,0xc0]       
vmovupd %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovupd %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x79,0x11,0x7c,0x82,0x40]       
vmovupd %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovupd %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x79,0x11,0x3a]       
vmovupd %xmm15, (%rdx) 

// CHECK: vmovupd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x10,0xff]       
vmovupd %xmm15, %xmm15 

// CHECK: vmovupd %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf9,0x11,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd %xmm6, 485498096 

// CHECK: vmovupd %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x72,0x40]       
vmovupd %xmm6, 64(%rdx) 

// CHECK: vmovupd %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x74,0x02,0x40]       
vmovupd %xmm6, 64(%rdx,%rax) 

// CHECK: vmovupd %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x74,0x82,0xc0]       
vmovupd %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovupd %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x74,0x82,0x40]       
vmovupd %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovupd %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf9,0x11,0x32]       
vmovupd %xmm6, (%rdx) 

// CHECK: vmovupd %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x10,0xf6]       
vmovupd %xmm6, %xmm6 

// CHECK: vmovupd %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfd,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd %ymm7, 485498096 

// CHECK: vmovupd %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x11,0x7a,0x40]       
vmovupd %ymm7, 64(%rdx) 

// CHECK: vmovupd %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfd,0x11,0x7c,0x02,0x40]       
vmovupd %ymm7, 64(%rdx,%rax) 

// CHECK: vmovupd %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x11,0x7c,0x82,0xc0]       
vmovupd %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovupd %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfd,0x11,0x7c,0x82,0x40]       
vmovupd %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovupd %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfd,0x11,0x3a]       
vmovupd %ymm7, (%rdx) 

// CHECK: vmovupd %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x10,0xff]       
vmovupd %ymm7, %ymm7 

// CHECK: vmovupd %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7d,0x11,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovupd %ymm9, 485498096 

// CHECK: vmovupd %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x11,0x4a,0x40]       
vmovupd %ymm9, 64(%rdx) 

// CHECK: vmovupd %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7d,0x11,0x4c,0x02,0x40]       
vmovupd %ymm9, 64(%rdx,%rax) 

// CHECK: vmovupd %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x11,0x4c,0x82,0xc0]       
vmovupd %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovupd %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7d,0x11,0x4c,0x82,0x40]       
vmovupd %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovupd %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7d,0x11,0x0a]       
vmovupd %ymm9, (%rdx) 

// CHECK: vmovupd %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7d,0x10,0xc9]       
vmovupd %ymm9, %ymm9 

// CHECK: vmovups 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096, %xmm15 

// CHECK: vmovups 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x10,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096, %xmm6 

// CHECK: vmovups 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096, %ymm7 

// CHECK: vmovups 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups 485498096, %ymm9 

// CHECK: vmovups -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x10,0x7c,0x82,0xc0]       
vmovups -64(%rdx,%rax,4), %xmm15 

// CHECK: vmovups 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x10,0x7c,0x82,0x40]       
vmovups 64(%rdx,%rax,4), %xmm15 

// CHECK: vmovups -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x10,0x74,0x82,0xc0]       
vmovups -64(%rdx,%rax,4), %xmm6 

// CHECK: vmovups 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x10,0x74,0x82,0x40]       
vmovups 64(%rdx,%rax,4), %xmm6 

// CHECK: vmovups -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x10,0x7c,0x82,0xc0]       
vmovups -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovups 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x10,0x7c,0x82,0x40]       
vmovups 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovups -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x10,0x4c,0x82,0xc0]       
vmovups -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovups 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x10,0x4c,0x82,0x40]       
vmovups 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovups 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x10,0x7c,0x02,0x40]       
vmovups 64(%rdx,%rax), %xmm15 

// CHECK: vmovups 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x10,0x74,0x02,0x40]       
vmovups 64(%rdx,%rax), %xmm6 

// CHECK: vmovups 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x10,0x7c,0x02,0x40]       
vmovups 64(%rdx,%rax), %ymm7 

// CHECK: vmovups 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x10,0x4c,0x02,0x40]       
vmovups 64(%rdx,%rax), %ymm9 

// CHECK: vmovups 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x10,0x7a,0x40]       
vmovups 64(%rdx), %xmm15 

// CHECK: vmovups 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x10,0x72,0x40]       
vmovups 64(%rdx), %xmm6 

// CHECK: vmovups 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x10,0x7a,0x40]       
vmovups 64(%rdx), %ymm7 

// CHECK: vmovups 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x10,0x4a,0x40]       
vmovups 64(%rdx), %ymm9 

// CHECK: vmovups (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x10,0x3a]       
vmovups (%rdx), %xmm15 

// CHECK: vmovups (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x10,0x32]       
vmovups (%rdx), %xmm6 

// CHECK: vmovups (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x10,0x3a]       
vmovups (%rdx), %ymm7 

// CHECK: vmovups (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x10,0x0a]       
vmovups (%rdx), %ymm9 

// CHECK: vmovups %xmm15, 485498096 
// CHECK: encoding: [0xc5,0x78,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups %xmm15, 485498096 

// CHECK: vmovups %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc5,0x78,0x11,0x7a,0x40]       
vmovups %xmm15, 64(%rdx) 

// CHECK: vmovups %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x78,0x11,0x7c,0x02,0x40]       
vmovups %xmm15, 64(%rdx,%rax) 

// CHECK: vmovups %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x11,0x7c,0x82,0xc0]       
vmovups %xmm15, -64(%rdx,%rax,4) 

// CHECK: vmovups %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x78,0x11,0x7c,0x82,0x40]       
vmovups %xmm15, 64(%rdx,%rax,4) 

// CHECK: vmovups %xmm15, (%rdx) 
// CHECK: encoding: [0xc5,0x78,0x11,0x3a]       
vmovups %xmm15, (%rdx) 

// CHECK: vmovups %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x10,0xff]       
vmovups %xmm15, %xmm15 

// CHECK: vmovups %xmm6, 485498096 
// CHECK: encoding: [0xc5,0xf8,0x11,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups %xmm6, 485498096 

// CHECK: vmovups %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x72,0x40]       
vmovups %xmm6, 64(%rdx) 

// CHECK: vmovups %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x74,0x02,0x40]       
vmovups %xmm6, 64(%rdx,%rax) 

// CHECK: vmovups %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x74,0x82,0xc0]       
vmovups %xmm6, -64(%rdx,%rax,4) 

// CHECK: vmovups %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x74,0x82,0x40]       
vmovups %xmm6, 64(%rdx,%rax,4) 

// CHECK: vmovups %xmm6, (%rdx) 
// CHECK: encoding: [0xc5,0xf8,0x11,0x32]       
vmovups %xmm6, (%rdx) 

// CHECK: vmovups %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x10,0xf6]       
vmovups %xmm6, %xmm6 

// CHECK: vmovups %ymm7, 485498096 
// CHECK: encoding: [0xc5,0xfc,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups %ymm7, 485498096 

// CHECK: vmovups %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc5,0xfc,0x11,0x7a,0x40]       
vmovups %ymm7, 64(%rdx) 

// CHECK: vmovups %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xfc,0x11,0x7c,0x02,0x40]       
vmovups %ymm7, 64(%rdx,%rax) 

// CHECK: vmovups %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfc,0x11,0x7c,0x82,0xc0]       
vmovups %ymm7, -64(%rdx,%rax,4) 

// CHECK: vmovups %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xfc,0x11,0x7c,0x82,0x40]       
vmovups %ymm7, 64(%rdx,%rax,4) 

// CHECK: vmovups %ymm7, (%rdx) 
// CHECK: encoding: [0xc5,0xfc,0x11,0x3a]       
vmovups %ymm7, (%rdx) 

// CHECK: vmovups %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x10,0xff]       
vmovups %ymm7, %ymm7 

// CHECK: vmovups %ymm9, 485498096 
// CHECK: encoding: [0xc5,0x7c,0x11,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovups %ymm9, 485498096 

// CHECK: vmovups %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc5,0x7c,0x11,0x4a,0x40]       
vmovups %ymm9, 64(%rdx) 

// CHECK: vmovups %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0x7c,0x11,0x4c,0x02,0x40]       
vmovups %ymm9, 64(%rdx,%rax) 

// CHECK: vmovups %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7c,0x11,0x4c,0x82,0xc0]       
vmovups %ymm9, -64(%rdx,%rax,4) 

// CHECK: vmovups %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0x7c,0x11,0x4c,0x82,0x40]       
vmovups %ymm9, 64(%rdx,%rax,4) 

// CHECK: vmovups %ymm9, (%rdx) 
// CHECK: encoding: [0xc5,0x7c,0x11,0x0a]       
vmovups %ymm9, (%rdx) 

// CHECK: vmovups %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7c,0x10,0xc9]       
vmovups %ymm9, %ymm9 

// CHECK: vmpsadbw $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x42,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096, %xmm15, %xmm15 

// CHECK: vmpsadbw $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x42,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096, %xmm6, %xmm6 

// CHECK: vmpsadbw $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x42,0x7c,0x82,0xc0,0x00]     
vmpsadbw $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmpsadbw $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x42,0x7c,0x82,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmpsadbw $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x42,0x74,0x82,0xc0,0x00]     
vmpsadbw $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmpsadbw $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x42,0x74,0x82,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmpsadbw $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x42,0x7c,0x02,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmpsadbw $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x42,0x74,0x02,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmpsadbw $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x42,0x7a,0x40,0x00]     
vmpsadbw $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vmpsadbw $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x42,0x72,0x40,0x00]     
vmpsadbw $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vmpsadbw $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x42,0x3a,0x00]     
vmpsadbw $0, (%rdx), %xmm15, %xmm15 

// CHECK: vmpsadbw $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x42,0x32,0x00]     
vmpsadbw $0, (%rdx), %xmm6, %xmm6 

// CHECK: vmpsadbw $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x42,0xff,0x00]     
vmpsadbw $0, %xmm15, %xmm15, %xmm15 

// CHECK: vmpsadbw $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x42,0xf6,0x00]     
vmpsadbw $0, %xmm6, %xmm6, %xmm6 

// CHECK: vmulpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096, %xmm15, %xmm15 

// CHECK: vmulpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x59,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096, %xmm6, %xmm6 

// CHECK: vmulpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096, %ymm7, %ymm7 

// CHECK: vmulpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulpd 485498096, %ymm9, %ymm9 

// CHECK: vmulpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x59,0x7c,0x82,0xc0]      
vmulpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x59,0x7c,0x82,0x40]      
vmulpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x59,0x74,0x82,0xc0]      
vmulpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x59,0x74,0x82,0x40]      
vmulpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x59,0x7c,0x82,0xc0]      
vmulpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmulpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x59,0x7c,0x82,0x40]      
vmulpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmulpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x59,0x4c,0x82,0xc0]      
vmulpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmulpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x59,0x4c,0x82,0x40]      
vmulpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmulpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x59,0x7c,0x02,0x40]      
vmulpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmulpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x59,0x74,0x02,0x40]      
vmulpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmulpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x59,0x7c,0x02,0x40]      
vmulpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vmulpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x59,0x4c,0x02,0x40]      
vmulpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vmulpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x59,0x7a,0x40]      
vmulpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vmulpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x59,0x72,0x40]      
vmulpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vmulpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x59,0x7a,0x40]      
vmulpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vmulpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x59,0x4a,0x40]      
vmulpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vmulpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x59,0x3a]      
vmulpd (%rdx), %xmm15, %xmm15 

// CHECK: vmulpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x59,0x32]      
vmulpd (%rdx), %xmm6, %xmm6 

// CHECK: vmulpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x59,0x3a]      
vmulpd (%rdx), %ymm7, %ymm7 

// CHECK: vmulpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x59,0x0a]      
vmulpd (%rdx), %ymm9, %ymm9 

// CHECK: vmulpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x59,0xff]      
vmulpd %xmm15, %xmm15, %xmm15 

// CHECK: vmulpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x59,0xf6]      
vmulpd %xmm6, %xmm6, %xmm6 

// CHECK: vmulpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x59,0xff]      
vmulpd %ymm7, %ymm7, %ymm7 

// CHECK: vmulpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x59,0xc9]      
vmulpd %ymm9, %ymm9, %ymm9 

// CHECK: vmulps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096, %xmm15, %xmm15 

// CHECK: vmulps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x59,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096, %xmm6, %xmm6 

// CHECK: vmulps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096, %ymm7, %ymm7 

// CHECK: vmulps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulps 485498096, %ymm9, %ymm9 

// CHECK: vmulps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x59,0x7c,0x82,0xc0]      
vmulps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x59,0x7c,0x82,0x40]      
vmulps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x59,0x74,0x82,0xc0]      
vmulps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x59,0x74,0x82,0x40]      
vmulps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x59,0x7c,0x82,0xc0]      
vmulps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmulps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x59,0x7c,0x82,0x40]      
vmulps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmulps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x59,0x4c,0x82,0xc0]      
vmulps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmulps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x59,0x4c,0x82,0x40]      
vmulps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmulps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x59,0x7c,0x02,0x40]      
vmulps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmulps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x59,0x74,0x02,0x40]      
vmulps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmulps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x59,0x7c,0x02,0x40]      
vmulps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vmulps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x59,0x4c,0x02,0x40]      
vmulps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vmulps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x59,0x7a,0x40]      
vmulps 64(%rdx), %xmm15, %xmm15 

// CHECK: vmulps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x59,0x72,0x40]      
vmulps 64(%rdx), %xmm6, %xmm6 

// CHECK: vmulps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x59,0x7a,0x40]      
vmulps 64(%rdx), %ymm7, %ymm7 

// CHECK: vmulps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x59,0x4a,0x40]      
vmulps 64(%rdx), %ymm9, %ymm9 

// CHECK: vmulps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x59,0x3a]      
vmulps (%rdx), %xmm15, %xmm15 

// CHECK: vmulps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x59,0x32]      
vmulps (%rdx), %xmm6, %xmm6 

// CHECK: vmulps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x59,0x3a]      
vmulps (%rdx), %ymm7, %ymm7 

// CHECK: vmulps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x59,0x0a]      
vmulps (%rdx), %ymm9, %ymm9 

// CHECK: vmulps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x59,0xff]      
vmulps %xmm15, %xmm15, %xmm15 

// CHECK: vmulps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x59,0xf6]      
vmulps %xmm6, %xmm6, %xmm6 

// CHECK: vmulps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x59,0xff]      
vmulps %ymm7, %ymm7, %ymm7 

// CHECK: vmulps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x59,0xc9]      
vmulps %ymm9, %ymm9, %ymm9 

// CHECK: vmulsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulsd 485498096, %xmm15, %xmm15 

// CHECK: vmulsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x59,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulsd 485498096, %xmm6, %xmm6 

// CHECK: vmulsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x59,0x7c,0x82,0xc0]      
vmulsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x59,0x7c,0x82,0x40]      
vmulsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x59,0x74,0x82,0xc0]      
vmulsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x59,0x74,0x82,0x40]      
vmulsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x59,0x7c,0x02,0x40]      
vmulsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmulsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x59,0x74,0x02,0x40]      
vmulsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmulsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x59,0x7a,0x40]      
vmulsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vmulsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x59,0x72,0x40]      
vmulsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vmulsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x59,0x3a]      
vmulsd (%rdx), %xmm15, %xmm15 

// CHECK: vmulsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x59,0x32]      
vmulsd (%rdx), %xmm6, %xmm6 

// CHECK: vmulsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x59,0xff]      
vmulsd %xmm15, %xmm15, %xmm15 

// CHECK: vmulsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x59,0xf6]      
vmulsd %xmm6, %xmm6, %xmm6 

// CHECK: vmulss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulss 485498096, %xmm15, %xmm15 

// CHECK: vmulss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x59,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vmulss 485498096, %xmm6, %xmm6 

// CHECK: vmulss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x59,0x7c,0x82,0xc0]      
vmulss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x59,0x7c,0x82,0x40]      
vmulss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vmulss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x59,0x74,0x82,0xc0]      
vmulss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x59,0x74,0x82,0x40]      
vmulss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vmulss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x59,0x7c,0x02,0x40]      
vmulss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vmulss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x59,0x74,0x02,0x40]      
vmulss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vmulss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x59,0x7a,0x40]      
vmulss 64(%rdx), %xmm15, %xmm15 

// CHECK: vmulss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x59,0x72,0x40]      
vmulss 64(%rdx), %xmm6, %xmm6 

// CHECK: vmulss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x59,0x3a]      
vmulss (%rdx), %xmm15, %xmm15 

// CHECK: vmulss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x59,0x32]      
vmulss (%rdx), %xmm6, %xmm6 

// CHECK: vmulss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x59,0xff]      
vmulss %xmm15, %xmm15, %xmm15 

// CHECK: vmulss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x59,0xf6]      
vmulss %xmm6, %xmm6, %xmm6 

// CHECK: vorpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x56,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096, %xmm15, %xmm15 

// CHECK: vorpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x56,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096, %xmm6, %xmm6 

// CHECK: vorpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x56,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096, %ymm7, %ymm7 

// CHECK: vorpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x56,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vorpd 485498096, %ymm9, %ymm9 

// CHECK: vorpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x56,0x7c,0x82,0xc0]      
vorpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vorpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x56,0x7c,0x82,0x40]      
vorpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vorpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x56,0x74,0x82,0xc0]      
vorpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vorpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x56,0x74,0x82,0x40]      
vorpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vorpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x56,0x7c,0x82,0xc0]      
vorpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vorpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x56,0x7c,0x82,0x40]      
vorpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vorpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x56,0x4c,0x82,0xc0]      
vorpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vorpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x56,0x4c,0x82,0x40]      
vorpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vorpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x56,0x7c,0x02,0x40]      
vorpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vorpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x56,0x74,0x02,0x40]      
vorpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vorpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x56,0x7c,0x02,0x40]      
vorpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vorpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x56,0x4c,0x02,0x40]      
vorpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vorpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x56,0x7a,0x40]      
vorpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vorpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x56,0x72,0x40]      
vorpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vorpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x56,0x7a,0x40]      
vorpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vorpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x56,0x4a,0x40]      
vorpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vorpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x56,0x3a]      
vorpd (%rdx), %xmm15, %xmm15 

// CHECK: vorpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x56,0x32]      
vorpd (%rdx), %xmm6, %xmm6 

// CHECK: vorpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x56,0x3a]      
vorpd (%rdx), %ymm7, %ymm7 

// CHECK: vorpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x56,0x0a]      
vorpd (%rdx), %ymm9, %ymm9 

// CHECK: vorpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x56,0xff]      
vorpd %xmm15, %xmm15, %xmm15 

// CHECK: vorpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x56,0xf6]      
vorpd %xmm6, %xmm6, %xmm6 

// CHECK: vorpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x56,0xff]      
vorpd %ymm7, %ymm7, %ymm7 

// CHECK: vorpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x56,0xc9]      
vorpd %ymm9, %ymm9, %ymm9 

// CHECK: vorps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x56,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096, %xmm15, %xmm15 

// CHECK: vorps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x56,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096, %xmm6, %xmm6 

// CHECK: vorps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x56,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096, %ymm7, %ymm7 

// CHECK: vorps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x56,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vorps 485498096, %ymm9, %ymm9 

// CHECK: vorps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x56,0x7c,0x82,0xc0]      
vorps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vorps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x56,0x7c,0x82,0x40]      
vorps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vorps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x56,0x74,0x82,0xc0]      
vorps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vorps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x56,0x74,0x82,0x40]      
vorps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vorps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x56,0x7c,0x82,0xc0]      
vorps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vorps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x56,0x7c,0x82,0x40]      
vorps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vorps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x56,0x4c,0x82,0xc0]      
vorps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vorps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x56,0x4c,0x82,0x40]      
vorps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vorps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x56,0x7c,0x02,0x40]      
vorps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vorps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x56,0x74,0x02,0x40]      
vorps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vorps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x56,0x7c,0x02,0x40]      
vorps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vorps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x56,0x4c,0x02,0x40]      
vorps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vorps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x56,0x7a,0x40]      
vorps 64(%rdx), %xmm15, %xmm15 

// CHECK: vorps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x56,0x72,0x40]      
vorps 64(%rdx), %xmm6, %xmm6 

// CHECK: vorps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x56,0x7a,0x40]      
vorps 64(%rdx), %ymm7, %ymm7 

// CHECK: vorps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x56,0x4a,0x40]      
vorps 64(%rdx), %ymm9, %ymm9 

// CHECK: vorps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x56,0x3a]      
vorps (%rdx), %xmm15, %xmm15 

// CHECK: vorps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x56,0x32]      
vorps (%rdx), %xmm6, %xmm6 

// CHECK: vorps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x56,0x3a]      
vorps (%rdx), %ymm7, %ymm7 

// CHECK: vorps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x56,0x0a]      
vorps (%rdx), %ymm9, %ymm9 

// CHECK: vorps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x56,0xff]      
vorps %xmm15, %xmm15, %xmm15 

// CHECK: vorps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x56,0xf6]      
vorps %xmm6, %xmm6, %xmm6 

// CHECK: vorps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x56,0xff]      
vorps %ymm7, %ymm7, %ymm7 

// CHECK: vorps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x56,0xc9]      
vorps %ymm9, %ymm9, %ymm9 

// CHECK: vpabsb 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096, %xmm15 

// CHECK: vpabsb 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096, %xmm6 

// CHECK: vpabsb -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1c,0x7c,0x82,0xc0]       
vpabsb -64(%rdx,%rax,4), %xmm15 

// CHECK: vpabsb 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1c,0x7c,0x82,0x40]       
vpabsb 64(%rdx,%rax,4), %xmm15 

// CHECK: vpabsb -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x74,0x82,0xc0]       
vpabsb -64(%rdx,%rax,4), %xmm6 

// CHECK: vpabsb 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x74,0x82,0x40]       
vpabsb 64(%rdx,%rax,4), %xmm6 

// CHECK: vpabsb 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1c,0x7c,0x02,0x40]       
vpabsb 64(%rdx,%rax), %xmm15 

// CHECK: vpabsb 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x74,0x02,0x40]       
vpabsb 64(%rdx,%rax), %xmm6 

// CHECK: vpabsb 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1c,0x7a,0x40]       
vpabsb 64(%rdx), %xmm15 

// CHECK: vpabsb 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x72,0x40]       
vpabsb 64(%rdx), %xmm6 

// CHECK: vpabsb (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1c,0x3a]       
vpabsb (%rdx), %xmm15 

// CHECK: vpabsb (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x32]       
vpabsb (%rdx), %xmm6 

// CHECK: vpabsb %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x1c,0xff]       
vpabsb %xmm15, %xmm15 

// CHECK: vpabsb %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0xf6]       
vpabsb %xmm6, %xmm6 

// CHECK: vpabsd 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096, %xmm15 

// CHECK: vpabsd 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096, %xmm6 

// CHECK: vpabsd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1e,0x7c,0x82,0xc0]       
vpabsd -64(%rdx,%rax,4), %xmm15 

// CHECK: vpabsd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1e,0x7c,0x82,0x40]       
vpabsd 64(%rdx,%rax,4), %xmm15 

// CHECK: vpabsd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x74,0x82,0xc0]       
vpabsd -64(%rdx,%rax,4), %xmm6 

// CHECK: vpabsd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x74,0x82,0x40]       
vpabsd 64(%rdx,%rax,4), %xmm6 

// CHECK: vpabsd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1e,0x7c,0x02,0x40]       
vpabsd 64(%rdx,%rax), %xmm15 

// CHECK: vpabsd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x74,0x02,0x40]       
vpabsd 64(%rdx,%rax), %xmm6 

// CHECK: vpabsd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1e,0x7a,0x40]       
vpabsd 64(%rdx), %xmm15 

// CHECK: vpabsd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x72,0x40]       
vpabsd 64(%rdx), %xmm6 

// CHECK: vpabsd (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1e,0x3a]       
vpabsd (%rdx), %xmm15 

// CHECK: vpabsd (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x32]       
vpabsd (%rdx), %xmm6 

// CHECK: vpabsd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x1e,0xff]       
vpabsd %xmm15, %xmm15 

// CHECK: vpabsd %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0xf6]       
vpabsd %xmm6, %xmm6 

// CHECK: vpabsw 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096, %xmm15 

// CHECK: vpabsw 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096, %xmm6 

// CHECK: vpabsw -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1d,0x7c,0x82,0xc0]       
vpabsw -64(%rdx,%rax,4), %xmm15 

// CHECK: vpabsw 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1d,0x7c,0x82,0x40]       
vpabsw 64(%rdx,%rax,4), %xmm15 

// CHECK: vpabsw -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x74,0x82,0xc0]       
vpabsw -64(%rdx,%rax,4), %xmm6 

// CHECK: vpabsw 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x74,0x82,0x40]       
vpabsw 64(%rdx,%rax,4), %xmm6 

// CHECK: vpabsw 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1d,0x7c,0x02,0x40]       
vpabsw 64(%rdx,%rax), %xmm15 

// CHECK: vpabsw 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x74,0x02,0x40]       
vpabsw 64(%rdx,%rax), %xmm6 

// CHECK: vpabsw 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1d,0x7a,0x40]       
vpabsw 64(%rdx), %xmm15 

// CHECK: vpabsw 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x72,0x40]       
vpabsw 64(%rdx), %xmm6 

// CHECK: vpabsw (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x1d,0x3a]       
vpabsw (%rdx), %xmm15 

// CHECK: vpabsw (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x32]       
vpabsw (%rdx), %xmm6 

// CHECK: vpabsw %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x1d,0xff]       
vpabsw %xmm15, %xmm15 

// CHECK: vpabsw %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0xf6]       
vpabsw %xmm6, %xmm6 

// CHECK: vpackssdw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096, %xmm15, %xmm15 

// CHECK: vpackssdw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096, %xmm6, %xmm6 

// CHECK: vpackssdw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6b,0x7c,0x82,0xc0]      
vpackssdw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpackssdw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6b,0x7c,0x82,0x40]      
vpackssdw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpackssdw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6b,0x74,0x82,0xc0]      
vpackssdw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpackssdw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6b,0x74,0x82,0x40]      
vpackssdw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpackssdw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6b,0x7c,0x02,0x40]      
vpackssdw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpackssdw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6b,0x74,0x02,0x40]      
vpackssdw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpackssdw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6b,0x7a,0x40]      
vpackssdw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpackssdw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6b,0x72,0x40]      
vpackssdw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpackssdw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6b,0x3a]      
vpackssdw (%rdx), %xmm15, %xmm15 

// CHECK: vpackssdw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6b,0x32]      
vpackssdw (%rdx), %xmm6, %xmm6 

// CHECK: vpackssdw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x6b,0xff]      
vpackssdw %xmm15, %xmm15, %xmm15 

// CHECK: vpackssdw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6b,0xf6]      
vpackssdw %xmm6, %xmm6, %xmm6 

// CHECK: vpacksswb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x63,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096, %xmm15, %xmm15 

// CHECK: vpacksswb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x63,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096, %xmm6, %xmm6 

// CHECK: vpacksswb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x63,0x7c,0x82,0xc0]      
vpacksswb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpacksswb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x63,0x7c,0x82,0x40]      
vpacksswb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpacksswb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x63,0x74,0x82,0xc0]      
vpacksswb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpacksswb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x63,0x74,0x82,0x40]      
vpacksswb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpacksswb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x63,0x7c,0x02,0x40]      
vpacksswb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpacksswb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x63,0x74,0x02,0x40]      
vpacksswb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpacksswb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x63,0x7a,0x40]      
vpacksswb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpacksswb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x63,0x72,0x40]      
vpacksswb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpacksswb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x63,0x3a]      
vpacksswb (%rdx), %xmm15, %xmm15 

// CHECK: vpacksswb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x63,0x32]      
vpacksswb (%rdx), %xmm6, %xmm6 

// CHECK: vpacksswb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x63,0xff]      
vpacksswb %xmm15, %xmm15, %xmm15 

// CHECK: vpacksswb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x63,0xf6]      
vpacksswb %xmm6, %xmm6, %xmm6 

// CHECK: vpackusdw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096, %xmm15, %xmm15 

// CHECK: vpackusdw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096, %xmm6, %xmm6 

// CHECK: vpackusdw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2b,0x7c,0x82,0xc0]      
vpackusdw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpackusdw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2b,0x7c,0x82,0x40]      
vpackusdw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpackusdw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2b,0x74,0x82,0xc0]      
vpackusdw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpackusdw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2b,0x74,0x82,0x40]      
vpackusdw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpackusdw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2b,0x7c,0x02,0x40]      
vpackusdw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpackusdw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2b,0x74,0x02,0x40]      
vpackusdw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpackusdw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2b,0x7a,0x40]      
vpackusdw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpackusdw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2b,0x72,0x40]      
vpackusdw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpackusdw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x2b,0x3a]      
vpackusdw (%rdx), %xmm15, %xmm15 

// CHECK: vpackusdw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2b,0x32]      
vpackusdw (%rdx), %xmm6, %xmm6 

// CHECK: vpackusdw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x2b,0xff]      
vpackusdw %xmm15, %xmm15, %xmm15 

// CHECK: vpackusdw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x2b,0xf6]      
vpackusdw %xmm6, %xmm6, %xmm6 

// CHECK: vpackuswb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x67,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096, %xmm15, %xmm15 

// CHECK: vpackuswb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x67,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096, %xmm6, %xmm6 

// CHECK: vpackuswb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x67,0x7c,0x82,0xc0]      
vpackuswb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpackuswb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x67,0x7c,0x82,0x40]      
vpackuswb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpackuswb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x67,0x74,0x82,0xc0]      
vpackuswb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpackuswb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x67,0x74,0x82,0x40]      
vpackuswb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpackuswb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x67,0x7c,0x02,0x40]      
vpackuswb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpackuswb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x67,0x74,0x02,0x40]      
vpackuswb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpackuswb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x67,0x7a,0x40]      
vpackuswb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpackuswb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x67,0x72,0x40]      
vpackuswb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpackuswb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x67,0x3a]      
vpackuswb (%rdx), %xmm15, %xmm15 

// CHECK: vpackuswb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x67,0x32]      
vpackuswb (%rdx), %xmm6, %xmm6 

// CHECK: vpackuswb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x67,0xff]      
vpackuswb %xmm15, %xmm15, %xmm15 

// CHECK: vpackuswb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x67,0xf6]      
vpackuswb %xmm6, %xmm6, %xmm6 

// CHECK: vpaddb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096, %xmm15, %xmm15 

// CHECK: vpaddb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfc,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096, %xmm6, %xmm6 

// CHECK: vpaddb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfc,0x7c,0x82,0xc0]      
vpaddb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfc,0x7c,0x82,0x40]      
vpaddb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfc,0x74,0x82,0xc0]      
vpaddb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfc,0x74,0x82,0x40]      
vpaddb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfc,0x7c,0x02,0x40]      
vpaddb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfc,0x74,0x02,0x40]      
vpaddb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfc,0x7a,0x40]      
vpaddb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfc,0x72,0x40]      
vpaddb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfc,0x3a]      
vpaddb (%rdx), %xmm15, %xmm15 

// CHECK: vpaddb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfc,0x32]      
vpaddb (%rdx), %xmm6, %xmm6 

// CHECK: vpaddb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xfc,0xff]      
vpaddb %xmm15, %xmm15, %xmm15 

// CHECK: vpaddb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfc,0xf6]      
vpaddb %xmm6, %xmm6, %xmm6 

// CHECK: vpaddd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfe,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096, %xmm15, %xmm15 

// CHECK: vpaddd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfe,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096, %xmm6, %xmm6 

// CHECK: vpaddd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfe,0x7c,0x82,0xc0]      
vpaddd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfe,0x7c,0x82,0x40]      
vpaddd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfe,0x74,0x82,0xc0]      
vpaddd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfe,0x74,0x82,0x40]      
vpaddd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfe,0x7c,0x02,0x40]      
vpaddd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfe,0x74,0x02,0x40]      
vpaddd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfe,0x7a,0x40]      
vpaddd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfe,0x72,0x40]      
vpaddd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfe,0x3a]      
vpaddd (%rdx), %xmm15, %xmm15 

// CHECK: vpaddd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfe,0x32]      
vpaddd (%rdx), %xmm6, %xmm6 

// CHECK: vpaddd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xfe,0xff]      
vpaddd %xmm15, %xmm15, %xmm15 

// CHECK: vpaddd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfe,0xf6]      
vpaddd %xmm6, %xmm6, %xmm6 

// CHECK: vpaddq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd4,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096, %xmm15, %xmm15 

// CHECK: vpaddq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd4,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096, %xmm6, %xmm6 

// CHECK: vpaddq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd4,0x7c,0x82,0xc0]      
vpaddq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd4,0x7c,0x82,0x40]      
vpaddq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd4,0x74,0x82,0xc0]      
vpaddq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd4,0x74,0x82,0x40]      
vpaddq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd4,0x7c,0x02,0x40]      
vpaddq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd4,0x74,0x02,0x40]      
vpaddq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd4,0x7a,0x40]      
vpaddq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd4,0x72,0x40]      
vpaddq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd4,0x3a]      
vpaddq (%rdx), %xmm15, %xmm15 

// CHECK: vpaddq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd4,0x32]      
vpaddq (%rdx), %xmm6, %xmm6 

// CHECK: vpaddq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd4,0xff]      
vpaddq %xmm15, %xmm15, %xmm15 

// CHECK: vpaddq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd4,0xf6]      
vpaddq %xmm6, %xmm6, %xmm6 

// CHECK: vpaddsb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xec,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096, %xmm15, %xmm15 

// CHECK: vpaddsb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xec,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096, %xmm6, %xmm6 

// CHECK: vpaddsb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xec,0x7c,0x82,0xc0]      
vpaddsb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddsb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xec,0x7c,0x82,0x40]      
vpaddsb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddsb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xec,0x74,0x82,0xc0]      
vpaddsb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddsb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xec,0x74,0x82,0x40]      
vpaddsb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddsb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xec,0x7c,0x02,0x40]      
vpaddsb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddsb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xec,0x74,0x02,0x40]      
vpaddsb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddsb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xec,0x7a,0x40]      
vpaddsb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddsb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xec,0x72,0x40]      
vpaddsb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddsb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xec,0x3a]      
vpaddsb (%rdx), %xmm15, %xmm15 

// CHECK: vpaddsb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xec,0x32]      
vpaddsb (%rdx), %xmm6, %xmm6 

// CHECK: vpaddsb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xec,0xff]      
vpaddsb %xmm15, %xmm15, %xmm15 

// CHECK: vpaddsb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xec,0xf6]      
vpaddsb %xmm6, %xmm6, %xmm6 

// CHECK: vpaddsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xed,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096, %xmm15, %xmm15 

// CHECK: vpaddsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xed,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096, %xmm6, %xmm6 

// CHECK: vpaddsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xed,0x7c,0x82,0xc0]      
vpaddsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xed,0x7c,0x82,0x40]      
vpaddsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xed,0x74,0x82,0xc0]      
vpaddsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xed,0x74,0x82,0x40]      
vpaddsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xed,0x7c,0x02,0x40]      
vpaddsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xed,0x74,0x02,0x40]      
vpaddsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xed,0x7a,0x40]      
vpaddsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xed,0x72,0x40]      
vpaddsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xed,0x3a]      
vpaddsw (%rdx), %xmm15, %xmm15 

// CHECK: vpaddsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xed,0x32]      
vpaddsw (%rdx), %xmm6, %xmm6 

// CHECK: vpaddsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xed,0xff]      
vpaddsw %xmm15, %xmm15, %xmm15 

// CHECK: vpaddsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xed,0xf6]      
vpaddsw %xmm6, %xmm6, %xmm6 

// CHECK: vpaddusb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096, %xmm15, %xmm15 

// CHECK: vpaddusb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdc,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096, %xmm6, %xmm6 

// CHECK: vpaddusb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdc,0x7c,0x82,0xc0]      
vpaddusb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddusb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdc,0x7c,0x82,0x40]      
vpaddusb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddusb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdc,0x74,0x82,0xc0]      
vpaddusb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddusb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdc,0x74,0x82,0x40]      
vpaddusb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddusb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdc,0x7c,0x02,0x40]      
vpaddusb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddusb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdc,0x74,0x02,0x40]      
vpaddusb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddusb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdc,0x7a,0x40]      
vpaddusb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddusb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdc,0x72,0x40]      
vpaddusb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddusb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdc,0x3a]      
vpaddusb (%rdx), %xmm15, %xmm15 

// CHECK: vpaddusb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdc,0x32]      
vpaddusb (%rdx), %xmm6, %xmm6 

// CHECK: vpaddusb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xdc,0xff]      
vpaddusb %xmm15, %xmm15, %xmm15 

// CHECK: vpaddusb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdc,0xf6]      
vpaddusb %xmm6, %xmm6, %xmm6 

// CHECK: vpaddusw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096, %xmm15, %xmm15 

// CHECK: vpaddusw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdd,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096, %xmm6, %xmm6 

// CHECK: vpaddusw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdd,0x7c,0x82,0xc0]      
vpaddusw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddusw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdd,0x7c,0x82,0x40]      
vpaddusw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddusw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdd,0x74,0x82,0xc0]      
vpaddusw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddusw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdd,0x74,0x82,0x40]      
vpaddusw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddusw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdd,0x7c,0x02,0x40]      
vpaddusw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddusw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdd,0x74,0x02,0x40]      
vpaddusw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddusw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdd,0x7a,0x40]      
vpaddusw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddusw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdd,0x72,0x40]      
vpaddusw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddusw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdd,0x3a]      
vpaddusw (%rdx), %xmm15, %xmm15 

// CHECK: vpaddusw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdd,0x32]      
vpaddusw (%rdx), %xmm6, %xmm6 

// CHECK: vpaddusw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xdd,0xff]      
vpaddusw %xmm15, %xmm15, %xmm15 

// CHECK: vpaddusw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdd,0xf6]      
vpaddusw %xmm6, %xmm6, %xmm6 

// CHECK: vpaddw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096, %xmm15, %xmm15 

// CHECK: vpaddw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfd,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096, %xmm6, %xmm6 

// CHECK: vpaddw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfd,0x7c,0x82,0xc0]      
vpaddw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfd,0x7c,0x82,0x40]      
vpaddw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpaddw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfd,0x74,0x82,0xc0]      
vpaddw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfd,0x74,0x82,0x40]      
vpaddw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpaddw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfd,0x7c,0x02,0x40]      
vpaddw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpaddw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfd,0x74,0x02,0x40]      
vpaddw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpaddw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfd,0x7a,0x40]      
vpaddw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpaddw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfd,0x72,0x40]      
vpaddw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpaddw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfd,0x3a]      
vpaddw (%rdx), %xmm15, %xmm15 

// CHECK: vpaddw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfd,0x32]      
vpaddw (%rdx), %xmm6, %xmm6 

// CHECK: vpaddw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xfd,0xff]      
vpaddw %xmm15, %xmm15, %xmm15 

// CHECK: vpaddw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfd,0xf6]      
vpaddw %xmm6, %xmm6, %xmm6 

// CHECK: vpalignr $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096, %xmm15, %xmm15 

// CHECK: vpalignr $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0f,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096, %xmm6, %xmm6 

// CHECK: vpalignr $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0f,0x7c,0x82,0xc0,0x00]     
vpalignr $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpalignr $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0f,0x7c,0x82,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpalignr $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0f,0x74,0x82,0xc0,0x00]     
vpalignr $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpalignr $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0f,0x74,0x82,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpalignr $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0f,0x7c,0x02,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpalignr $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0f,0x74,0x02,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpalignr $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0f,0x7a,0x40,0x00]     
vpalignr $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpalignr $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0f,0x72,0x40,0x00]     
vpalignr $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpalignr $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0f,0x3a,0x00]     
vpalignr $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpalignr $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0f,0x32,0x00]     
vpalignr $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpalignr $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x0f,0xff,0x00]     
vpalignr $0, %xmm15, %xmm15, %xmm15 

// CHECK: vpalignr $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0f,0xf6,0x00]     
vpalignr $0, %xmm6, %xmm6, %xmm6 

// CHECK: vpand 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096, %xmm15, %xmm15 

// CHECK: vpand 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdb,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096, %xmm6, %xmm6 

// CHECK: vpand -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdb,0x7c,0x82,0xc0]      
vpand -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpand 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdb,0x7c,0x82,0x40]      
vpand 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpand -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdb,0x74,0x82,0xc0]      
vpand -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpand 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdb,0x74,0x82,0x40]      
vpand 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpand 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdb,0x7c,0x02,0x40]      
vpand 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpand 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdb,0x74,0x02,0x40]      
vpand 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpand 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdb,0x7a,0x40]      
vpand 64(%rdx), %xmm15, %xmm15 

// CHECK: vpand 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdb,0x72,0x40]      
vpand 64(%rdx), %xmm6, %xmm6 

// CHECK: vpandn 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096, %xmm15, %xmm15 

// CHECK: vpandn 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdf,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096, %xmm6, %xmm6 

// CHECK: vpandn -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdf,0x7c,0x82,0xc0]      
vpandn -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpandn 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdf,0x7c,0x82,0x40]      
vpandn 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpandn -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdf,0x74,0x82,0xc0]      
vpandn -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpandn 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdf,0x74,0x82,0x40]      
vpandn 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpandn 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdf,0x7c,0x02,0x40]      
vpandn 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpandn 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdf,0x74,0x02,0x40]      
vpandn 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpandn 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdf,0x7a,0x40]      
vpandn 64(%rdx), %xmm15, %xmm15 

// CHECK: vpandn 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdf,0x72,0x40]      
vpandn 64(%rdx), %xmm6, %xmm6 

// CHECK: vpandn (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdf,0x3a]      
vpandn (%rdx), %xmm15, %xmm15 

// CHECK: vpandn (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdf,0x32]      
vpandn (%rdx), %xmm6, %xmm6 

// CHECK: vpandn %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xdf,0xff]      
vpandn %xmm15, %xmm15, %xmm15 

// CHECK: vpandn %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdf,0xf6]      
vpandn %xmm6, %xmm6, %xmm6 

// CHECK: vpand (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xdb,0x3a]      
vpand (%rdx), %xmm15, %xmm15 

// CHECK: vpand (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdb,0x32]      
vpand (%rdx), %xmm6, %xmm6 

// CHECK: vpand %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xdb,0xff]      
vpand %xmm15, %xmm15, %xmm15 

// CHECK: vpand %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xdb,0xf6]      
vpand %xmm6, %xmm6, %xmm6 

// CHECK: vpavgb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096, %xmm15, %xmm15 

// CHECK: vpavgb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096, %xmm6, %xmm6 

// CHECK: vpavgb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe0,0x7c,0x82,0xc0]      
vpavgb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpavgb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe0,0x7c,0x82,0x40]      
vpavgb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpavgb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe0,0x74,0x82,0xc0]      
vpavgb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpavgb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe0,0x74,0x82,0x40]      
vpavgb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpavgb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe0,0x7c,0x02,0x40]      
vpavgb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpavgb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe0,0x74,0x02,0x40]      
vpavgb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpavgb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe0,0x7a,0x40]      
vpavgb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpavgb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe0,0x72,0x40]      
vpavgb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpavgb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe0,0x3a]      
vpavgb (%rdx), %xmm15, %xmm15 

// CHECK: vpavgb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe0,0x32]      
vpavgb (%rdx), %xmm6, %xmm6 

// CHECK: vpavgb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe0,0xff]      
vpavgb %xmm15, %xmm15, %xmm15 

// CHECK: vpavgb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe0,0xf6]      
vpavgb %xmm6, %xmm6, %xmm6 

// CHECK: vpavgw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096, %xmm15, %xmm15 

// CHECK: vpavgw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe3,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096, %xmm6, %xmm6 

// CHECK: vpavgw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe3,0x7c,0x82,0xc0]      
vpavgw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpavgw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe3,0x7c,0x82,0x40]      
vpavgw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpavgw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe3,0x74,0x82,0xc0]      
vpavgw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpavgw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe3,0x74,0x82,0x40]      
vpavgw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpavgw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe3,0x7c,0x02,0x40]      
vpavgw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpavgw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe3,0x74,0x02,0x40]      
vpavgw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpavgw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe3,0x7a,0x40]      
vpavgw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpavgw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe3,0x72,0x40]      
vpavgw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpavgw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe3,0x3a]      
vpavgw (%rdx), %xmm15, %xmm15 

// CHECK: vpavgw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe3,0x32]      
vpavgw (%rdx), %xmm6, %xmm6 

// CHECK: vpavgw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe3,0xff]      
vpavgw %xmm15, %xmm15, %xmm15 

// CHECK: vpavgw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe3,0xf6]      
vpavgw %xmm6, %xmm6, %xmm6 

// CHECK: vpblendvb %xmm15, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0xf0]     
vpblendvb %xmm15, 485498096, %xmm15, %xmm15 

// CHECK: vpblendvb %xmm15, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4c,0x7c,0x82,0xc0,0xf0]     
vpblendvb %xmm15, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpblendvb %xmm15, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4c,0x7c,0x82,0x40,0xf0]     
vpblendvb %xmm15, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpblendvb %xmm15, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4c,0x7c,0x02,0x40,0xf0]     
vpblendvb %xmm15, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpblendvb %xmm15, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4c,0x7a,0x40,0xf0]     
vpblendvb %xmm15, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpblendvb %xmm15, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x4c,0x3a,0xf0]     
vpblendvb %xmm15, (%rdx), %xmm15, %xmm15 

// CHECK: vpblendvb %xmm15, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x4c,0xff,0xf0]     
vpblendvb %xmm15, %xmm15, %xmm15, %xmm15 

// CHECK: vpblendvb %xmm6, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4c,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x60]     
vpblendvb %xmm6, 485498096, %xmm6, %xmm6 

// CHECK: vpblendvb %xmm6, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4c,0x74,0x82,0xc0,0x60]     
vpblendvb %xmm6, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpblendvb %xmm6, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4c,0x74,0x82,0x40,0x60]     
vpblendvb %xmm6, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpblendvb %xmm6, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4c,0x74,0x02,0x40,0x60]     
vpblendvb %xmm6, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpblendvb %xmm6, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4c,0x72,0x40,0x60]     
vpblendvb %xmm6, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpblendvb %xmm6, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4c,0x32,0x60]     
vpblendvb %xmm6, (%rdx), %xmm6, %xmm6 

// CHECK: vpblendvb %xmm6, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x4c,0xf6,0x60]     
vpblendvb %xmm6, %xmm6, %xmm6, %xmm6 

// CHECK: vpblendw $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096, %xmm15, %xmm15 

// CHECK: vpblendw $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0e,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096, %xmm6, %xmm6 

// CHECK: vpblendw $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0e,0x7c,0x82,0xc0,0x00]     
vpblendw $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpblendw $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0e,0x7c,0x82,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpblendw $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0e,0x74,0x82,0xc0,0x00]     
vpblendw $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpblendw $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0e,0x74,0x82,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpblendw $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0e,0x7c,0x02,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpblendw $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0e,0x74,0x02,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpblendw $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0e,0x7a,0x40,0x00]     
vpblendw $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpblendw $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0e,0x72,0x40,0x00]     
vpblendw $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpblendw $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0e,0x3a,0x00]     
vpblendw $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpblendw $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0e,0x32,0x00]     
vpblendw $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpblendw $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x0e,0xff,0x00]     
vpblendw $0, %xmm15, %xmm15, %xmm15 

// CHECK: vpblendw $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0e,0xf6,0x00]     
vpblendw $0, %xmm6, %xmm6, %xmm6 

// CHECK: vpclmulqdq $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x44,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpclmulqdq $0, 485498096, %xmm15, %xmm15 

// CHECK: vpclmulqdq $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x44,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpclmulqdq $0, 485498096, %xmm6, %xmm6 

// CHECK: vpclmulqdq $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x44,0x7c,0x82,0xc0,0x00]     
vpclmulqdq $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpclmulqdq $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x44,0x7c,0x82,0x40,0x00]     
vpclmulqdq $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpclmulqdq $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x44,0x74,0x82,0xc0,0x00]     
vpclmulqdq $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpclmulqdq $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x44,0x74,0x82,0x40,0x00]     
vpclmulqdq $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpclmulqdq $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x44,0x7c,0x02,0x40,0x00]     
vpclmulqdq $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpclmulqdq $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x44,0x74,0x02,0x40,0x00]     
vpclmulqdq $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpclmulqdq $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x44,0x7a,0x40,0x00]     
vpclmulqdq $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpclmulqdq $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x44,0x72,0x40,0x00]     
vpclmulqdq $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpclmulqdq $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x44,0x3a,0x00]     
vpclmulqdq $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpclmulqdq $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x44,0x32,0x00]     
vpclmulqdq $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpclmulqdq $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x44,0xff,0x00]     
vpclmulqdq $0, %xmm15, %xmm15, %xmm15 

// CHECK: vpclmulqdq $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x44,0xf6,0x00]     
vpclmulqdq $0, %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpeqb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x74,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096, %xmm15, %xmm15 

// CHECK: vpcmpeqb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x74,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096, %xmm6, %xmm6 

// CHECK: vpcmpeqb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x74,0x7c,0x82,0xc0]      
vpcmpeqb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x74,0x7c,0x82,0x40]      
vpcmpeqb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x74,0x74,0x82,0xc0]      
vpcmpeqb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x74,0x74,0x82,0x40]      
vpcmpeqb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x74,0x7c,0x02,0x40]      
vpcmpeqb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpeqb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x74,0x74,0x02,0x40]      
vpcmpeqb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpeqb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x74,0x7a,0x40]      
vpcmpeqb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x74,0x72,0x40]      
vpcmpeqb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x74,0x3a]      
vpcmpeqb (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x74,0x32]      
vpcmpeqb (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x74,0xff]      
vpcmpeqb %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpeqb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x74,0xf6]      
vpcmpeqb %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpeqd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x76,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096, %xmm15, %xmm15 

// CHECK: vpcmpeqd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x76,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096, %xmm6, %xmm6 

// CHECK: vpcmpeqd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x76,0x7c,0x82,0xc0]      
vpcmpeqd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x76,0x7c,0x82,0x40]      
vpcmpeqd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x76,0x74,0x82,0xc0]      
vpcmpeqd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x76,0x74,0x82,0x40]      
vpcmpeqd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x76,0x7c,0x02,0x40]      
vpcmpeqd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpeqd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x76,0x74,0x02,0x40]      
vpcmpeqd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpeqd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x76,0x7a,0x40]      
vpcmpeqd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x76,0x72,0x40]      
vpcmpeqd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x76,0x3a]      
vpcmpeqd (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x76,0x32]      
vpcmpeqd (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x76,0xff]      
vpcmpeqd %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpeqd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x76,0xf6]      
vpcmpeqd %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpeqq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x29,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096, %xmm15, %xmm15 

// CHECK: vpcmpeqq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x29,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096, %xmm6, %xmm6 

// CHECK: vpcmpeqq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x29,0x7c,0x82,0xc0]      
vpcmpeqq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x29,0x7c,0x82,0x40]      
vpcmpeqq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x29,0x74,0x82,0xc0]      
vpcmpeqq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x29,0x74,0x82,0x40]      
vpcmpeqq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x29,0x7c,0x02,0x40]      
vpcmpeqq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpeqq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x29,0x74,0x02,0x40]      
vpcmpeqq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpeqq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x29,0x7a,0x40]      
vpcmpeqq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x29,0x72,0x40]      
vpcmpeqq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x29,0x3a]      
vpcmpeqq (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x29,0x32]      
vpcmpeqq (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x29,0xff]      
vpcmpeqq %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpeqq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x29,0xf6]      
vpcmpeqq %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpeqw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x75,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096, %xmm15, %xmm15 

// CHECK: vpcmpeqw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x75,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096, %xmm6, %xmm6 

// CHECK: vpcmpeqw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x75,0x7c,0x82,0xc0]      
vpcmpeqw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x75,0x7c,0x82,0x40]      
vpcmpeqw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpeqw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x75,0x74,0x82,0xc0]      
vpcmpeqw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x75,0x74,0x82,0x40]      
vpcmpeqw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpeqw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x75,0x7c,0x02,0x40]      
vpcmpeqw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpeqw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x75,0x74,0x02,0x40]      
vpcmpeqw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpeqw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x75,0x7a,0x40]      
vpcmpeqw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x75,0x72,0x40]      
vpcmpeqw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x75,0x3a]      
vpcmpeqw (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpeqw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x75,0x32]      
vpcmpeqw (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpeqw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x75,0xff]      
vpcmpeqw %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpeqw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x75,0xf6]      
vpcmpeqw %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpestri $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x61,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestri $0, 485498096, %xmm15 

// CHECK: vpcmpestri $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestri $0, 485498096, %xmm6 

// CHECK: vpcmpestri $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x61,0x7c,0x82,0xc0,0x00]      
vpcmpestri $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpestri $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x61,0x7c,0x82,0x40,0x00]      
vpcmpestri $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpestri $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x74,0x82,0xc0,0x00]      
vpcmpestri $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpestri $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x74,0x82,0x40,0x00]      
vpcmpestri $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpestri $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x61,0x7c,0x02,0x40,0x00]      
vpcmpestri $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpcmpestri $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x74,0x02,0x40,0x00]      
vpcmpestri $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpcmpestri $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x61,0x7a,0x40,0x00]      
vpcmpestri $0, 64(%rdx), %xmm15 

// CHECK: vpcmpestri $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x72,0x40,0x00]      
vpcmpestri $0, 64(%rdx), %xmm6 

// CHECK: vpcmpestri $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x61,0x3a,0x00]      
vpcmpestri $0, (%rdx), %xmm15 

// CHECK: vpcmpestri $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x32,0x00]      
vpcmpestri $0, (%rdx), %xmm6 

// CHECK: vpcmpestri $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x61,0xff,0x00]      
vpcmpestri $0, %xmm15, %xmm15 

// CHECK: vpcmpestri $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0xf6,0x00]      
vpcmpestri $0, %xmm6, %xmm6 

// CHECK: vpcmpestrm $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x60,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestrm $0, 485498096, %xmm15 

// CHECK: vpcmpestrm $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpestrm $0, 485498096, %xmm6 

// CHECK: vpcmpestrm $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x60,0x7c,0x82,0xc0,0x00]      
vpcmpestrm $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpestrm $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x60,0x7c,0x82,0x40,0x00]      
vpcmpestrm $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpestrm $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x74,0x82,0xc0,0x00]      
vpcmpestrm $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpestrm $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x74,0x82,0x40,0x00]      
vpcmpestrm $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpestrm $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x60,0x7c,0x02,0x40,0x00]      
vpcmpestrm $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpcmpestrm $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x74,0x02,0x40,0x00]      
vpcmpestrm $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpcmpestrm $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x60,0x7a,0x40,0x00]      
vpcmpestrm $0, 64(%rdx), %xmm15 

// CHECK: vpcmpestrm $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x72,0x40,0x00]      
vpcmpestrm $0, 64(%rdx), %xmm6 

// CHECK: vpcmpestrm $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x60,0x3a,0x00]      
vpcmpestrm $0, (%rdx), %xmm15 

// CHECK: vpcmpestrm $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x32,0x00]      
vpcmpestrm $0, (%rdx), %xmm6 

// CHECK: vpcmpestrm $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x60,0xff,0x00]      
vpcmpestrm $0, %xmm15, %xmm15 

// CHECK: vpcmpestrm $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0xf6,0x00]      
vpcmpestrm $0, %xmm6, %xmm6 

// CHECK: vpcmpgtb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x64,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096, %xmm15, %xmm15 

// CHECK: vpcmpgtb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x64,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096, %xmm6, %xmm6 

// CHECK: vpcmpgtb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x64,0x7c,0x82,0xc0]      
vpcmpgtb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x64,0x7c,0x82,0x40]      
vpcmpgtb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x64,0x74,0x82,0xc0]      
vpcmpgtb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x64,0x74,0x82,0x40]      
vpcmpgtb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x64,0x7c,0x02,0x40]      
vpcmpgtb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpgtb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x64,0x74,0x02,0x40]      
vpcmpgtb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpgtb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x64,0x7a,0x40]      
vpcmpgtb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x64,0x72,0x40]      
vpcmpgtb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x64,0x3a]      
vpcmpgtb (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x64,0x32]      
vpcmpgtb (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x64,0xff]      
vpcmpgtb %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpgtb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x64,0xf6]      
vpcmpgtb %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpgtd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x66,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096, %xmm15, %xmm15 

// CHECK: vpcmpgtd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x66,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096, %xmm6, %xmm6 

// CHECK: vpcmpgtd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x66,0x7c,0x82,0xc0]      
vpcmpgtd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x66,0x7c,0x82,0x40]      
vpcmpgtd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x66,0x74,0x82,0xc0]      
vpcmpgtd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x66,0x74,0x82,0x40]      
vpcmpgtd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x66,0x7c,0x02,0x40]      
vpcmpgtd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpgtd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x66,0x74,0x02,0x40]      
vpcmpgtd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpgtd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x66,0x7a,0x40]      
vpcmpgtd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x66,0x72,0x40]      
vpcmpgtd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x66,0x3a]      
vpcmpgtd (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x66,0x32]      
vpcmpgtd (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x66,0xff]      
vpcmpgtd %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpgtd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x66,0xf6]      
vpcmpgtd %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpgtq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x37,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096, %xmm15, %xmm15 

// CHECK: vpcmpgtq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x37,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096, %xmm6, %xmm6 

// CHECK: vpcmpgtq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x37,0x7c,0x82,0xc0]      
vpcmpgtq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x37,0x7c,0x82,0x40]      
vpcmpgtq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x37,0x74,0x82,0xc0]      
vpcmpgtq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x37,0x74,0x82,0x40]      
vpcmpgtq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x37,0x7c,0x02,0x40]      
vpcmpgtq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpgtq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x37,0x74,0x02,0x40]      
vpcmpgtq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpgtq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x37,0x7a,0x40]      
vpcmpgtq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x37,0x72,0x40]      
vpcmpgtq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x37,0x3a]      
vpcmpgtq (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x37,0x32]      
vpcmpgtq (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x37,0xff]      
vpcmpgtq %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpgtq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x37,0xf6]      
vpcmpgtq %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpgtw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x65,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096, %xmm15, %xmm15 

// CHECK: vpcmpgtw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x65,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096, %xmm6, %xmm6 

// CHECK: vpcmpgtw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x65,0x7c,0x82,0xc0]      
vpcmpgtw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x65,0x7c,0x82,0x40]      
vpcmpgtw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpcmpgtw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x65,0x74,0x82,0xc0]      
vpcmpgtw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x65,0x74,0x82,0x40]      
vpcmpgtw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpcmpgtw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x65,0x7c,0x02,0x40]      
vpcmpgtw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpcmpgtw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x65,0x74,0x02,0x40]      
vpcmpgtw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpcmpgtw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x65,0x7a,0x40]      
vpcmpgtw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x65,0x72,0x40]      
vpcmpgtw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x65,0x3a]      
vpcmpgtw (%rdx), %xmm15, %xmm15 

// CHECK: vpcmpgtw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x65,0x32]      
vpcmpgtw (%rdx), %xmm6, %xmm6 

// CHECK: vpcmpgtw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x65,0xff]      
vpcmpgtw %xmm15, %xmm15, %xmm15 

// CHECK: vpcmpgtw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x65,0xf6]      
vpcmpgtw %xmm6, %xmm6, %xmm6 

// CHECK: vpcmpistri $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x63,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistri $0, 485498096, %xmm15 

// CHECK: vpcmpistri $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistri $0, 485498096, %xmm6 

// CHECK: vpcmpistri $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x63,0x7c,0x82,0xc0,0x00]      
vpcmpistri $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpistri $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x63,0x7c,0x82,0x40,0x00]      
vpcmpistri $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpistri $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x74,0x82,0xc0,0x00]      
vpcmpistri $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpistri $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x74,0x82,0x40,0x00]      
vpcmpistri $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpistri $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x63,0x7c,0x02,0x40,0x00]      
vpcmpistri $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpcmpistri $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x74,0x02,0x40,0x00]      
vpcmpistri $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpcmpistri $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x63,0x7a,0x40,0x00]      
vpcmpistri $0, 64(%rdx), %xmm15 

// CHECK: vpcmpistri $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x72,0x40,0x00]      
vpcmpistri $0, 64(%rdx), %xmm6 

// CHECK: vpcmpistri $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x63,0x3a,0x00]      
vpcmpistri $0, (%rdx), %xmm15 

// CHECK: vpcmpistri $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x32,0x00]      
vpcmpistri $0, (%rdx), %xmm6 

// CHECK: vpcmpistri $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x63,0xff,0x00]      
vpcmpistri $0, %xmm15, %xmm15 

// CHECK: vpcmpistri $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0xf6,0x00]      
vpcmpistri $0, %xmm6, %xmm6 

// CHECK: vpcmpistrm $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x62,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistrm $0, 485498096, %xmm15 

// CHECK: vpcmpistrm $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpcmpistrm $0, 485498096, %xmm6 

// CHECK: vpcmpistrm $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x62,0x7c,0x82,0xc0,0x00]      
vpcmpistrm $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpistrm $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x62,0x7c,0x82,0x40,0x00]      
vpcmpistrm $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpcmpistrm $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x74,0x82,0xc0,0x00]      
vpcmpistrm $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpistrm $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x74,0x82,0x40,0x00]      
vpcmpistrm $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpcmpistrm $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x62,0x7c,0x02,0x40,0x00]      
vpcmpistrm $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpcmpistrm $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x74,0x02,0x40,0x00]      
vpcmpistrm $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpcmpistrm $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x62,0x7a,0x40,0x00]      
vpcmpistrm $0, 64(%rdx), %xmm15 

// CHECK: vpcmpistrm $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x72,0x40,0x00]      
vpcmpistrm $0, 64(%rdx), %xmm6 

// CHECK: vpcmpistrm $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x62,0x3a,0x00]      
vpcmpistrm $0, (%rdx), %xmm15 

// CHECK: vpcmpistrm $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x32,0x00]      
vpcmpistrm $0, (%rdx), %xmm6 

// CHECK: vpcmpistrm $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x62,0xff,0x00]      
vpcmpistrm $0, %xmm15, %xmm15 

// CHECK: vpcmpistrm $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0xf6,0x00]      
vpcmpistrm $0, %xmm6, %xmm6 

// CHECK: vperm2f128 $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x06,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2f128 $0, 485498096, %ymm7, %ymm7 

// CHECK: vperm2f128 $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x06,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2f128 $0, 485498096, %ymm9, %ymm9 

// CHECK: vperm2f128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x06,0x7c,0x82,0xc0,0x00]     
vperm2f128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vperm2f128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x06,0x7c,0x82,0x40,0x00]     
vperm2f128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vperm2f128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x06,0x4c,0x82,0xc0,0x00]     
vperm2f128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vperm2f128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x06,0x4c,0x82,0x40,0x00]     
vperm2f128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vperm2f128 $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x06,0x7c,0x02,0x40,0x00]     
vperm2f128 $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vperm2f128 $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x06,0x4c,0x02,0x40,0x00]     
vperm2f128 $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vperm2f128 $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x06,0x7a,0x40,0x00]     
vperm2f128 $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vperm2f128 $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x06,0x4a,0x40,0x00]     
vperm2f128 $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vperm2f128 $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x06,0x3a,0x00]     
vperm2f128 $0, (%rdx), %ymm7, %ymm7 

// CHECK: vperm2f128 $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x06,0x0a,0x00]     
vperm2f128 $0, (%rdx), %ymm9, %ymm9 

// CHECK: vperm2f128 $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x06,0xff,0x00]     
vperm2f128 $0, %ymm7, %ymm7, %ymm7 

// CHECK: vperm2f128 $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x06,0xc9,0x00]     
vperm2f128 $0, %ymm9, %ymm9, %ymm9 

// CHECK: vpermilpd $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x05,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096, %xmm15 

// CHECK: vpermilpd $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096, %xmm6 

// CHECK: vpermilpd $0, 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096, %ymm7 

// CHECK: vpermilpd $0, 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x05,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilpd $0, 485498096, %ymm9 

// CHECK: vpermilpd $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x05,0x7c,0x82,0xc0,0x00]      
vpermilpd $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpermilpd $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x05,0x7c,0x82,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpermilpd $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x74,0x82,0xc0,0x00]      
vpermilpd $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpermilpd $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x74,0x82,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpermilpd $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x7c,0x82,0xc0,0x00]      
vpermilpd $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vpermilpd $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x7c,0x82,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vpermilpd $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x05,0x4c,0x82,0xc0,0x00]      
vpermilpd $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vpermilpd $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x05,0x4c,0x82,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vpermilpd $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x05,0x7c,0x02,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpermilpd $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x74,0x02,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpermilpd $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x7c,0x02,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax), %ymm7 

// CHECK: vpermilpd $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x05,0x4c,0x02,0x40,0x00]      
vpermilpd $0, 64(%rdx,%rax), %ymm9 

// CHECK: vpermilpd $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x05,0x7a,0x40,0x00]      
vpermilpd $0, 64(%rdx), %xmm15 

// CHECK: vpermilpd $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x72,0x40,0x00]      
vpermilpd $0, 64(%rdx), %xmm6 

// CHECK: vpermilpd $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x7a,0x40,0x00]      
vpermilpd $0, 64(%rdx), %ymm7 

// CHECK: vpermilpd $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x05,0x4a,0x40,0x00]      
vpermilpd $0, 64(%rdx), %ymm9 

// CHECK: vpermilpd $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x05,0x3a,0x00]      
vpermilpd $0, (%rdx), %xmm15 

// CHECK: vpermilpd $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x32,0x00]      
vpermilpd $0, (%rdx), %xmm6 

// CHECK: vpermilpd $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x3a,0x00]      
vpermilpd $0, (%rdx), %ymm7 

// CHECK: vpermilpd $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x05,0x0a,0x00]      
vpermilpd $0, (%rdx), %ymm9 

// CHECK: vpermilpd $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x05,0xff,0x00]      
vpermilpd $0, %xmm15, %xmm15 

// CHECK: vpermilpd $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0xf6,0x00]      
vpermilpd $0, %xmm6, %xmm6 

// CHECK: vpermilpd $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0xff,0x00]      
vpermilpd $0, %ymm7, %ymm7 

// CHECK: vpermilpd $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x7d,0x05,0xc9,0x00]      
vpermilpd $0, %ymm9, %ymm9 

// CHECK: vpermilpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096, %xmm15, %xmm15 

// CHECK: vpermilpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096, %xmm6, %xmm6 

// CHECK: vpermilpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096, %ymm7, %ymm7 

// CHECK: vpermilpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilpd 485498096, %ymm9, %ymm9 

// CHECK: vpermilpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0d,0x7c,0x82,0xc0]      
vpermilpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpermilpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0d,0x7c,0x82,0x40]      
vpermilpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpermilpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0d,0x74,0x82,0xc0]      
vpermilpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpermilpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0d,0x74,0x82,0x40]      
vpermilpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpermilpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0d,0x7c,0x82,0xc0]      
vpermilpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermilpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0d,0x7c,0x82,0x40]      
vpermilpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermilpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0d,0x4c,0x82,0xc0]      
vpermilpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermilpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0d,0x4c,0x82,0x40]      
vpermilpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermilpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0d,0x7c,0x02,0x40]      
vpermilpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpermilpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0d,0x74,0x02,0x40]      
vpermilpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpermilpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0d,0x7c,0x02,0x40]      
vpermilpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpermilpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0d,0x4c,0x02,0x40]      
vpermilpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpermilpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0d,0x7a,0x40]      
vpermilpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpermilpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0d,0x72,0x40]      
vpermilpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpermilpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0d,0x7a,0x40]      
vpermilpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpermilpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0d,0x4a,0x40]      
vpermilpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpermilpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0d,0x3a]      
vpermilpd (%rdx), %xmm15, %xmm15 

// CHECK: vpermilpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0d,0x32]      
vpermilpd (%rdx), %xmm6, %xmm6 

// CHECK: vpermilpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0d,0x3a]      
vpermilpd (%rdx), %ymm7, %ymm7 

// CHECK: vpermilpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0d,0x0a]      
vpermilpd (%rdx), %ymm9, %ymm9 

// CHECK: vpermilpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x0d,0xff]      
vpermilpd %xmm15, %xmm15, %xmm15 

// CHECK: vpermilpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0d,0xf6]      
vpermilpd %xmm6, %xmm6, %xmm6 

// CHECK: vpermilpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0d,0xff]      
vpermilpd %ymm7, %ymm7, %ymm7 

// CHECK: vpermilpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x0d,0xc9]      
vpermilpd %ymm9, %ymm9, %ymm9 

// CHECK: vpermilps $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x04,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096, %xmm15 

// CHECK: vpermilps $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096, %xmm6 

// CHECK: vpermilps $0, 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096, %ymm7 

// CHECK: vpermilps $0, 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x04,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermilps $0, 485498096, %ymm9 

// CHECK: vpermilps $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x04,0x7c,0x82,0xc0,0x00]      
vpermilps $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpermilps $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x04,0x7c,0x82,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpermilps $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x74,0x82,0xc0,0x00]      
vpermilps $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpermilps $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x74,0x82,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpermilps $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x7c,0x82,0xc0,0x00]      
vpermilps $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vpermilps $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x7c,0x82,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vpermilps $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x04,0x4c,0x82,0xc0,0x00]      
vpermilps $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vpermilps $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x04,0x4c,0x82,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vpermilps $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x04,0x7c,0x02,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpermilps $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x74,0x02,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpermilps $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x7c,0x02,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax), %ymm7 

// CHECK: vpermilps $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x04,0x4c,0x02,0x40,0x00]      
vpermilps $0, 64(%rdx,%rax), %ymm9 

// CHECK: vpermilps $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x04,0x7a,0x40,0x00]      
vpermilps $0, 64(%rdx), %xmm15 

// CHECK: vpermilps $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x72,0x40,0x00]      
vpermilps $0, 64(%rdx), %xmm6 

// CHECK: vpermilps $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x7a,0x40,0x00]      
vpermilps $0, 64(%rdx), %ymm7 

// CHECK: vpermilps $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x04,0x4a,0x40,0x00]      
vpermilps $0, 64(%rdx), %ymm9 

// CHECK: vpermilps $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x04,0x3a,0x00]      
vpermilps $0, (%rdx), %xmm15 

// CHECK: vpermilps $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x32,0x00]      
vpermilps $0, (%rdx), %xmm6 

// CHECK: vpermilps $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x3a,0x00]      
vpermilps $0, (%rdx), %ymm7 

// CHECK: vpermilps $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x04,0x0a,0x00]      
vpermilps $0, (%rdx), %ymm9 

// CHECK: vpermilps $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x04,0xff,0x00]      
vpermilps $0, %xmm15, %xmm15 

// CHECK: vpermilps $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0xf6,0x00]      
vpermilps $0, %xmm6, %xmm6 

// CHECK: vpermilps $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0xff,0x00]      
vpermilps $0, %ymm7, %ymm7 

// CHECK: vpermilps $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x7d,0x04,0xc9,0x00]      
vpermilps $0, %ymm9, %ymm9 

// CHECK: vpermilps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096, %xmm15, %xmm15 

// CHECK: vpermilps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096, %xmm6, %xmm6 

// CHECK: vpermilps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096, %ymm7, %ymm7 

// CHECK: vpermilps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermilps 485498096, %ymm9, %ymm9 

// CHECK: vpermilps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0c,0x7c,0x82,0xc0]      
vpermilps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpermilps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0c,0x7c,0x82,0x40]      
vpermilps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpermilps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0c,0x74,0x82,0xc0]      
vpermilps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpermilps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0c,0x74,0x82,0x40]      
vpermilps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpermilps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0c,0x7c,0x82,0xc0]      
vpermilps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermilps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0c,0x7c,0x82,0x40]      
vpermilps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermilps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0c,0x4c,0x82,0xc0]      
vpermilps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermilps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0c,0x4c,0x82,0x40]      
vpermilps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermilps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0c,0x7c,0x02,0x40]      
vpermilps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpermilps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0c,0x74,0x02,0x40]      
vpermilps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpermilps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0c,0x7c,0x02,0x40]      
vpermilps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpermilps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0c,0x4c,0x02,0x40]      
vpermilps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpermilps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0c,0x7a,0x40]      
vpermilps 64(%rdx), %xmm15, %xmm15 

// CHECK: vpermilps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0c,0x72,0x40]      
vpermilps 64(%rdx), %xmm6, %xmm6 

// CHECK: vpermilps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0c,0x7a,0x40]      
vpermilps 64(%rdx), %ymm7, %ymm7 

// CHECK: vpermilps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0c,0x4a,0x40]      
vpermilps 64(%rdx), %ymm9, %ymm9 

// CHECK: vpermilps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0c,0x3a]      
vpermilps (%rdx), %xmm15, %xmm15 

// CHECK: vpermilps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0c,0x32]      
vpermilps (%rdx), %xmm6, %xmm6 

// CHECK: vpermilps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0c,0x3a]      
vpermilps (%rdx), %ymm7, %ymm7 

// CHECK: vpermilps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0c,0x0a]      
vpermilps (%rdx), %ymm9, %ymm9 

// CHECK: vpermilps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x0c,0xff]      
vpermilps %xmm15, %xmm15, %xmm15 

// CHECK: vpermilps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0c,0xf6]      
vpermilps %xmm6, %xmm6, %xmm6 

// CHECK: vpermilps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0c,0xff]      
vpermilps %ymm7, %ymm7, %ymm7 

// CHECK: vpermilps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x0c,0xc9]      
vpermilps %ymm9, %ymm9, %ymm9 

// CHECK: vpextrb $0, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrb $0, %xmm15, 485498096 

// CHECK: vpextrb $0, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0x7a,0x40,0x00]      
vpextrb $0, %xmm15, 64(%rdx) 

// CHECK: vpextrb $0, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0x7c,0x02,0x40,0x00]      
vpextrb $0, %xmm15, 64(%rdx,%rax) 

// CHECK: vpextrb $0, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0x7c,0x82,0xc0,0x00]      
vpextrb $0, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vpextrb $0, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0x7c,0x82,0x40,0x00]      
vpextrb $0, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vpextrb $0, %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x43,0x79,0x14,0xfd,0x00]      
vpextrb $0, %xmm15, %r13d 

// CHECK: vpextrb $0, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0x3a,0x00]      
vpextrb $0, %xmm15, (%rdx) 

// CHECK: vpextrb $0, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrb $0, %xmm6, 485498096 

// CHECK: vpextrb $0, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x72,0x40,0x00]      
vpextrb $0, %xmm6, 64(%rdx) 

// CHECK: vpextrb $0, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x74,0x02,0x40,0x00]      
vpextrb $0, %xmm6, 64(%rdx,%rax) 

// CHECK: vpextrb $0, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x74,0x82,0xc0,0x00]      
vpextrb $0, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vpextrb $0, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x74,0x82,0x40,0x00]      
vpextrb $0, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vpextrb $0, %xmm6, %r13d 
// CHECK: encoding: [0xc4,0xc3,0x79,0x14,0xf5,0x00]      
vpextrb $0, %xmm6, %r13d 

// CHECK: vpextrb $0, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x32,0x00]      
vpextrb $0, %xmm6, (%rdx) 

// CHECK: vpextrd $0, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrd $0, %xmm15, 485498096 

// CHECK: vpextrd $0, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0x7a,0x40,0x00]      
vpextrd $0, %xmm15, 64(%rdx) 

// CHECK: vpextrd $0, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0x7c,0x02,0x40,0x00]      
vpextrd $0, %xmm15, 64(%rdx,%rax) 

// CHECK: vpextrd $0, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0x7c,0x82,0xc0,0x00]      
vpextrd $0, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vpextrd $0, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0x7c,0x82,0x40,0x00]      
vpextrd $0, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vpextrd $0, %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x43,0x79,0x16,0xfd,0x00]      
vpextrd $0, %xmm15, %r13d 

// CHECK: vpextrd $0, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0x3a,0x00]      
vpextrd $0, %xmm15, (%rdx) 

// CHECK: vpextrd $0, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrd $0, %xmm6, 485498096 

// CHECK: vpextrd $0, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x72,0x40,0x00]      
vpextrd $0, %xmm6, 64(%rdx) 

// CHECK: vpextrd $0, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x74,0x02,0x40,0x00]      
vpextrd $0, %xmm6, 64(%rdx,%rax) 

// CHECK: vpextrd $0, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x74,0x82,0xc0,0x00]      
vpextrd $0, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vpextrd $0, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x74,0x82,0x40,0x00]      
vpextrd $0, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vpextrd $0, %xmm6, %r13d 
// CHECK: encoding: [0xc4,0xc3,0x79,0x16,0xf5,0x00]      
vpextrd $0, %xmm6, %r13d 

// CHECK: vpextrd $0, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x32,0x00]      
vpextrd $0, %xmm6, (%rdx) 

// CHECK: vpextrq $0, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrq $0, %xmm15, 485498096 

// CHECK: vpextrq $0, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0x7a,0x40,0x00]      
vpextrq $0, %xmm15, 64(%rdx) 

// CHECK: vpextrq $0, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0x7c,0x02,0x40,0x00]      
vpextrq $0, %xmm15, 64(%rdx,%rax) 

// CHECK: vpextrq $0, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0x7c,0x82,0xc0,0x00]      
vpextrq $0, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vpextrq $0, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0x7c,0x82,0x40,0x00]      
vpextrq $0, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vpextrq $0, %xmm15, %r15 
// CHECK: encoding: [0xc4,0x43,0xf9,0x16,0xff,0x00]      
vpextrq $0, %xmm15, %r15 

// CHECK: vpextrq $0, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0x3a,0x00]      
vpextrq $0, %xmm15, (%rdx) 

// CHECK: vpextrq $0, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe3,0xf9,0x16,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrq $0, %xmm6, 485498096 

// CHECK: vpextrq $0, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe3,0xf9,0x16,0x72,0x40,0x00]      
vpextrq $0, %xmm6, 64(%rdx) 

// CHECK: vpextrq $0, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe3,0xf9,0x16,0x74,0x02,0x40,0x00]      
vpextrq $0, %xmm6, 64(%rdx,%rax) 

// CHECK: vpextrq $0, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0xf9,0x16,0x74,0x82,0xc0,0x00]      
vpextrq $0, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vpextrq $0, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0xf9,0x16,0x74,0x82,0x40,0x00]      
vpextrq $0, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vpextrq $0, %xmm6, %r15 
// CHECK: encoding: [0xc4,0xc3,0xf9,0x16,0xf7,0x00]      
vpextrq $0, %xmm6, %r15 

// CHECK: vpextrq $0, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe3,0xf9,0x16,0x32,0x00]      
vpextrq $0, %xmm6, (%rdx) 

// CHECK: vpextrw $0, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x63,0x79,0x15,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrw $0, %xmm15, 485498096 

// CHECK: vpextrw $0, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x15,0x7a,0x40,0x00]      
vpextrw $0, %xmm15, 64(%rdx) 

// CHECK: vpextrw $0, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x63,0x79,0x15,0x7c,0x02,0x40,0x00]      
vpextrw $0, %xmm15, 64(%rdx,%rax) 

// CHECK: vpextrw $0, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x15,0x7c,0x82,0xc0,0x00]      
vpextrw $0, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vpextrw $0, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x79,0x15,0x7c,0x82,0x40,0x00]      
vpextrw $0, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vpextrw $0, %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x79,0xc5,0xef,0x00]      
vpextrw $0, %xmm15, %r13d 

// CHECK: vpextrw $0, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x63,0x79,0x15,0x3a,0x00]      
vpextrw $0, %xmm15, (%rdx) 

// CHECK: vpextrw $0, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpextrw $0, %xmm6, 485498096 

// CHECK: vpextrw $0, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x72,0x40,0x00]      
vpextrw $0, %xmm6, 64(%rdx) 

// CHECK: vpextrw $0, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x74,0x02,0x40,0x00]      
vpextrw $0, %xmm6, 64(%rdx,%rax) 

// CHECK: vpextrw $0, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x74,0x82,0xc0,0x00]      
vpextrw $0, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vpextrw $0, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x74,0x82,0x40,0x00]      
vpextrw $0, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vpextrw $0, %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x79,0xc5,0xee,0x00]      
vpextrw $0, %xmm6, %r13d 

// CHECK: vpextrw $0, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x32,0x00]      
vpextrw $0, %xmm6, (%rdx) 

// CHECK: vphaddd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x02,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096, %xmm15, %xmm15 

// CHECK: vphaddd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x02,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096, %xmm6, %xmm6 

// CHECK: vphaddd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x02,0x7c,0x82,0xc0]      
vphaddd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphaddd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x02,0x7c,0x82,0x40]      
vphaddd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphaddd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x02,0x74,0x82,0xc0]      
vphaddd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphaddd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x02,0x74,0x82,0x40]      
vphaddd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphaddd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x02,0x7c,0x02,0x40]      
vphaddd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vphaddd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x02,0x74,0x02,0x40]      
vphaddd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vphaddd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x02,0x7a,0x40]      
vphaddd 64(%rdx), %xmm15, %xmm15 

// CHECK: vphaddd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x02,0x72,0x40]      
vphaddd 64(%rdx), %xmm6, %xmm6 

// CHECK: vphaddd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x02,0x3a]      
vphaddd (%rdx), %xmm15, %xmm15 

// CHECK: vphaddd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x02,0x32]      
vphaddd (%rdx), %xmm6, %xmm6 

// CHECK: vphaddd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x02,0xff]      
vphaddd %xmm15, %xmm15, %xmm15 

// CHECK: vphaddd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x02,0xf6]      
vphaddd %xmm6, %xmm6, %xmm6 

// CHECK: vphaddsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x03,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096, %xmm15, %xmm15 

// CHECK: vphaddsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x03,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096, %xmm6, %xmm6 

// CHECK: vphaddsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x03,0x7c,0x82,0xc0]      
vphaddsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphaddsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x03,0x7c,0x82,0x40]      
vphaddsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphaddsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x03,0x74,0x82,0xc0]      
vphaddsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphaddsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x03,0x74,0x82,0x40]      
vphaddsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphaddsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x03,0x7c,0x02,0x40]      
vphaddsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vphaddsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x03,0x74,0x02,0x40]      
vphaddsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vphaddsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x03,0x7a,0x40]      
vphaddsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vphaddsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x03,0x72,0x40]      
vphaddsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vphaddsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x03,0x3a]      
vphaddsw (%rdx), %xmm15, %xmm15 

// CHECK: vphaddsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x03,0x32]      
vphaddsw (%rdx), %xmm6, %xmm6 

// CHECK: vphaddsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x03,0xff]      
vphaddsw %xmm15, %xmm15, %xmm15 

// CHECK: vphaddsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x03,0xf6]      
vphaddsw %xmm6, %xmm6, %xmm6 

// CHECK: vphaddw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x01,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096, %xmm15, %xmm15 

// CHECK: vphaddw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x01,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096, %xmm6, %xmm6 

// CHECK: vphaddw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x01,0x7c,0x82,0xc0]      
vphaddw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphaddw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x01,0x7c,0x82,0x40]      
vphaddw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphaddw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x01,0x74,0x82,0xc0]      
vphaddw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphaddw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x01,0x74,0x82,0x40]      
vphaddw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphaddw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x01,0x7c,0x02,0x40]      
vphaddw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vphaddw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x01,0x74,0x02,0x40]      
vphaddw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vphaddw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x01,0x7a,0x40]      
vphaddw 64(%rdx), %xmm15, %xmm15 

// CHECK: vphaddw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x01,0x72,0x40]      
vphaddw 64(%rdx), %xmm6, %xmm6 

// CHECK: vphaddw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x01,0x3a]      
vphaddw (%rdx), %xmm15, %xmm15 

// CHECK: vphaddw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x01,0x32]      
vphaddw (%rdx), %xmm6, %xmm6 

// CHECK: vphaddw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x01,0xff]      
vphaddw %xmm15, %xmm15, %xmm15 

// CHECK: vphaddw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x01,0xf6]      
vphaddw %xmm6, %xmm6, %xmm6 

// CHECK: vphminposuw 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x41,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vphminposuw 485498096, %xmm15 

// CHECK: vphminposuw 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vphminposuw 485498096, %xmm6 

// CHECK: vphminposuw -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x41,0x7c,0x82,0xc0]       
vphminposuw -64(%rdx,%rax,4), %xmm15 

// CHECK: vphminposuw 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x41,0x7c,0x82,0x40]       
vphminposuw 64(%rdx,%rax,4), %xmm15 

// CHECK: vphminposuw -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x74,0x82,0xc0]       
vphminposuw -64(%rdx,%rax,4), %xmm6 

// CHECK: vphminposuw 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x74,0x82,0x40]       
vphminposuw 64(%rdx,%rax,4), %xmm6 

// CHECK: vphminposuw 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x41,0x7c,0x02,0x40]       
vphminposuw 64(%rdx,%rax), %xmm15 

// CHECK: vphminposuw 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x74,0x02,0x40]       
vphminposuw 64(%rdx,%rax), %xmm6 

// CHECK: vphminposuw 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x41,0x7a,0x40]       
vphminposuw 64(%rdx), %xmm15 

// CHECK: vphminposuw 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x72,0x40]       
vphminposuw 64(%rdx), %xmm6 

// CHECK: vphminposuw (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x41,0x3a]       
vphminposuw (%rdx), %xmm15 

// CHECK: vphminposuw (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x32]       
vphminposuw (%rdx), %xmm6 

// CHECK: vphminposuw %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x41,0xff]       
vphminposuw %xmm15, %xmm15 

// CHECK: vphminposuw %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0xf6]       
vphminposuw %xmm6, %xmm6 

// CHECK: vphsubd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x06,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096, %xmm15, %xmm15 

// CHECK: vphsubd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x06,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096, %xmm6, %xmm6 

// CHECK: vphsubd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x06,0x7c,0x82,0xc0]      
vphsubd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphsubd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x06,0x7c,0x82,0x40]      
vphsubd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphsubd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x06,0x74,0x82,0xc0]      
vphsubd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphsubd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x06,0x74,0x82,0x40]      
vphsubd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphsubd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x06,0x7c,0x02,0x40]      
vphsubd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vphsubd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x06,0x74,0x02,0x40]      
vphsubd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vphsubd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x06,0x7a,0x40]      
vphsubd 64(%rdx), %xmm15, %xmm15 

// CHECK: vphsubd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x06,0x72,0x40]      
vphsubd 64(%rdx), %xmm6, %xmm6 

// CHECK: vphsubd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x06,0x3a]      
vphsubd (%rdx), %xmm15, %xmm15 

// CHECK: vphsubd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x06,0x32]      
vphsubd (%rdx), %xmm6, %xmm6 

// CHECK: vphsubd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x06,0xff]      
vphsubd %xmm15, %xmm15, %xmm15 

// CHECK: vphsubd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x06,0xf6]      
vphsubd %xmm6, %xmm6, %xmm6 

// CHECK: vphsubsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x07,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096, %xmm15, %xmm15 

// CHECK: vphsubsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x07,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096, %xmm6, %xmm6 

// CHECK: vphsubsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x07,0x7c,0x82,0xc0]      
vphsubsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphsubsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x07,0x7c,0x82,0x40]      
vphsubsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphsubsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x07,0x74,0x82,0xc0]      
vphsubsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphsubsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x07,0x74,0x82,0x40]      
vphsubsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphsubsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x07,0x7c,0x02,0x40]      
vphsubsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vphsubsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x07,0x74,0x02,0x40]      
vphsubsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vphsubsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x07,0x7a,0x40]      
vphsubsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vphsubsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x07,0x72,0x40]      
vphsubsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vphsubsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x07,0x3a]      
vphsubsw (%rdx), %xmm15, %xmm15 

// CHECK: vphsubsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x07,0x32]      
vphsubsw (%rdx), %xmm6, %xmm6 

// CHECK: vphsubsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x07,0xff]      
vphsubsw %xmm15, %xmm15, %xmm15 

// CHECK: vphsubsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x07,0xf6]      
vphsubsw %xmm6, %xmm6, %xmm6 

// CHECK: vphsubw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x05,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096, %xmm15, %xmm15 

// CHECK: vphsubw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x05,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096, %xmm6, %xmm6 

// CHECK: vphsubw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x05,0x7c,0x82,0xc0]      
vphsubw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphsubw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x05,0x7c,0x82,0x40]      
vphsubw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vphsubw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x05,0x74,0x82,0xc0]      
vphsubw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphsubw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x05,0x74,0x82,0x40]      
vphsubw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vphsubw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x05,0x7c,0x02,0x40]      
vphsubw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vphsubw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x05,0x74,0x02,0x40]      
vphsubw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vphsubw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x05,0x7a,0x40]      
vphsubw 64(%rdx), %xmm15, %xmm15 

// CHECK: vphsubw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x05,0x72,0x40]      
vphsubw 64(%rdx), %xmm6, %xmm6 

// CHECK: vphsubw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x05,0x3a]      
vphsubw (%rdx), %xmm15, %xmm15 

// CHECK: vphsubw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x05,0x32]      
vphsubw (%rdx), %xmm6, %xmm6 

// CHECK: vphsubw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x05,0xff]      
vphsubw %xmm15, %xmm15, %xmm15 

// CHECK: vphsubw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x05,0xf6]      
vphsubw %xmm6, %xmm6, %xmm6 

// CHECK: vpinsrb $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x20,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrb $0, 485498096, %xmm15, %xmm15 

// CHECK: vpinsrb $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x20,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrb $0, 485498096, %xmm6, %xmm6 

// CHECK: vpinsrb $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x20,0x7c,0x82,0xc0,0x00]     
vpinsrb $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrb $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x20,0x7c,0x82,0x40,0x00]     
vpinsrb $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrb $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x20,0x74,0x82,0xc0,0x00]     
vpinsrb $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrb $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x20,0x74,0x82,0x40,0x00]     
vpinsrb $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrb $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x20,0x7c,0x02,0x40,0x00]     
vpinsrb $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpinsrb $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x20,0x74,0x02,0x40,0x00]     
vpinsrb $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpinsrb $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x20,0x7a,0x40,0x00]     
vpinsrb $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpinsrb $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x20,0x72,0x40,0x00]     
vpinsrb $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpinsrb $0, %r13d, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x20,0xfd,0x00]     
vpinsrb $0, %r13d, %xmm15, %xmm15 

// CHECK: vpinsrb $0, %r13d, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc3,0x49,0x20,0xf5,0x00]     
vpinsrb $0, %r13d, %xmm6, %xmm6 

// CHECK: vpinsrb $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x20,0x3a,0x00]     
vpinsrb $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpinsrb $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x20,0x32,0x00]     
vpinsrb $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpinsrd $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x22,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrd $0, 485498096, %xmm15, %xmm15 

// CHECK: vpinsrd $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x22,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrd $0, 485498096, %xmm6, %xmm6 

// CHECK: vpinsrd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x22,0x7c,0x82,0xc0,0x00]     
vpinsrd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x22,0x7c,0x82,0x40,0x00]     
vpinsrd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x22,0x74,0x82,0xc0,0x00]     
vpinsrd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x22,0x74,0x82,0x40,0x00]     
vpinsrd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrd $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x22,0x7c,0x02,0x40,0x00]     
vpinsrd $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpinsrd $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x22,0x74,0x02,0x40,0x00]     
vpinsrd $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpinsrd $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x22,0x7a,0x40,0x00]     
vpinsrd $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpinsrd $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x22,0x72,0x40,0x00]     
vpinsrd $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpinsrd $0, %r13d, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x22,0xfd,0x00]     
vpinsrd $0, %r13d, %xmm15, %xmm15 

// CHECK: vpinsrd $0, %r13d, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc3,0x49,0x22,0xf5,0x00]     
vpinsrd $0, %r13d, %xmm6, %xmm6 

// CHECK: vpinsrd $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x22,0x3a,0x00]     
vpinsrd $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpinsrd $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x22,0x32,0x00]     
vpinsrd $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpinsrq $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x81,0x22,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrq $0, 485498096, %xmm15, %xmm15 

// CHECK: vpinsrq $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0xc9,0x22,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrq $0, 485498096, %xmm6, %xmm6 

// CHECK: vpinsrq $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x81,0x22,0x7c,0x82,0xc0,0x00]     
vpinsrq $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrq $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x81,0x22,0x7c,0x82,0x40,0x00]     
vpinsrq $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrq $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0xc9,0x22,0x74,0x82,0xc0,0x00]     
vpinsrq $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrq $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0xc9,0x22,0x74,0x82,0x40,0x00]     
vpinsrq $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrq $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x81,0x22,0x7c,0x02,0x40,0x00]     
vpinsrq $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpinsrq $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0xc9,0x22,0x74,0x02,0x40,0x00]     
vpinsrq $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpinsrq $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x81,0x22,0x7a,0x40,0x00]     
vpinsrq $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpinsrq $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0xc9,0x22,0x72,0x40,0x00]     
vpinsrq $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpinsrq $0, %r15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x81,0x22,0xff,0x00]     
vpinsrq $0, %r15, %xmm15, %xmm15 

// CHECK: vpinsrq $0, %r15, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc3,0xc9,0x22,0xf7,0x00]     
vpinsrq $0, %r15, %xmm6, %xmm6 

// CHECK: vpinsrq $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x81,0x22,0x3a,0x00]     
vpinsrq $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpinsrq $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0xc9,0x22,0x32,0x00]     
vpinsrq $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpinsrw $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc4,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrw $0, 485498096, %xmm15, %xmm15 

// CHECK: vpinsrw $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc4,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpinsrw $0, 485498096, %xmm6, %xmm6 

// CHECK: vpinsrw $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc4,0x7c,0x82,0xc0,0x00]     
vpinsrw $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrw $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc4,0x7c,0x82,0x40,0x00]     
vpinsrw $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpinsrw $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc4,0x74,0x82,0xc0,0x00]     
vpinsrw $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrw $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc4,0x74,0x82,0x40,0x00]     
vpinsrw $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpinsrw $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc4,0x7c,0x02,0x40,0x00]     
vpinsrw $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpinsrw $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc4,0x74,0x02,0x40,0x00]     
vpinsrw $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpinsrw $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc4,0x7a,0x40,0x00]     
vpinsrw $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpinsrw $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc4,0x72,0x40,0x00]     
vpinsrw $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpinsrw $0, %r13d, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xc4,0xfd,0x00]     
vpinsrw $0, %r13d, %xmm15, %xmm15 

// CHECK: vpinsrw $0, %r13d, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xc1,0x49,0xc4,0xf5,0x00]     
vpinsrw $0, %r13d, %xmm6, %xmm6 

// CHECK: vpinsrw $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc4,0x3a,0x00]     
vpinsrw $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpinsrw $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc4,0x32,0x00]     
vpinsrw $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpmaddubsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x04,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096, %xmm15, %xmm15 

// CHECK: vpmaddubsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x04,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096, %xmm6, %xmm6 

// CHECK: vpmaddubsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x04,0x7c,0x82,0xc0]      
vpmaddubsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaddubsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x04,0x7c,0x82,0x40]      
vpmaddubsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaddubsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x04,0x74,0x82,0xc0]      
vpmaddubsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaddubsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x04,0x74,0x82,0x40]      
vpmaddubsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaddubsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x04,0x7c,0x02,0x40]      
vpmaddubsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaddubsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x04,0x74,0x02,0x40]      
vpmaddubsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaddubsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x04,0x7a,0x40]      
vpmaddubsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaddubsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x04,0x72,0x40]      
vpmaddubsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaddubsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x04,0x3a]      
vpmaddubsw (%rdx), %xmm15, %xmm15 

// CHECK: vpmaddubsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x04,0x32]      
vpmaddubsw (%rdx), %xmm6, %xmm6 

// CHECK: vpmaddubsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x04,0xff]      
vpmaddubsw %xmm15, %xmm15, %xmm15 

// CHECK: vpmaddubsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x04,0xf6]      
vpmaddubsw %xmm6, %xmm6, %xmm6 

// CHECK: vpmaddwd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf5,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096, %xmm15, %xmm15 

// CHECK: vpmaddwd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf5,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096, %xmm6, %xmm6 

// CHECK: vpmaddwd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf5,0x7c,0x82,0xc0]      
vpmaddwd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaddwd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf5,0x7c,0x82,0x40]      
vpmaddwd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaddwd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf5,0x74,0x82,0xc0]      
vpmaddwd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaddwd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf5,0x74,0x82,0x40]      
vpmaddwd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaddwd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf5,0x7c,0x02,0x40]      
vpmaddwd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaddwd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf5,0x74,0x02,0x40]      
vpmaddwd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaddwd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf5,0x7a,0x40]      
vpmaddwd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaddwd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf5,0x72,0x40]      
vpmaddwd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaddwd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf5,0x3a]      
vpmaddwd (%rdx), %xmm15, %xmm15 

// CHECK: vpmaddwd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf5,0x32]      
vpmaddwd (%rdx), %xmm6, %xmm6 

// CHECK: vpmaddwd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf5,0xff]      
vpmaddwd %xmm15, %xmm15, %xmm15 

// CHECK: vpmaddwd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf5,0xf6]      
vpmaddwd %xmm6, %xmm6, %xmm6 

// CHECK: vpmaxsb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096, %xmm15, %xmm15 

// CHECK: vpmaxsb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096, %xmm6, %xmm6 

// CHECK: vpmaxsb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3c,0x7c,0x82,0xc0]      
vpmaxsb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxsb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3c,0x7c,0x82,0x40]      
vpmaxsb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxsb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3c,0x74,0x82,0xc0]      
vpmaxsb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxsb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3c,0x74,0x82,0x40]      
vpmaxsb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxsb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3c,0x7c,0x02,0x40]      
vpmaxsb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaxsb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3c,0x74,0x02,0x40]      
vpmaxsb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaxsb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3c,0x7a,0x40]      
vpmaxsb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaxsb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3c,0x72,0x40]      
vpmaxsb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaxsb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3c,0x3a]      
vpmaxsb (%rdx), %xmm15, %xmm15 

// CHECK: vpmaxsb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3c,0x32]      
vpmaxsb (%rdx), %xmm6, %xmm6 

// CHECK: vpmaxsb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x3c,0xff]      
vpmaxsb %xmm15, %xmm15, %xmm15 

// CHECK: vpmaxsb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3c,0xf6]      
vpmaxsb %xmm6, %xmm6, %xmm6 

// CHECK: vpmaxsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096, %xmm15, %xmm15 

// CHECK: vpmaxsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096, %xmm6, %xmm6 

// CHECK: vpmaxsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3d,0x7c,0x82,0xc0]      
vpmaxsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3d,0x7c,0x82,0x40]      
vpmaxsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3d,0x74,0x82,0xc0]      
vpmaxsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3d,0x74,0x82,0x40]      
vpmaxsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3d,0x7c,0x02,0x40]      
vpmaxsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaxsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3d,0x74,0x02,0x40]      
vpmaxsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaxsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3d,0x7a,0x40]      
vpmaxsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaxsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3d,0x72,0x40]      
vpmaxsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaxsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3d,0x3a]      
vpmaxsd (%rdx), %xmm15, %xmm15 

// CHECK: vpmaxsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3d,0x32]      
vpmaxsd (%rdx), %xmm6, %xmm6 

// CHECK: vpmaxsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x3d,0xff]      
vpmaxsd %xmm15, %xmm15, %xmm15 

// CHECK: vpmaxsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3d,0xf6]      
vpmaxsd %xmm6, %xmm6, %xmm6 

// CHECK: vpmaxsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xee,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096, %xmm15, %xmm15 

// CHECK: vpmaxsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xee,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096, %xmm6, %xmm6 

// CHECK: vpmaxsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xee,0x7c,0x82,0xc0]      
vpmaxsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xee,0x7c,0x82,0x40]      
vpmaxsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xee,0x74,0x82,0xc0]      
vpmaxsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xee,0x74,0x82,0x40]      
vpmaxsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xee,0x7c,0x02,0x40]      
vpmaxsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaxsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xee,0x74,0x02,0x40]      
vpmaxsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaxsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xee,0x7a,0x40]      
vpmaxsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaxsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xee,0x72,0x40]      
vpmaxsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaxsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xee,0x3a]      
vpmaxsw (%rdx), %xmm15, %xmm15 

// CHECK: vpmaxsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xee,0x32]      
vpmaxsw (%rdx), %xmm6, %xmm6 

// CHECK: vpmaxsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xee,0xff]      
vpmaxsw %xmm15, %xmm15, %xmm15 

// CHECK: vpmaxsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xee,0xf6]      
vpmaxsw %xmm6, %xmm6, %xmm6 

// CHECK: vpmaxub 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xde,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096, %xmm15, %xmm15 

// CHECK: vpmaxub 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xde,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096, %xmm6, %xmm6 

// CHECK: vpmaxub -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xde,0x7c,0x82,0xc0]      
vpmaxub -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxub 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xde,0x7c,0x82,0x40]      
vpmaxub 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxub -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xde,0x74,0x82,0xc0]      
vpmaxub -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxub 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xde,0x74,0x82,0x40]      
vpmaxub 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxub 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xde,0x7c,0x02,0x40]      
vpmaxub 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaxub 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xde,0x74,0x02,0x40]      
vpmaxub 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaxub 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xde,0x7a,0x40]      
vpmaxub 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaxub 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xde,0x72,0x40]      
vpmaxub 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaxub (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xde,0x3a]      
vpmaxub (%rdx), %xmm15, %xmm15 

// CHECK: vpmaxub (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xde,0x32]      
vpmaxub (%rdx), %xmm6, %xmm6 

// CHECK: vpmaxub %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xde,0xff]      
vpmaxub %xmm15, %xmm15, %xmm15 

// CHECK: vpmaxub %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xde,0xf6]      
vpmaxub %xmm6, %xmm6, %xmm6 

// CHECK: vpmaxud 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096, %xmm15, %xmm15 

// CHECK: vpmaxud 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096, %xmm6, %xmm6 

// CHECK: vpmaxud -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3f,0x7c,0x82,0xc0]      
vpmaxud -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxud 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3f,0x7c,0x82,0x40]      
vpmaxud 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxud -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3f,0x74,0x82,0xc0]      
vpmaxud -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxud 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3f,0x74,0x82,0x40]      
vpmaxud 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxud 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3f,0x7c,0x02,0x40]      
vpmaxud 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaxud 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3f,0x74,0x02,0x40]      
vpmaxud 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaxud 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3f,0x7a,0x40]      
vpmaxud 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaxud 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3f,0x72,0x40]      
vpmaxud 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaxud (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3f,0x3a]      
vpmaxud (%rdx), %xmm15, %xmm15 

// CHECK: vpmaxud (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3f,0x32]      
vpmaxud (%rdx), %xmm6, %xmm6 

// CHECK: vpmaxud %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x3f,0xff]      
vpmaxud %xmm15, %xmm15, %xmm15 

// CHECK: vpmaxud %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3f,0xf6]      
vpmaxud %xmm6, %xmm6, %xmm6 

// CHECK: vpmaxuw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096, %xmm15, %xmm15 

// CHECK: vpmaxuw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096, %xmm6, %xmm6 

// CHECK: vpmaxuw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3e,0x7c,0x82,0xc0]      
vpmaxuw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxuw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3e,0x7c,0x82,0x40]      
vpmaxuw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaxuw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3e,0x74,0x82,0xc0]      
vpmaxuw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxuw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3e,0x74,0x82,0x40]      
vpmaxuw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaxuw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3e,0x7c,0x02,0x40]      
vpmaxuw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaxuw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3e,0x74,0x02,0x40]      
vpmaxuw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaxuw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3e,0x7a,0x40]      
vpmaxuw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaxuw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3e,0x72,0x40]      
vpmaxuw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaxuw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3e,0x3a]      
vpmaxuw (%rdx), %xmm15, %xmm15 

// CHECK: vpmaxuw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3e,0x32]      
vpmaxuw (%rdx), %xmm6, %xmm6 

// CHECK: vpmaxuw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x3e,0xff]      
vpmaxuw %xmm15, %xmm15, %xmm15 

// CHECK: vpmaxuw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3e,0xf6]      
vpmaxuw %xmm6, %xmm6, %xmm6 

// CHECK: vpminsb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x38,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096, %xmm15, %xmm15 

// CHECK: vpminsb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x38,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096, %xmm6, %xmm6 

// CHECK: vpminsb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x38,0x7c,0x82,0xc0]      
vpminsb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminsb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x38,0x7c,0x82,0x40]      
vpminsb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminsb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x38,0x74,0x82,0xc0]      
vpminsb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminsb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x38,0x74,0x82,0x40]      
vpminsb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminsb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x38,0x7c,0x02,0x40]      
vpminsb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpminsb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x38,0x74,0x02,0x40]      
vpminsb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpminsb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x38,0x7a,0x40]      
vpminsb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpminsb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x38,0x72,0x40]      
vpminsb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpminsb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x38,0x3a]      
vpminsb (%rdx), %xmm15, %xmm15 

// CHECK: vpminsb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x38,0x32]      
vpminsb (%rdx), %xmm6, %xmm6 

// CHECK: vpminsb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x38,0xff]      
vpminsb %xmm15, %xmm15, %xmm15 

// CHECK: vpminsb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x38,0xf6]      
vpminsb %xmm6, %xmm6, %xmm6 

// CHECK: vpminsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x39,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096, %xmm15, %xmm15 

// CHECK: vpminsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x39,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096, %xmm6, %xmm6 

// CHECK: vpminsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x39,0x7c,0x82,0xc0]      
vpminsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x39,0x7c,0x82,0x40]      
vpminsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x39,0x74,0x82,0xc0]      
vpminsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x39,0x74,0x82,0x40]      
vpminsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x39,0x7c,0x02,0x40]      
vpminsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpminsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x39,0x74,0x02,0x40]      
vpminsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpminsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x39,0x7a,0x40]      
vpminsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpminsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x39,0x72,0x40]      
vpminsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpminsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x39,0x3a]      
vpminsd (%rdx), %xmm15, %xmm15 

// CHECK: vpminsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x39,0x32]      
vpminsd (%rdx), %xmm6, %xmm6 

// CHECK: vpminsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x39,0xff]      
vpminsd %xmm15, %xmm15, %xmm15 

// CHECK: vpminsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x39,0xf6]      
vpminsd %xmm6, %xmm6, %xmm6 

// CHECK: vpminsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xea,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096, %xmm15, %xmm15 

// CHECK: vpminsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xea,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096, %xmm6, %xmm6 

// CHECK: vpminsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xea,0x7c,0x82,0xc0]      
vpminsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xea,0x7c,0x82,0x40]      
vpminsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xea,0x74,0x82,0xc0]      
vpminsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xea,0x74,0x82,0x40]      
vpminsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xea,0x7c,0x02,0x40]      
vpminsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpminsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xea,0x74,0x02,0x40]      
vpminsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpminsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xea,0x7a,0x40]      
vpminsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpminsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xea,0x72,0x40]      
vpminsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpminsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xea,0x3a]      
vpminsw (%rdx), %xmm15, %xmm15 

// CHECK: vpminsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xea,0x32]      
vpminsw (%rdx), %xmm6, %xmm6 

// CHECK: vpminsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xea,0xff]      
vpminsw %xmm15, %xmm15, %xmm15 

// CHECK: vpminsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xea,0xf6]      
vpminsw %xmm6, %xmm6, %xmm6 

// CHECK: vpminub 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xda,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096, %xmm15, %xmm15 

// CHECK: vpminub 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xda,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096, %xmm6, %xmm6 

// CHECK: vpminub -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xda,0x7c,0x82,0xc0]      
vpminub -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminub 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xda,0x7c,0x82,0x40]      
vpminub 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminub -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xda,0x74,0x82,0xc0]      
vpminub -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminub 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xda,0x74,0x82,0x40]      
vpminub 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminub 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xda,0x7c,0x02,0x40]      
vpminub 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpminub 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xda,0x74,0x02,0x40]      
vpminub 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpminub 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xda,0x7a,0x40]      
vpminub 64(%rdx), %xmm15, %xmm15 

// CHECK: vpminub 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xda,0x72,0x40]      
vpminub 64(%rdx), %xmm6, %xmm6 

// CHECK: vpminub (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xda,0x3a]      
vpminub (%rdx), %xmm15, %xmm15 

// CHECK: vpminub (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xda,0x32]      
vpminub (%rdx), %xmm6, %xmm6 

// CHECK: vpminub %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xda,0xff]      
vpminub %xmm15, %xmm15, %xmm15 

// CHECK: vpminub %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xda,0xf6]      
vpminub %xmm6, %xmm6, %xmm6 

// CHECK: vpminud 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096, %xmm15, %xmm15 

// CHECK: vpminud 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096, %xmm6, %xmm6 

// CHECK: vpminud -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3b,0x7c,0x82,0xc0]      
vpminud -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminud 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3b,0x7c,0x82,0x40]      
vpminud 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminud -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3b,0x74,0x82,0xc0]      
vpminud -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminud 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3b,0x74,0x82,0x40]      
vpminud 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminud 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3b,0x7c,0x02,0x40]      
vpminud 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpminud 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3b,0x74,0x02,0x40]      
vpminud 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpminud 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3b,0x7a,0x40]      
vpminud 64(%rdx), %xmm15, %xmm15 

// CHECK: vpminud 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3b,0x72,0x40]      
vpminud 64(%rdx), %xmm6, %xmm6 

// CHECK: vpminud (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3b,0x3a]      
vpminud (%rdx), %xmm15, %xmm15 

// CHECK: vpminud (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3b,0x32]      
vpminud (%rdx), %xmm6, %xmm6 

// CHECK: vpminud %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x3b,0xff]      
vpminud %xmm15, %xmm15, %xmm15 

// CHECK: vpminud %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3b,0xf6]      
vpminud %xmm6, %xmm6, %xmm6 

// CHECK: vpminuw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096, %xmm15, %xmm15 

// CHECK: vpminuw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096, %xmm6, %xmm6 

// CHECK: vpminuw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3a,0x7c,0x82,0xc0]      
vpminuw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminuw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3a,0x7c,0x82,0x40]      
vpminuw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpminuw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3a,0x74,0x82,0xc0]      
vpminuw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminuw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3a,0x74,0x82,0x40]      
vpminuw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpminuw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3a,0x7c,0x02,0x40]      
vpminuw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpminuw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3a,0x74,0x02,0x40]      
vpminuw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpminuw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3a,0x7a,0x40]      
vpminuw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpminuw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3a,0x72,0x40]      
vpminuw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpminuw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x3a,0x3a]      
vpminuw (%rdx), %xmm15, %xmm15 

// CHECK: vpminuw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3a,0x32]      
vpminuw (%rdx), %xmm6, %xmm6 

// CHECK: vpminuw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x3a,0xff]      
vpminuw %xmm15, %xmm15, %xmm15 

// CHECK: vpminuw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x3a,0xf6]      
vpminuw %xmm6, %xmm6, %xmm6 

// CHECK: vpmovmskb %xmm15, %r13d 
// CHECK: encoding: [0xc4,0x41,0x79,0xd7,0xef]       
vpmovmskb %xmm15, %r13d 

// CHECK: vpmovmskb %xmm6, %r13d 
// CHECK: encoding: [0xc5,0x79,0xd7,0xee]       
vpmovmskb %xmm6, %r13d 

// CHECK: vpmovsxbd 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x21,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096, %xmm15 

// CHECK: vpmovsxbd 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096, %xmm6 

// CHECK: vpmovsxbd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x21,0x7c,0x82,0xc0]       
vpmovsxbd -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxbd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x21,0x7c,0x82,0x40]       
vpmovsxbd 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxbd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x74,0x82,0xc0]       
vpmovsxbd -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxbd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x74,0x82,0x40]       
vpmovsxbd 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxbd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x21,0x7c,0x02,0x40]       
vpmovsxbd 64(%rdx,%rax), %xmm15 

// CHECK: vpmovsxbd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x74,0x02,0x40]       
vpmovsxbd 64(%rdx,%rax), %xmm6 

// CHECK: vpmovsxbd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x21,0x7a,0x40]       
vpmovsxbd 64(%rdx), %xmm15 

// CHECK: vpmovsxbd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x72,0x40]       
vpmovsxbd 64(%rdx), %xmm6 

// CHECK: vpmovsxbd (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x21,0x3a]       
vpmovsxbd (%rdx), %xmm15 

// CHECK: vpmovsxbd (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x32]       
vpmovsxbd (%rdx), %xmm6 

// CHECK: vpmovsxbd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x21,0xff]       
vpmovsxbd %xmm15, %xmm15 

// CHECK: vpmovsxbd %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0xf6]       
vpmovsxbd %xmm6, %xmm6 

// CHECK: vpmovsxbq 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x22,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096, %xmm15 

// CHECK: vpmovsxbq 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096, %xmm6 

// CHECK: vpmovsxbq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x22,0x7c,0x82,0xc0]       
vpmovsxbq -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxbq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x22,0x7c,0x82,0x40]       
vpmovsxbq 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxbq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x74,0x82,0xc0]       
vpmovsxbq -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxbq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x74,0x82,0x40]       
vpmovsxbq 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxbq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x22,0x7c,0x02,0x40]       
vpmovsxbq 64(%rdx,%rax), %xmm15 

// CHECK: vpmovsxbq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x74,0x02,0x40]       
vpmovsxbq 64(%rdx,%rax), %xmm6 

// CHECK: vpmovsxbq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x22,0x7a,0x40]       
vpmovsxbq 64(%rdx), %xmm15 

// CHECK: vpmovsxbq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x72,0x40]       
vpmovsxbq 64(%rdx), %xmm6 

// CHECK: vpmovsxbq (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x22,0x3a]       
vpmovsxbq (%rdx), %xmm15 

// CHECK: vpmovsxbq (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x32]       
vpmovsxbq (%rdx), %xmm6 

// CHECK: vpmovsxbq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x22,0xff]       
vpmovsxbq %xmm15, %xmm15 

// CHECK: vpmovsxbq %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0xf6]       
vpmovsxbq %xmm6, %xmm6 

// CHECK: vpmovsxbw 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x20,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096, %xmm15 

// CHECK: vpmovsxbw 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096, %xmm6 

// CHECK: vpmovsxbw -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x20,0x7c,0x82,0xc0]       
vpmovsxbw -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxbw 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x20,0x7c,0x82,0x40]       
vpmovsxbw 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxbw -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x74,0x82,0xc0]       
vpmovsxbw -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxbw 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x74,0x82,0x40]       
vpmovsxbw 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxbw 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x20,0x7c,0x02,0x40]       
vpmovsxbw 64(%rdx,%rax), %xmm15 

// CHECK: vpmovsxbw 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x74,0x02,0x40]       
vpmovsxbw 64(%rdx,%rax), %xmm6 

// CHECK: vpmovsxbw 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x20,0x7a,0x40]       
vpmovsxbw 64(%rdx), %xmm15 

// CHECK: vpmovsxbw 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x72,0x40]       
vpmovsxbw 64(%rdx), %xmm6 

// CHECK: vpmovsxbw (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x20,0x3a]       
vpmovsxbw (%rdx), %xmm15 

// CHECK: vpmovsxbw (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x32]       
vpmovsxbw (%rdx), %xmm6 

// CHECK: vpmovsxbw %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x20,0xff]       
vpmovsxbw %xmm15, %xmm15 

// CHECK: vpmovsxbw %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0xf6]       
vpmovsxbw %xmm6, %xmm6 

// CHECK: vpmovsxdq 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x25,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096, %xmm15 

// CHECK: vpmovsxdq 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096, %xmm6 

// CHECK: vpmovsxdq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x25,0x7c,0x82,0xc0]       
vpmovsxdq -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxdq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x25,0x7c,0x82,0x40]       
vpmovsxdq 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxdq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x74,0x82,0xc0]       
vpmovsxdq -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxdq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x74,0x82,0x40]       
vpmovsxdq 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxdq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x25,0x7c,0x02,0x40]       
vpmovsxdq 64(%rdx,%rax), %xmm15 

// CHECK: vpmovsxdq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x74,0x02,0x40]       
vpmovsxdq 64(%rdx,%rax), %xmm6 

// CHECK: vpmovsxdq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x25,0x7a,0x40]       
vpmovsxdq 64(%rdx), %xmm15 

// CHECK: vpmovsxdq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x72,0x40]       
vpmovsxdq 64(%rdx), %xmm6 

// CHECK: vpmovsxdq (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x25,0x3a]       
vpmovsxdq (%rdx), %xmm15 

// CHECK: vpmovsxdq (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x32]       
vpmovsxdq (%rdx), %xmm6 

// CHECK: vpmovsxdq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x25,0xff]       
vpmovsxdq %xmm15, %xmm15 

// CHECK: vpmovsxdq %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0xf6]       
vpmovsxdq %xmm6, %xmm6 

// CHECK: vpmovsxwd 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x23,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096, %xmm15 

// CHECK: vpmovsxwd 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096, %xmm6 

// CHECK: vpmovsxwd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x23,0x7c,0x82,0xc0]       
vpmovsxwd -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxwd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x23,0x7c,0x82,0x40]       
vpmovsxwd 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxwd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x74,0x82,0xc0]       
vpmovsxwd -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxwd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x74,0x82,0x40]       
vpmovsxwd 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxwd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x23,0x7c,0x02,0x40]       
vpmovsxwd 64(%rdx,%rax), %xmm15 

// CHECK: vpmovsxwd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x74,0x02,0x40]       
vpmovsxwd 64(%rdx,%rax), %xmm6 

// CHECK: vpmovsxwd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x23,0x7a,0x40]       
vpmovsxwd 64(%rdx), %xmm15 

// CHECK: vpmovsxwd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x72,0x40]       
vpmovsxwd 64(%rdx), %xmm6 

// CHECK: vpmovsxwd (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x23,0x3a]       
vpmovsxwd (%rdx), %xmm15 

// CHECK: vpmovsxwd (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x32]       
vpmovsxwd (%rdx), %xmm6 

// CHECK: vpmovsxwd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x23,0xff]       
vpmovsxwd %xmm15, %xmm15 

// CHECK: vpmovsxwd %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0xf6]       
vpmovsxwd %xmm6, %xmm6 

// CHECK: vpmovsxwq 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x24,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096, %xmm15 

// CHECK: vpmovsxwq 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096, %xmm6 

// CHECK: vpmovsxwq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x24,0x7c,0x82,0xc0]       
vpmovsxwq -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxwq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x24,0x7c,0x82,0x40]       
vpmovsxwq 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovsxwq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x74,0x82,0xc0]       
vpmovsxwq -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxwq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x74,0x82,0x40]       
vpmovsxwq 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovsxwq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x24,0x7c,0x02,0x40]       
vpmovsxwq 64(%rdx,%rax), %xmm15 

// CHECK: vpmovsxwq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x74,0x02,0x40]       
vpmovsxwq 64(%rdx,%rax), %xmm6 

// CHECK: vpmovsxwq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x24,0x7a,0x40]       
vpmovsxwq 64(%rdx), %xmm15 

// CHECK: vpmovsxwq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x72,0x40]       
vpmovsxwq 64(%rdx), %xmm6 

// CHECK: vpmovsxwq (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x24,0x3a]       
vpmovsxwq (%rdx), %xmm15 

// CHECK: vpmovsxwq (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x32]       
vpmovsxwq (%rdx), %xmm6 

// CHECK: vpmovsxwq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x24,0xff]       
vpmovsxwq %xmm15, %xmm15 

// CHECK: vpmovsxwq %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0xf6]       
vpmovsxwq %xmm6, %xmm6 

// CHECK: vpmovzxbd 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x31,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096, %xmm15 

// CHECK: vpmovzxbd 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096, %xmm6 

// CHECK: vpmovzxbd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x31,0x7c,0x82,0xc0]       
vpmovzxbd -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxbd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x31,0x7c,0x82,0x40]       
vpmovzxbd 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxbd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x74,0x82,0xc0]       
vpmovzxbd -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxbd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x74,0x82,0x40]       
vpmovzxbd 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxbd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x31,0x7c,0x02,0x40]       
vpmovzxbd 64(%rdx,%rax), %xmm15 

// CHECK: vpmovzxbd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x74,0x02,0x40]       
vpmovzxbd 64(%rdx,%rax), %xmm6 

// CHECK: vpmovzxbd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x31,0x7a,0x40]       
vpmovzxbd 64(%rdx), %xmm15 

// CHECK: vpmovzxbd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x72,0x40]       
vpmovzxbd 64(%rdx), %xmm6 

// CHECK: vpmovzxbd (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x31,0x3a]       
vpmovzxbd (%rdx), %xmm15 

// CHECK: vpmovzxbd (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x32]       
vpmovzxbd (%rdx), %xmm6 

// CHECK: vpmovzxbd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x31,0xff]       
vpmovzxbd %xmm15, %xmm15 

// CHECK: vpmovzxbd %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0xf6]       
vpmovzxbd %xmm6, %xmm6 

// CHECK: vpmovzxbq 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x32,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096, %xmm15 

// CHECK: vpmovzxbq 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096, %xmm6 

// CHECK: vpmovzxbq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x32,0x7c,0x82,0xc0]       
vpmovzxbq -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxbq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x32,0x7c,0x82,0x40]       
vpmovzxbq 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxbq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x74,0x82,0xc0]       
vpmovzxbq -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxbq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x74,0x82,0x40]       
vpmovzxbq 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxbq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x32,0x7c,0x02,0x40]       
vpmovzxbq 64(%rdx,%rax), %xmm15 

// CHECK: vpmovzxbq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x74,0x02,0x40]       
vpmovzxbq 64(%rdx,%rax), %xmm6 

// CHECK: vpmovzxbq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x32,0x7a,0x40]       
vpmovzxbq 64(%rdx), %xmm15 

// CHECK: vpmovzxbq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x72,0x40]       
vpmovzxbq 64(%rdx), %xmm6 

// CHECK: vpmovzxbq (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x32,0x3a]       
vpmovzxbq (%rdx), %xmm15 

// CHECK: vpmovzxbq (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x32]       
vpmovzxbq (%rdx), %xmm6 

// CHECK: vpmovzxbq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x32,0xff]       
vpmovzxbq %xmm15, %xmm15 

// CHECK: vpmovzxbq %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0xf6]       
vpmovzxbq %xmm6, %xmm6 

// CHECK: vpmovzxbw 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x30,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096, %xmm15 

// CHECK: vpmovzxbw 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096, %xmm6 

// CHECK: vpmovzxbw -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x30,0x7c,0x82,0xc0]       
vpmovzxbw -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxbw 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x30,0x7c,0x82,0x40]       
vpmovzxbw 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxbw -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x74,0x82,0xc0]       
vpmovzxbw -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxbw 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x74,0x82,0x40]       
vpmovzxbw 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxbw 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x30,0x7c,0x02,0x40]       
vpmovzxbw 64(%rdx,%rax), %xmm15 

// CHECK: vpmovzxbw 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x74,0x02,0x40]       
vpmovzxbw 64(%rdx,%rax), %xmm6 

// CHECK: vpmovzxbw 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x30,0x7a,0x40]       
vpmovzxbw 64(%rdx), %xmm15 

// CHECK: vpmovzxbw 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x72,0x40]       
vpmovzxbw 64(%rdx), %xmm6 

// CHECK: vpmovzxbw (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x30,0x3a]       
vpmovzxbw (%rdx), %xmm15 

// CHECK: vpmovzxbw (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x32]       
vpmovzxbw (%rdx), %xmm6 

// CHECK: vpmovzxbw %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x30,0xff]       
vpmovzxbw %xmm15, %xmm15 

// CHECK: vpmovzxbw %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0xf6]       
vpmovzxbw %xmm6, %xmm6 

// CHECK: vpmovzxdq 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x35,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096, %xmm15 

// CHECK: vpmovzxdq 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096, %xmm6 

// CHECK: vpmovzxdq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x35,0x7c,0x82,0xc0]       
vpmovzxdq -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxdq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x35,0x7c,0x82,0x40]       
vpmovzxdq 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxdq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x74,0x82,0xc0]       
vpmovzxdq -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxdq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x74,0x82,0x40]       
vpmovzxdq 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxdq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x35,0x7c,0x02,0x40]       
vpmovzxdq 64(%rdx,%rax), %xmm15 

// CHECK: vpmovzxdq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x74,0x02,0x40]       
vpmovzxdq 64(%rdx,%rax), %xmm6 

// CHECK: vpmovzxdq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x35,0x7a,0x40]       
vpmovzxdq 64(%rdx), %xmm15 

// CHECK: vpmovzxdq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x72,0x40]       
vpmovzxdq 64(%rdx), %xmm6 

// CHECK: vpmovzxdq (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x35,0x3a]       
vpmovzxdq (%rdx), %xmm15 

// CHECK: vpmovzxdq (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x32]       
vpmovzxdq (%rdx), %xmm6 

// CHECK: vpmovzxdq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x35,0xff]       
vpmovzxdq %xmm15, %xmm15 

// CHECK: vpmovzxdq %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0xf6]       
vpmovzxdq %xmm6, %xmm6 

// CHECK: vpmovzxwd 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x33,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096, %xmm15 

// CHECK: vpmovzxwd 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096, %xmm6 

// CHECK: vpmovzxwd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x33,0x7c,0x82,0xc0]       
vpmovzxwd -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxwd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x33,0x7c,0x82,0x40]       
vpmovzxwd 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxwd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x74,0x82,0xc0]       
vpmovzxwd -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxwd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x74,0x82,0x40]       
vpmovzxwd 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxwd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x33,0x7c,0x02,0x40]       
vpmovzxwd 64(%rdx,%rax), %xmm15 

// CHECK: vpmovzxwd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x74,0x02,0x40]       
vpmovzxwd 64(%rdx,%rax), %xmm6 

// CHECK: vpmovzxwd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x33,0x7a,0x40]       
vpmovzxwd 64(%rdx), %xmm15 

// CHECK: vpmovzxwd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x72,0x40]       
vpmovzxwd 64(%rdx), %xmm6 

// CHECK: vpmovzxwd (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x33,0x3a]       
vpmovzxwd (%rdx), %xmm15 

// CHECK: vpmovzxwd (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x32]       
vpmovzxwd (%rdx), %xmm6 

// CHECK: vpmovzxwd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x33,0xff]       
vpmovzxwd %xmm15, %xmm15 

// CHECK: vpmovzxwd %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0xf6]       
vpmovzxwd %xmm6, %xmm6 

// CHECK: vpmovzxwq 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x34,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096, %xmm15 

// CHECK: vpmovzxwq 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096, %xmm6 

// CHECK: vpmovzxwq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x34,0x7c,0x82,0xc0]       
vpmovzxwq -64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxwq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x34,0x7c,0x82,0x40]       
vpmovzxwq 64(%rdx,%rax,4), %xmm15 

// CHECK: vpmovzxwq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x74,0x82,0xc0]       
vpmovzxwq -64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxwq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x74,0x82,0x40]       
vpmovzxwq 64(%rdx,%rax,4), %xmm6 

// CHECK: vpmovzxwq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x34,0x7c,0x02,0x40]       
vpmovzxwq 64(%rdx,%rax), %xmm15 

// CHECK: vpmovzxwq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x74,0x02,0x40]       
vpmovzxwq 64(%rdx,%rax), %xmm6 

// CHECK: vpmovzxwq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x34,0x7a,0x40]       
vpmovzxwq 64(%rdx), %xmm15 

// CHECK: vpmovzxwq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x72,0x40]       
vpmovzxwq 64(%rdx), %xmm6 

// CHECK: vpmovzxwq (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x34,0x3a]       
vpmovzxwq (%rdx), %xmm15 

// CHECK: vpmovzxwq (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x32]       
vpmovzxwq (%rdx), %xmm6 

// CHECK: vpmovzxwq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x34,0xff]       
vpmovzxwq %xmm15, %xmm15 

// CHECK: vpmovzxwq %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0xf6]       
vpmovzxwq %xmm6, %xmm6 

// CHECK: vpmuldq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x28,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096, %xmm15, %xmm15 

// CHECK: vpmuldq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x28,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096, %xmm6, %xmm6 

// CHECK: vpmuldq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x28,0x7c,0x82,0xc0]      
vpmuldq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmuldq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x28,0x7c,0x82,0x40]      
vpmuldq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmuldq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x28,0x74,0x82,0xc0]      
vpmuldq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmuldq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x28,0x74,0x82,0x40]      
vpmuldq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmuldq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x28,0x7c,0x02,0x40]      
vpmuldq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmuldq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x28,0x74,0x02,0x40]      
vpmuldq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmuldq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x28,0x7a,0x40]      
vpmuldq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmuldq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x28,0x72,0x40]      
vpmuldq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmuldq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x28,0x3a]      
vpmuldq (%rdx), %xmm15, %xmm15 

// CHECK: vpmuldq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x28,0x32]      
vpmuldq (%rdx), %xmm6, %xmm6 

// CHECK: vpmuldq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x28,0xff]      
vpmuldq %xmm15, %xmm15, %xmm15 

// CHECK: vpmuldq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x28,0xf6]      
vpmuldq %xmm6, %xmm6, %xmm6 

// CHECK: vpmulhrsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096, %xmm15, %xmm15 

// CHECK: vpmulhrsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096, %xmm6, %xmm6 

// CHECK: vpmulhrsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0b,0x7c,0x82,0xc0]      
vpmulhrsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulhrsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0b,0x7c,0x82,0x40]      
vpmulhrsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulhrsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0b,0x74,0x82,0xc0]      
vpmulhrsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulhrsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0b,0x74,0x82,0x40]      
vpmulhrsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulhrsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0b,0x7c,0x02,0x40]      
vpmulhrsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmulhrsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0b,0x74,0x02,0x40]      
vpmulhrsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmulhrsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0b,0x7a,0x40]      
vpmulhrsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmulhrsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0b,0x72,0x40]      
vpmulhrsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmulhrsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0b,0x3a]      
vpmulhrsw (%rdx), %xmm15, %xmm15 

// CHECK: vpmulhrsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0b,0x32]      
vpmulhrsw (%rdx), %xmm6, %xmm6 

// CHECK: vpmulhrsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x0b,0xff]      
vpmulhrsw %xmm15, %xmm15, %xmm15 

// CHECK: vpmulhrsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0b,0xf6]      
vpmulhrsw %xmm6, %xmm6, %xmm6 

// CHECK: vpmulhuw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe4,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096, %xmm15, %xmm15 

// CHECK: vpmulhuw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe4,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096, %xmm6, %xmm6 

// CHECK: vpmulhuw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe4,0x7c,0x82,0xc0]      
vpmulhuw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulhuw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe4,0x7c,0x82,0x40]      
vpmulhuw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulhuw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe4,0x74,0x82,0xc0]      
vpmulhuw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulhuw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe4,0x74,0x82,0x40]      
vpmulhuw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulhuw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe4,0x7c,0x02,0x40]      
vpmulhuw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmulhuw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe4,0x74,0x02,0x40]      
vpmulhuw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmulhuw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe4,0x7a,0x40]      
vpmulhuw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmulhuw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe4,0x72,0x40]      
vpmulhuw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmulhuw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe4,0x3a]      
vpmulhuw (%rdx), %xmm15, %xmm15 

// CHECK: vpmulhuw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe4,0x32]      
vpmulhuw (%rdx), %xmm6, %xmm6 

// CHECK: vpmulhuw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe4,0xff]      
vpmulhuw %xmm15, %xmm15, %xmm15 

// CHECK: vpmulhuw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe4,0xf6]      
vpmulhuw %xmm6, %xmm6, %xmm6 

// CHECK: vpmulhw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe5,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096, %xmm15, %xmm15 

// CHECK: vpmulhw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe5,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096, %xmm6, %xmm6 

// CHECK: vpmulhw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe5,0x7c,0x82,0xc0]      
vpmulhw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulhw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe5,0x7c,0x82,0x40]      
vpmulhw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulhw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe5,0x74,0x82,0xc0]      
vpmulhw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulhw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe5,0x74,0x82,0x40]      
vpmulhw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulhw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe5,0x7c,0x02,0x40]      
vpmulhw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmulhw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe5,0x74,0x02,0x40]      
vpmulhw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmulhw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe5,0x7a,0x40]      
vpmulhw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmulhw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe5,0x72,0x40]      
vpmulhw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmulhw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe5,0x3a]      
vpmulhw (%rdx), %xmm15, %xmm15 

// CHECK: vpmulhw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe5,0x32]      
vpmulhw (%rdx), %xmm6, %xmm6 

// CHECK: vpmulhw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe5,0xff]      
vpmulhw %xmm15, %xmm15, %xmm15 

// CHECK: vpmulhw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe5,0xf6]      
vpmulhw %xmm6, %xmm6, %xmm6 

// CHECK: vpmulld 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x40,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096, %xmm15, %xmm15 

// CHECK: vpmulld 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x40,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096, %xmm6, %xmm6 

// CHECK: vpmulld -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x40,0x7c,0x82,0xc0]      
vpmulld -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulld 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x40,0x7c,0x82,0x40]      
vpmulld 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmulld -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x40,0x74,0x82,0xc0]      
vpmulld -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulld 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x40,0x74,0x82,0x40]      
vpmulld 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmulld 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x40,0x7c,0x02,0x40]      
vpmulld 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmulld 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x40,0x74,0x02,0x40]      
vpmulld 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmulld 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x40,0x7a,0x40]      
vpmulld 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmulld 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x40,0x72,0x40]      
vpmulld 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmulld (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x40,0x3a]      
vpmulld (%rdx), %xmm15, %xmm15 

// CHECK: vpmulld (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x40,0x32]      
vpmulld (%rdx), %xmm6, %xmm6 

// CHECK: vpmulld %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x40,0xff]      
vpmulld %xmm15, %xmm15, %xmm15 

// CHECK: vpmulld %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x40,0xf6]      
vpmulld %xmm6, %xmm6, %xmm6 

// CHECK: vpmullw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd5,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096, %xmm15, %xmm15 

// CHECK: vpmullw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd5,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096, %xmm6, %xmm6 

// CHECK: vpmullw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd5,0x7c,0x82,0xc0]      
vpmullw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmullw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd5,0x7c,0x82,0x40]      
vpmullw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmullw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd5,0x74,0x82,0xc0]      
vpmullw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmullw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd5,0x74,0x82,0x40]      
vpmullw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmullw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd5,0x7c,0x02,0x40]      
vpmullw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmullw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd5,0x74,0x02,0x40]      
vpmullw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmullw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd5,0x7a,0x40]      
vpmullw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmullw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd5,0x72,0x40]      
vpmullw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmullw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd5,0x3a]      
vpmullw (%rdx), %xmm15, %xmm15 

// CHECK: vpmullw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd5,0x32]      
vpmullw (%rdx), %xmm6, %xmm6 

// CHECK: vpmullw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd5,0xff]      
vpmullw %xmm15, %xmm15, %xmm15 

// CHECK: vpmullw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd5,0xf6]      
vpmullw %xmm6, %xmm6, %xmm6 

// CHECK: vpmuludq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf4,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096, %xmm15, %xmm15 

// CHECK: vpmuludq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf4,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096, %xmm6, %xmm6 

// CHECK: vpmuludq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf4,0x7c,0x82,0xc0]      
vpmuludq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmuludq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf4,0x7c,0x82,0x40]      
vpmuludq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmuludq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf4,0x74,0x82,0xc0]      
vpmuludq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmuludq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf4,0x74,0x82,0x40]      
vpmuludq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmuludq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf4,0x7c,0x02,0x40]      
vpmuludq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmuludq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf4,0x74,0x02,0x40]      
vpmuludq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmuludq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf4,0x7a,0x40]      
vpmuludq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmuludq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf4,0x72,0x40]      
vpmuludq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmuludq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf4,0x3a]      
vpmuludq (%rdx), %xmm15, %xmm15 

// CHECK: vpmuludq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf4,0x32]      
vpmuludq (%rdx), %xmm6, %xmm6 

// CHECK: vpmuludq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf4,0xff]      
vpmuludq %xmm15, %xmm15, %xmm15 

// CHECK: vpmuludq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf4,0xf6]      
vpmuludq %xmm6, %xmm6, %xmm6 

// CHECK: vpor 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xeb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096, %xmm15, %xmm15 

// CHECK: vpor 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xeb,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096, %xmm6, %xmm6 

// CHECK: vpor -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xeb,0x7c,0x82,0xc0]      
vpor -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpor 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xeb,0x7c,0x82,0x40]      
vpor 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpor -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xeb,0x74,0x82,0xc0]      
vpor -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpor 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xeb,0x74,0x82,0x40]      
vpor 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpor 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xeb,0x7c,0x02,0x40]      
vpor 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpor 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xeb,0x74,0x02,0x40]      
vpor 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpor 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xeb,0x7a,0x40]      
vpor 64(%rdx), %xmm15, %xmm15 

// CHECK: vpor 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xeb,0x72,0x40]      
vpor 64(%rdx), %xmm6, %xmm6 

// CHECK: vpor (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xeb,0x3a]      
vpor (%rdx), %xmm15, %xmm15 

// CHECK: vpor (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xeb,0x32]      
vpor (%rdx), %xmm6, %xmm6 

// CHECK: vpor %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xeb,0xff]      
vpor %xmm15, %xmm15, %xmm15 

// CHECK: vpor %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xeb,0xf6]      
vpor %xmm6, %xmm6, %xmm6 

// CHECK: vpsadbw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096, %xmm15, %xmm15 

// CHECK: vpsadbw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096, %xmm6, %xmm6 

// CHECK: vpsadbw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf6,0x7c,0x82,0xc0]      
vpsadbw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsadbw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf6,0x7c,0x82,0x40]      
vpsadbw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsadbw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf6,0x74,0x82,0xc0]      
vpsadbw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsadbw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf6,0x74,0x82,0x40]      
vpsadbw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsadbw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf6,0x7c,0x02,0x40]      
vpsadbw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsadbw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf6,0x74,0x02,0x40]      
vpsadbw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsadbw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf6,0x7a,0x40]      
vpsadbw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsadbw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf6,0x72,0x40]      
vpsadbw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsadbw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf6,0x3a]      
vpsadbw (%rdx), %xmm15, %xmm15 

// CHECK: vpsadbw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf6,0x32]      
vpsadbw (%rdx), %xmm6, %xmm6 

// CHECK: vpsadbw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf6,0xff]      
vpsadbw %xmm15, %xmm15, %xmm15 

// CHECK: vpsadbw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf6,0xf6]      
vpsadbw %xmm6, %xmm6, %xmm6 

// CHECK: vpshufb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x00,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096, %xmm15, %xmm15 

// CHECK: vpshufb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x00,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096, %xmm6, %xmm6 

// CHECK: vpshufb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x00,0x7c,0x82,0xc0]      
vpshufb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpshufb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x00,0x7c,0x82,0x40]      
vpshufb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpshufb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x00,0x74,0x82,0xc0]      
vpshufb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpshufb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x00,0x74,0x82,0x40]      
vpshufb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpshufb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x00,0x7c,0x02,0x40]      
vpshufb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpshufb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x00,0x74,0x02,0x40]      
vpshufb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpshufb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x00,0x7a,0x40]      
vpshufb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpshufb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x00,0x72,0x40]      
vpshufb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpshufb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x00,0x3a]      
vpshufb (%rdx), %xmm15, %xmm15 

// CHECK: vpshufb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x00,0x32]      
vpshufb (%rdx), %xmm6, %xmm6 

// CHECK: vpshufb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x00,0xff]      
vpshufb %xmm15, %xmm15, %xmm15 

// CHECK: vpshufb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x00,0xf6]      
vpshufb %xmm6, %xmm6, %xmm6 

// CHECK: vpshufd $0, 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x70,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096, %xmm15 

// CHECK: vpshufd $0, 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x70,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096, %xmm6 

// CHECK: vpshufd $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x70,0x7c,0x82,0xc0,0x00]      
vpshufd $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpshufd $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x70,0x7c,0x82,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpshufd $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x70,0x74,0x82,0xc0,0x00]      
vpshufd $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpshufd $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x70,0x74,0x82,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpshufd $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x70,0x7c,0x02,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpshufd $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x70,0x74,0x02,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpshufd $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x70,0x7a,0x40,0x00]      
vpshufd $0, 64(%rdx), %xmm15 

// CHECK: vpshufd $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x70,0x72,0x40,0x00]      
vpshufd $0, 64(%rdx), %xmm6 

// CHECK: vpshufd $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x70,0x3a,0x00]      
vpshufd $0, (%rdx), %xmm15 

// CHECK: vpshufd $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x70,0x32,0x00]      
vpshufd $0, (%rdx), %xmm6 

// CHECK: vpshufd $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x70,0xff,0x00]      
vpshufd $0, %xmm15, %xmm15 

// CHECK: vpshufd $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x70,0xf6,0x00]      
vpshufd $0, %xmm6, %xmm6 

// CHECK: vpshufhw $0, 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x70,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096, %xmm15 

// CHECK: vpshufhw $0, 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x70,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096, %xmm6 

// CHECK: vpshufhw $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x70,0x7c,0x82,0xc0,0x00]      
vpshufhw $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpshufhw $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x70,0x7c,0x82,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpshufhw $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x70,0x74,0x82,0xc0,0x00]      
vpshufhw $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpshufhw $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x70,0x74,0x82,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpshufhw $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x70,0x7c,0x02,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpshufhw $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x70,0x74,0x02,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpshufhw $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x70,0x7a,0x40,0x00]      
vpshufhw $0, 64(%rdx), %xmm15 

// CHECK: vpshufhw $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x70,0x72,0x40,0x00]      
vpshufhw $0, 64(%rdx), %xmm6 

// CHECK: vpshufhw $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7a,0x70,0x3a,0x00]      
vpshufhw $0, (%rdx), %xmm15 

// CHECK: vpshufhw $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x70,0x32,0x00]      
vpshufhw $0, (%rdx), %xmm6 

// CHECK: vpshufhw $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7a,0x70,0xff,0x00]      
vpshufhw $0, %xmm15, %xmm15 

// CHECK: vpshufhw $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfa,0x70,0xf6,0x00]      
vpshufhw $0, %xmm6, %xmm6 

// CHECK: vpshuflw $0, 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x70,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096, %xmm15 

// CHECK: vpshuflw $0, 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x70,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096, %xmm6 

// CHECK: vpshuflw $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x70,0x7c,0x82,0xc0,0x00]      
vpshuflw $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vpshuflw $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x70,0x7c,0x82,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vpshuflw $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x70,0x74,0x82,0xc0,0x00]      
vpshuflw $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vpshuflw $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x70,0x74,0x82,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vpshuflw $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x70,0x7c,0x02,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax), %xmm15 

// CHECK: vpshuflw $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x70,0x74,0x02,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax), %xmm6 

// CHECK: vpshuflw $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x70,0x7a,0x40,0x00]      
vpshuflw $0, 64(%rdx), %xmm15 

// CHECK: vpshuflw $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x70,0x72,0x40,0x00]      
vpshuflw $0, 64(%rdx), %xmm6 

// CHECK: vpshuflw $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x7b,0x70,0x3a,0x00]      
vpshuflw $0, (%rdx), %xmm15 

// CHECK: vpshuflw $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x70,0x32,0x00]      
vpshuflw $0, (%rdx), %xmm6 

// CHECK: vpshuflw $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x7b,0x70,0xff,0x00]      
vpshuflw $0, %xmm15, %xmm15 

// CHECK: vpshuflw $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xfb,0x70,0xf6,0x00]      
vpshuflw $0, %xmm6, %xmm6 

// CHECK: vpsignb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x08,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096, %xmm15, %xmm15 

// CHECK: vpsignb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x08,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096, %xmm6, %xmm6 

// CHECK: vpsignb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x08,0x7c,0x82,0xc0]      
vpsignb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsignb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x08,0x7c,0x82,0x40]      
vpsignb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsignb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x08,0x74,0x82,0xc0]      
vpsignb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsignb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x08,0x74,0x82,0x40]      
vpsignb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsignb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x08,0x7c,0x02,0x40]      
vpsignb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsignb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x08,0x74,0x02,0x40]      
vpsignb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsignb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x08,0x7a,0x40]      
vpsignb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsignb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x08,0x72,0x40]      
vpsignb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsignb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x08,0x3a]      
vpsignb (%rdx), %xmm15, %xmm15 

// CHECK: vpsignb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x08,0x32]      
vpsignb (%rdx), %xmm6, %xmm6 

// CHECK: vpsignb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x08,0xff]      
vpsignb %xmm15, %xmm15, %xmm15 

// CHECK: vpsignb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x08,0xf6]      
vpsignb %xmm6, %xmm6, %xmm6 

// CHECK: vpsignd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096, %xmm15, %xmm15 

// CHECK: vpsignd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096, %xmm6, %xmm6 

// CHECK: vpsignd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0a,0x7c,0x82,0xc0]      
vpsignd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsignd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0a,0x7c,0x82,0x40]      
vpsignd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsignd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0a,0x74,0x82,0xc0]      
vpsignd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsignd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0a,0x74,0x82,0x40]      
vpsignd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsignd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0a,0x7c,0x02,0x40]      
vpsignd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsignd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0a,0x74,0x02,0x40]      
vpsignd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsignd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0a,0x7a,0x40]      
vpsignd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsignd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0a,0x72,0x40]      
vpsignd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsignd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x0a,0x3a]      
vpsignd (%rdx), %xmm15, %xmm15 

// CHECK: vpsignd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0a,0x32]      
vpsignd (%rdx), %xmm6, %xmm6 

// CHECK: vpsignd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x0a,0xff]      
vpsignd %xmm15, %xmm15, %xmm15 

// CHECK: vpsignd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x0a,0xf6]      
vpsignd %xmm6, %xmm6, %xmm6 

// CHECK: vpsignw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x09,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096, %xmm15, %xmm15 

// CHECK: vpsignw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x09,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096, %xmm6, %xmm6 

// CHECK: vpsignw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x09,0x7c,0x82,0xc0]      
vpsignw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsignw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x09,0x7c,0x82,0x40]      
vpsignw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsignw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x09,0x74,0x82,0xc0]      
vpsignw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsignw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x09,0x74,0x82,0x40]      
vpsignw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsignw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x09,0x7c,0x02,0x40]      
vpsignw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsignw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x09,0x74,0x02,0x40]      
vpsignw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsignw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x09,0x7a,0x40]      
vpsignw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsignw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x09,0x72,0x40]      
vpsignw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsignw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x09,0x3a]      
vpsignw (%rdx), %xmm15, %xmm15 

// CHECK: vpsignw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x09,0x32]      
vpsignw (%rdx), %xmm6, %xmm6 

// CHECK: vpsignw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x09,0xff]      
vpsignw %xmm15, %xmm15, %xmm15 

// CHECK: vpsignw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x09,0xf6]      
vpsignw %xmm6, %xmm6, %xmm6 

// CHECK: vpslld $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x72,0xf7,0x00]      
vpslld $0, %xmm15, %xmm15 

// CHECK: vpslld $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x72,0xf6,0x00]      
vpslld $0, %xmm6, %xmm6 

// CHECK: vpslld 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096, %xmm15, %xmm15 

// CHECK: vpslld 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf2,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096, %xmm6, %xmm6 

// CHECK: vpslld -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf2,0x7c,0x82,0xc0]      
vpslld -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpslld 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf2,0x7c,0x82,0x40]      
vpslld 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpslld -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf2,0x74,0x82,0xc0]      
vpslld -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpslld 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf2,0x74,0x82,0x40]      
vpslld 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpslld 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf2,0x7c,0x02,0x40]      
vpslld 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpslld 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf2,0x74,0x02,0x40]      
vpslld 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpslld 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf2,0x7a,0x40]      
vpslld 64(%rdx), %xmm15, %xmm15 

// CHECK: vpslld 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf2,0x72,0x40]      
vpslld 64(%rdx), %xmm6, %xmm6 

// CHECK: vpslldq $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x73,0xff,0x00]      
vpslldq $0, %xmm15, %xmm15 

// CHECK: vpslldq $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x73,0xfe,0x00]      
vpslldq $0, %xmm6, %xmm6 

// CHECK: vpslld (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf2,0x3a]      
vpslld (%rdx), %xmm15, %xmm15 

// CHECK: vpslld (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf2,0x32]      
vpslld (%rdx), %xmm6, %xmm6 

// CHECK: vpslld %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf2,0xff]      
vpslld %xmm15, %xmm15, %xmm15 

// CHECK: vpslld %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf2,0xf6]      
vpslld %xmm6, %xmm6, %xmm6 

// CHECK: vpsllq $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x73,0xf7,0x00]      
vpsllq $0, %xmm15, %xmm15 

// CHECK: vpsllq $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x73,0xf6,0x00]      
vpsllq $0, %xmm6, %xmm6 

// CHECK: vpsllq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096, %xmm15, %xmm15 

// CHECK: vpsllq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf3,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096, %xmm6, %xmm6 

// CHECK: vpsllq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf3,0x7c,0x82,0xc0]      
vpsllq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf3,0x7c,0x82,0x40]      
vpsllq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf3,0x74,0x82,0xc0]      
vpsllq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf3,0x74,0x82,0x40]      
vpsllq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf3,0x7c,0x02,0x40]      
vpsllq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsllq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf3,0x74,0x02,0x40]      
vpsllq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsllq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf3,0x7a,0x40]      
vpsllq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsllq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf3,0x72,0x40]      
vpsllq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsllq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf3,0x3a]      
vpsllq (%rdx), %xmm15, %xmm15 

// CHECK: vpsllq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf3,0x32]      
vpsllq (%rdx), %xmm6, %xmm6 

// CHECK: vpsllq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf3,0xff]      
vpsllq %xmm15, %xmm15, %xmm15 

// CHECK: vpsllq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf3,0xf6]      
vpsllq %xmm6, %xmm6, %xmm6 

// CHECK: vpsllw $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x71,0xf7,0x00]      
vpsllw $0, %xmm15, %xmm15 

// CHECK: vpsllw $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x71,0xf6,0x00]      
vpsllw $0, %xmm6, %xmm6 

// CHECK: vpsllw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096, %xmm15, %xmm15 

// CHECK: vpsllw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf1,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096, %xmm6, %xmm6 

// CHECK: vpsllw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf1,0x7c,0x82,0xc0]      
vpsllw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf1,0x7c,0x82,0x40]      
vpsllw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf1,0x74,0x82,0xc0]      
vpsllw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf1,0x74,0x82,0x40]      
vpsllw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf1,0x7c,0x02,0x40]      
vpsllw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsllw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf1,0x74,0x02,0x40]      
vpsllw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsllw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf1,0x7a,0x40]      
vpsllw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsllw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf1,0x72,0x40]      
vpsllw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsllw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf1,0x3a]      
vpsllw (%rdx), %xmm15, %xmm15 

// CHECK: vpsllw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf1,0x32]      
vpsllw (%rdx), %xmm6, %xmm6 

// CHECK: vpsllw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf1,0xff]      
vpsllw %xmm15, %xmm15, %xmm15 

// CHECK: vpsllw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf1,0xf6]      
vpsllw %xmm6, %xmm6, %xmm6 

// CHECK: vpsrad $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x72,0xe7,0x00]      
vpsrad $0, %xmm15, %xmm15 

// CHECK: vpsrad $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x72,0xe6,0x00]      
vpsrad $0, %xmm6, %xmm6 

// CHECK: vpsrad 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096, %xmm15, %xmm15 

// CHECK: vpsrad 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe2,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096, %xmm6, %xmm6 

// CHECK: vpsrad -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe2,0x7c,0x82,0xc0]      
vpsrad -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrad 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe2,0x7c,0x82,0x40]      
vpsrad 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrad -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe2,0x74,0x82,0xc0]      
vpsrad -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrad 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe2,0x74,0x82,0x40]      
vpsrad 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrad 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe2,0x7c,0x02,0x40]      
vpsrad 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsrad 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe2,0x74,0x02,0x40]      
vpsrad 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsrad 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe2,0x7a,0x40]      
vpsrad 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsrad 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe2,0x72,0x40]      
vpsrad 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsrad (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe2,0x3a]      
vpsrad (%rdx), %xmm15, %xmm15 

// CHECK: vpsrad (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe2,0x32]      
vpsrad (%rdx), %xmm6, %xmm6 

// CHECK: vpsrad %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe2,0xff]      
vpsrad %xmm15, %xmm15, %xmm15 

// CHECK: vpsrad %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe2,0xf6]      
vpsrad %xmm6, %xmm6, %xmm6 

// CHECK: vpsraw $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x71,0xe7,0x00]      
vpsraw $0, %xmm15, %xmm15 

// CHECK: vpsraw $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x71,0xe6,0x00]      
vpsraw $0, %xmm6, %xmm6 

// CHECK: vpsraw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096, %xmm15, %xmm15 

// CHECK: vpsraw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe1,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096, %xmm6, %xmm6 

// CHECK: vpsraw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe1,0x7c,0x82,0xc0]      
vpsraw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsraw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe1,0x7c,0x82,0x40]      
vpsraw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsraw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe1,0x74,0x82,0xc0]      
vpsraw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsraw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe1,0x74,0x82,0x40]      
vpsraw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsraw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe1,0x7c,0x02,0x40]      
vpsraw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsraw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe1,0x74,0x02,0x40]      
vpsraw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsraw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe1,0x7a,0x40]      
vpsraw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsraw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe1,0x72,0x40]      
vpsraw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsraw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe1,0x3a]      
vpsraw (%rdx), %xmm15, %xmm15 

// CHECK: vpsraw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe1,0x32]      
vpsraw (%rdx), %xmm6, %xmm6 

// CHECK: vpsraw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe1,0xff]      
vpsraw %xmm15, %xmm15, %xmm15 

// CHECK: vpsraw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe1,0xf6]      
vpsraw %xmm6, %xmm6, %xmm6 

// CHECK: vpsrld $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x72,0xd7,0x00]      
vpsrld $0, %xmm15, %xmm15 

// CHECK: vpsrld $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x72,0xd6,0x00]      
vpsrld $0, %xmm6, %xmm6 

// CHECK: vpsrld 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096, %xmm15, %xmm15 

// CHECK: vpsrld 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd2,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096, %xmm6, %xmm6 

// CHECK: vpsrld -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd2,0x7c,0x82,0xc0]      
vpsrld -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrld 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd2,0x7c,0x82,0x40]      
vpsrld 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrld -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd2,0x74,0x82,0xc0]      
vpsrld -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrld 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd2,0x74,0x82,0x40]      
vpsrld 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrld 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd2,0x7c,0x02,0x40]      
vpsrld 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsrld 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd2,0x74,0x02,0x40]      
vpsrld 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsrld 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd2,0x7a,0x40]      
vpsrld 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsrld 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd2,0x72,0x40]      
vpsrld 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsrldq $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x73,0xdf,0x00]      
vpsrldq $0, %xmm15, %xmm15 

// CHECK: vpsrldq $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x73,0xde,0x00]      
vpsrldq $0, %xmm6, %xmm6 

// CHECK: vpsrld (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd2,0x3a]      
vpsrld (%rdx), %xmm15, %xmm15 

// CHECK: vpsrld (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd2,0x32]      
vpsrld (%rdx), %xmm6, %xmm6 

// CHECK: vpsrld %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd2,0xff]      
vpsrld %xmm15, %xmm15, %xmm15 

// CHECK: vpsrld %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd2,0xf6]      
vpsrld %xmm6, %xmm6, %xmm6 

// CHECK: vpsrlq $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x73,0xd7,0x00]      
vpsrlq $0, %xmm15, %xmm15 

// CHECK: vpsrlq $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x73,0xd6,0x00]      
vpsrlq $0, %xmm6, %xmm6 

// CHECK: vpsrlq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096, %xmm15, %xmm15 

// CHECK: vpsrlq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd3,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096, %xmm6, %xmm6 

// CHECK: vpsrlq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd3,0x7c,0x82,0xc0]      
vpsrlq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd3,0x7c,0x82,0x40]      
vpsrlq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd3,0x74,0x82,0xc0]      
vpsrlq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd3,0x74,0x82,0x40]      
vpsrlq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd3,0x7c,0x02,0x40]      
vpsrlq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsrlq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd3,0x74,0x02,0x40]      
vpsrlq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsrlq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd3,0x7a,0x40]      
vpsrlq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsrlq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd3,0x72,0x40]      
vpsrlq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsrlq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd3,0x3a]      
vpsrlq (%rdx), %xmm15, %xmm15 

// CHECK: vpsrlq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd3,0x32]      
vpsrlq (%rdx), %xmm6, %xmm6 

// CHECK: vpsrlq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd3,0xff]      
vpsrlq %xmm15, %xmm15, %xmm15 

// CHECK: vpsrlq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd3,0xf6]      
vpsrlq %xmm6, %xmm6, %xmm6 

// CHECK: vpsrlw $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0xc1,0x01,0x71,0xd7,0x00]      
vpsrlw $0, %xmm15, %xmm15 

// CHECK: vpsrlw $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x71,0xd6,0x00]      
vpsrlw $0, %xmm6, %xmm6 

// CHECK: vpsrlw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096, %xmm15, %xmm15 

// CHECK: vpsrlw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd1,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096, %xmm6, %xmm6 

// CHECK: vpsrlw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd1,0x7c,0x82,0xc0]      
vpsrlw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd1,0x7c,0x82,0x40]      
vpsrlw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd1,0x74,0x82,0xc0]      
vpsrlw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd1,0x74,0x82,0x40]      
vpsrlw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd1,0x7c,0x02,0x40]      
vpsrlw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsrlw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd1,0x74,0x02,0x40]      
vpsrlw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsrlw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd1,0x7a,0x40]      
vpsrlw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsrlw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd1,0x72,0x40]      
vpsrlw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsrlw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd1,0x3a]      
vpsrlw (%rdx), %xmm15, %xmm15 

// CHECK: vpsrlw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd1,0x32]      
vpsrlw (%rdx), %xmm6, %xmm6 

// CHECK: vpsrlw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd1,0xff]      
vpsrlw %xmm15, %xmm15, %xmm15 

// CHECK: vpsrlw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd1,0xf6]      
vpsrlw %xmm6, %xmm6, %xmm6 

// CHECK: vpsubb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096, %xmm15, %xmm15 

// CHECK: vpsubb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096, %xmm6, %xmm6 

// CHECK: vpsubb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf8,0x7c,0x82,0xc0]      
vpsubb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf8,0x7c,0x82,0x40]      
vpsubb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf8,0x74,0x82,0xc0]      
vpsubb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf8,0x74,0x82,0x40]      
vpsubb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf8,0x7c,0x02,0x40]      
vpsubb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf8,0x74,0x02,0x40]      
vpsubb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf8,0x7a,0x40]      
vpsubb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf8,0x72,0x40]      
vpsubb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf8,0x3a]      
vpsubb (%rdx), %xmm15, %xmm15 

// CHECK: vpsubb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf8,0x32]      
vpsubb (%rdx), %xmm6, %xmm6 

// CHECK: vpsubb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf8,0xff]      
vpsubb %xmm15, %xmm15, %xmm15 

// CHECK: vpsubb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf8,0xf6]      
vpsubb %xmm6, %xmm6, %xmm6 

// CHECK: vpsubd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfa,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096, %xmm15, %xmm15 

// CHECK: vpsubd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfa,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096, %xmm6, %xmm6 

// CHECK: vpsubd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfa,0x7c,0x82,0xc0]      
vpsubd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfa,0x7c,0x82,0x40]      
vpsubd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfa,0x74,0x82,0xc0]      
vpsubd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfa,0x74,0x82,0x40]      
vpsubd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfa,0x7c,0x02,0x40]      
vpsubd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfa,0x74,0x02,0x40]      
vpsubd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfa,0x7a,0x40]      
vpsubd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfa,0x72,0x40]      
vpsubd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfa,0x3a]      
vpsubd (%rdx), %xmm15, %xmm15 

// CHECK: vpsubd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfa,0x32]      
vpsubd (%rdx), %xmm6, %xmm6 

// CHECK: vpsubd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xfa,0xff]      
vpsubd %xmm15, %xmm15, %xmm15 

// CHECK: vpsubd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfa,0xf6]      
vpsubd %xmm6, %xmm6, %xmm6 

// CHECK: vpsubq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096, %xmm15, %xmm15 

// CHECK: vpsubq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfb,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096, %xmm6, %xmm6 

// CHECK: vpsubq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfb,0x7c,0x82,0xc0]      
vpsubq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfb,0x7c,0x82,0x40]      
vpsubq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfb,0x74,0x82,0xc0]      
vpsubq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfb,0x74,0x82,0x40]      
vpsubq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfb,0x7c,0x02,0x40]      
vpsubq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfb,0x74,0x02,0x40]      
vpsubq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfb,0x7a,0x40]      
vpsubq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfb,0x72,0x40]      
vpsubq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xfb,0x3a]      
vpsubq (%rdx), %xmm15, %xmm15 

// CHECK: vpsubq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfb,0x32]      
vpsubq (%rdx), %xmm6, %xmm6 

// CHECK: vpsubq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xfb,0xff]      
vpsubq %xmm15, %xmm15, %xmm15 

// CHECK: vpsubq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xfb,0xf6]      
vpsubq %xmm6, %xmm6, %xmm6 

// CHECK: vpsubsb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096, %xmm15, %xmm15 

// CHECK: vpsubsb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096, %xmm6, %xmm6 

// CHECK: vpsubsb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe8,0x7c,0x82,0xc0]      
vpsubsb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubsb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe8,0x7c,0x82,0x40]      
vpsubsb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubsb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe8,0x74,0x82,0xc0]      
vpsubsb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubsb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe8,0x74,0x82,0x40]      
vpsubsb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubsb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe8,0x7c,0x02,0x40]      
vpsubsb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubsb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe8,0x74,0x02,0x40]      
vpsubsb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubsb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe8,0x7a,0x40]      
vpsubsb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubsb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe8,0x72,0x40]      
vpsubsb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubsb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe8,0x3a]      
vpsubsb (%rdx), %xmm15, %xmm15 

// CHECK: vpsubsb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe8,0x32]      
vpsubsb (%rdx), %xmm6, %xmm6 

// CHECK: vpsubsb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe8,0xff]      
vpsubsb %xmm15, %xmm15, %xmm15 

// CHECK: vpsubsb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe8,0xf6]      
vpsubsb %xmm6, %xmm6, %xmm6 

// CHECK: vpsubsw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096, %xmm15, %xmm15 

// CHECK: vpsubsw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096, %xmm6, %xmm6 

// CHECK: vpsubsw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe9,0x7c,0x82,0xc0]      
vpsubsw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubsw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe9,0x7c,0x82,0x40]      
vpsubsw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubsw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe9,0x74,0x82,0xc0]      
vpsubsw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubsw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe9,0x74,0x82,0x40]      
vpsubsw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubsw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe9,0x7c,0x02,0x40]      
vpsubsw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubsw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe9,0x74,0x02,0x40]      
vpsubsw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubsw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe9,0x7a,0x40]      
vpsubsw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubsw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe9,0x72,0x40]      
vpsubsw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubsw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xe9,0x3a]      
vpsubsw (%rdx), %xmm15, %xmm15 

// CHECK: vpsubsw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe9,0x32]      
vpsubsw (%rdx), %xmm6, %xmm6 

// CHECK: vpsubsw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xe9,0xff]      
vpsubsw %xmm15, %xmm15, %xmm15 

// CHECK: vpsubsw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xe9,0xf6]      
vpsubsw %xmm6, %xmm6, %xmm6 

// CHECK: vpsubusb 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096, %xmm15, %xmm15 

// CHECK: vpsubusb 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096, %xmm6, %xmm6 

// CHECK: vpsubusb -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd8,0x7c,0x82,0xc0]      
vpsubusb -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubusb 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd8,0x7c,0x82,0x40]      
vpsubusb 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubusb -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd8,0x74,0x82,0xc0]      
vpsubusb -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubusb 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd8,0x74,0x82,0x40]      
vpsubusb 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubusb 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd8,0x7c,0x02,0x40]      
vpsubusb 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubusb 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd8,0x74,0x02,0x40]      
vpsubusb 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubusb 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd8,0x7a,0x40]      
vpsubusb 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubusb 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd8,0x72,0x40]      
vpsubusb 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubusb (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd8,0x3a]      
vpsubusb (%rdx), %xmm15, %xmm15 

// CHECK: vpsubusb (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd8,0x32]      
vpsubusb (%rdx), %xmm6, %xmm6 

// CHECK: vpsubusb %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd8,0xff]      
vpsubusb %xmm15, %xmm15, %xmm15 

// CHECK: vpsubusb %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd8,0xf6]      
vpsubusb %xmm6, %xmm6, %xmm6 

// CHECK: vpsubusw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096, %xmm15, %xmm15 

// CHECK: vpsubusw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096, %xmm6, %xmm6 

// CHECK: vpsubusw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd9,0x7c,0x82,0xc0]      
vpsubusw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubusw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd9,0x7c,0x82,0x40]      
vpsubusw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubusw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd9,0x74,0x82,0xc0]      
vpsubusw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubusw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd9,0x74,0x82,0x40]      
vpsubusw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubusw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd9,0x7c,0x02,0x40]      
vpsubusw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubusw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd9,0x74,0x02,0x40]      
vpsubusw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubusw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd9,0x7a,0x40]      
vpsubusw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubusw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd9,0x72,0x40]      
vpsubusw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubusw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xd9,0x3a]      
vpsubusw (%rdx), %xmm15, %xmm15 

// CHECK: vpsubusw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd9,0x32]      
vpsubusw (%rdx), %xmm6, %xmm6 

// CHECK: vpsubusw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xd9,0xff]      
vpsubusw %xmm15, %xmm15, %xmm15 

// CHECK: vpsubusw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xd9,0xf6]      
vpsubusw %xmm6, %xmm6, %xmm6 

// CHECK: vpsubw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096, %xmm15, %xmm15 

// CHECK: vpsubw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096, %xmm6, %xmm6 

// CHECK: vpsubw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf9,0x7c,0x82,0xc0]      
vpsubw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf9,0x7c,0x82,0x40]      
vpsubw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsubw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf9,0x74,0x82,0xc0]      
vpsubw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf9,0x74,0x82,0x40]      
vpsubw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsubw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf9,0x7c,0x02,0x40]      
vpsubw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsubw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf9,0x74,0x02,0x40]      
vpsubw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsubw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf9,0x7a,0x40]      
vpsubw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsubw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf9,0x72,0x40]      
vpsubw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsubw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xf9,0x3a]      
vpsubw (%rdx), %xmm15, %xmm15 

// CHECK: vpsubw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf9,0x32]      
vpsubw (%rdx), %xmm6, %xmm6 

// CHECK: vpsubw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xf9,0xff]      
vpsubw %xmm15, %xmm15, %xmm15 

// CHECK: vpsubw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xf9,0xf6]      
vpsubw %xmm6, %xmm6, %xmm6 

// CHECK: vptest 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x17,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096, %xmm15 

// CHECK: vptest 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096, %xmm6 

// CHECK: vptest 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096, %ymm7 

// CHECK: vptest 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x17,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vptest 485498096, %ymm9 

// CHECK: vptest -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x17,0x7c,0x82,0xc0]       
vptest -64(%rdx,%rax,4), %xmm15 

// CHECK: vptest 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x17,0x7c,0x82,0x40]       
vptest 64(%rdx,%rax,4), %xmm15 

// CHECK: vptest -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x74,0x82,0xc0]       
vptest -64(%rdx,%rax,4), %xmm6 

// CHECK: vptest 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x74,0x82,0x40]       
vptest 64(%rdx,%rax,4), %xmm6 

// CHECK: vptest -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x7c,0x82,0xc0]       
vptest -64(%rdx,%rax,4), %ymm7 

// CHECK: vptest 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x7c,0x82,0x40]       
vptest 64(%rdx,%rax,4), %ymm7 

// CHECK: vptest -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x17,0x4c,0x82,0xc0]       
vptest -64(%rdx,%rax,4), %ymm9 

// CHECK: vptest 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x17,0x4c,0x82,0x40]       
vptest 64(%rdx,%rax,4), %ymm9 

// CHECK: vptest 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x17,0x7c,0x02,0x40]       
vptest 64(%rdx,%rax), %xmm15 

// CHECK: vptest 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x74,0x02,0x40]       
vptest 64(%rdx,%rax), %xmm6 

// CHECK: vptest 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x7c,0x02,0x40]       
vptest 64(%rdx,%rax), %ymm7 

// CHECK: vptest 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x17,0x4c,0x02,0x40]       
vptest 64(%rdx,%rax), %ymm9 

// CHECK: vptest 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x17,0x7a,0x40]       
vptest 64(%rdx), %xmm15 

// CHECK: vptest 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x72,0x40]       
vptest 64(%rdx), %xmm6 

// CHECK: vptest 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x7a,0x40]       
vptest 64(%rdx), %ymm7 

// CHECK: vptest 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x17,0x4a,0x40]       
vptest 64(%rdx), %ymm9 

// CHECK: vptest (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x17,0x3a]       
vptest (%rdx), %xmm15 

// CHECK: vptest (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x32]       
vptest (%rdx), %xmm6 

// CHECK: vptest (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x3a]       
vptest (%rdx), %ymm7 

// CHECK: vptest (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x17,0x0a]       
vptest (%rdx), %ymm9 

// CHECK: vptest %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x17,0xff]       
vptest %xmm15, %xmm15 

// CHECK: vptest %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0xf6]       
vptest %xmm6, %xmm6 

// CHECK: vptest %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0xff]       
vptest %ymm7, %ymm7 

// CHECK: vptest %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x17,0xc9]       
vptest %ymm9, %ymm9 

// CHECK: vpunpckhbw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x68,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096, %xmm15, %xmm15 

// CHECK: vpunpckhbw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x68,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096, %xmm6, %xmm6 

// CHECK: vpunpckhbw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x68,0x7c,0x82,0xc0]      
vpunpckhbw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhbw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x68,0x7c,0x82,0x40]      
vpunpckhbw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhbw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x68,0x74,0x82,0xc0]      
vpunpckhbw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhbw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x68,0x74,0x82,0x40]      
vpunpckhbw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhbw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x68,0x7c,0x02,0x40]      
vpunpckhbw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpckhbw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x68,0x74,0x02,0x40]      
vpunpckhbw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpckhbw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x68,0x7a,0x40]      
vpunpckhbw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhbw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x68,0x72,0x40]      
vpunpckhbw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhbw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x68,0x3a]      
vpunpckhbw (%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhbw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x68,0x32]      
vpunpckhbw (%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhbw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x68,0xff]      
vpunpckhbw %xmm15, %xmm15, %xmm15 

// CHECK: vpunpckhbw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x68,0xf6]      
vpunpckhbw %xmm6, %xmm6, %xmm6 

// CHECK: vpunpckhdq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096, %xmm15, %xmm15 

// CHECK: vpunpckhdq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096, %xmm6, %xmm6 

// CHECK: vpunpckhdq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6a,0x7c,0x82,0xc0]      
vpunpckhdq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhdq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6a,0x7c,0x82,0x40]      
vpunpckhdq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhdq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6a,0x74,0x82,0xc0]      
vpunpckhdq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhdq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6a,0x74,0x82,0x40]      
vpunpckhdq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhdq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6a,0x7c,0x02,0x40]      
vpunpckhdq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpckhdq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6a,0x74,0x02,0x40]      
vpunpckhdq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpckhdq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6a,0x7a,0x40]      
vpunpckhdq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhdq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6a,0x72,0x40]      
vpunpckhdq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhdq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6a,0x3a]      
vpunpckhdq (%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhdq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6a,0x32]      
vpunpckhdq (%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhdq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x6a,0xff]      
vpunpckhdq %xmm15, %xmm15, %xmm15 

// CHECK: vpunpckhdq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6a,0xf6]      
vpunpckhdq %xmm6, %xmm6, %xmm6 

// CHECK: vpunpckhqdq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096, %xmm15, %xmm15 

// CHECK: vpunpckhqdq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096, %xmm6, %xmm6 

// CHECK: vpunpckhqdq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6d,0x7c,0x82,0xc0]      
vpunpckhqdq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhqdq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6d,0x7c,0x82,0x40]      
vpunpckhqdq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhqdq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6d,0x74,0x82,0xc0]      
vpunpckhqdq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhqdq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6d,0x74,0x82,0x40]      
vpunpckhqdq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhqdq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6d,0x7c,0x02,0x40]      
vpunpckhqdq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpckhqdq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6d,0x74,0x02,0x40]      
vpunpckhqdq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpckhqdq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6d,0x7a,0x40]      
vpunpckhqdq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhqdq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6d,0x72,0x40]      
vpunpckhqdq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhqdq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6d,0x3a]      
vpunpckhqdq (%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhqdq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6d,0x32]      
vpunpckhqdq (%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhqdq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x6d,0xff]      
vpunpckhqdq %xmm15, %xmm15, %xmm15 

// CHECK: vpunpckhqdq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6d,0xf6]      
vpunpckhqdq %xmm6, %xmm6, %xmm6 

// CHECK: vpunpckhwd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x69,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096, %xmm15, %xmm15 

// CHECK: vpunpckhwd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x69,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096, %xmm6, %xmm6 

// CHECK: vpunpckhwd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x69,0x7c,0x82,0xc0]      
vpunpckhwd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhwd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x69,0x7c,0x82,0x40]      
vpunpckhwd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckhwd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x69,0x74,0x82,0xc0]      
vpunpckhwd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhwd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x69,0x74,0x82,0x40]      
vpunpckhwd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckhwd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x69,0x7c,0x02,0x40]      
vpunpckhwd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpckhwd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x69,0x74,0x02,0x40]      
vpunpckhwd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpckhwd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x69,0x7a,0x40]      
vpunpckhwd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhwd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x69,0x72,0x40]      
vpunpckhwd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhwd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x69,0x3a]      
vpunpckhwd (%rdx), %xmm15, %xmm15 

// CHECK: vpunpckhwd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x69,0x32]      
vpunpckhwd (%rdx), %xmm6, %xmm6 

// CHECK: vpunpckhwd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x69,0xff]      
vpunpckhwd %xmm15, %xmm15, %xmm15 

// CHECK: vpunpckhwd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x69,0xf6]      
vpunpckhwd %xmm6, %xmm6, %xmm6 

// CHECK: vpunpcklbw 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x60,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096, %xmm15, %xmm15 

// CHECK: vpunpcklbw 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x60,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096, %xmm6, %xmm6 

// CHECK: vpunpcklbw -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x60,0x7c,0x82,0xc0]      
vpunpcklbw -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpcklbw 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x60,0x7c,0x82,0x40]      
vpunpcklbw 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpcklbw -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x60,0x74,0x82,0xc0]      
vpunpcklbw -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpcklbw 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x60,0x74,0x82,0x40]      
vpunpcklbw 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpcklbw 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x60,0x7c,0x02,0x40]      
vpunpcklbw 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpcklbw 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x60,0x74,0x02,0x40]      
vpunpcklbw 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpcklbw 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x60,0x7a,0x40]      
vpunpcklbw 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpcklbw 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x60,0x72,0x40]      
vpunpcklbw 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpcklbw (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x60,0x3a]      
vpunpcklbw (%rdx), %xmm15, %xmm15 

// CHECK: vpunpcklbw (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x60,0x32]      
vpunpcklbw (%rdx), %xmm6, %xmm6 

// CHECK: vpunpcklbw %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x60,0xff]      
vpunpcklbw %xmm15, %xmm15, %xmm15 

// CHECK: vpunpcklbw %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x60,0xf6]      
vpunpcklbw %xmm6, %xmm6, %xmm6 

// CHECK: vpunpckldq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x62,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096, %xmm15, %xmm15 

// CHECK: vpunpckldq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x62,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096, %xmm6, %xmm6 

// CHECK: vpunpckldq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x62,0x7c,0x82,0xc0]      
vpunpckldq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckldq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x62,0x7c,0x82,0x40]      
vpunpckldq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpckldq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x62,0x74,0x82,0xc0]      
vpunpckldq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckldq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x62,0x74,0x82,0x40]      
vpunpckldq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpckldq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x62,0x7c,0x02,0x40]      
vpunpckldq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpckldq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x62,0x74,0x02,0x40]      
vpunpckldq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpckldq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x62,0x7a,0x40]      
vpunpckldq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpckldq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x62,0x72,0x40]      
vpunpckldq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpckldq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x62,0x3a]      
vpunpckldq (%rdx), %xmm15, %xmm15 

// CHECK: vpunpckldq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x62,0x32]      
vpunpckldq (%rdx), %xmm6, %xmm6 

// CHECK: vpunpckldq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x62,0xff]      
vpunpckldq %xmm15, %xmm15, %xmm15 

// CHECK: vpunpckldq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x62,0xf6]      
vpunpckldq %xmm6, %xmm6, %xmm6 

// CHECK: vpunpcklqdq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096, %xmm15, %xmm15 

// CHECK: vpunpcklqdq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096, %xmm6, %xmm6 

// CHECK: vpunpcklqdq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6c,0x7c,0x82,0xc0]      
vpunpcklqdq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpcklqdq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6c,0x7c,0x82,0x40]      
vpunpcklqdq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpcklqdq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6c,0x74,0x82,0xc0]      
vpunpcklqdq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpcklqdq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6c,0x74,0x82,0x40]      
vpunpcklqdq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpcklqdq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6c,0x7c,0x02,0x40]      
vpunpcklqdq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpcklqdq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6c,0x74,0x02,0x40]      
vpunpcklqdq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpcklqdq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6c,0x7a,0x40]      
vpunpcklqdq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpcklqdq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6c,0x72,0x40]      
vpunpcklqdq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpcklqdq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x6c,0x3a]      
vpunpcklqdq (%rdx), %xmm15, %xmm15 

// CHECK: vpunpcklqdq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6c,0x32]      
vpunpcklqdq (%rdx), %xmm6, %xmm6 

// CHECK: vpunpcklqdq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x6c,0xff]      
vpunpcklqdq %xmm15, %xmm15, %xmm15 

// CHECK: vpunpcklqdq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x6c,0xf6]      
vpunpcklqdq %xmm6, %xmm6, %xmm6 

// CHECK: vpunpcklwd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x61,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096, %xmm15, %xmm15 

// CHECK: vpunpcklwd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x61,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096, %xmm6, %xmm6 

// CHECK: vpunpcklwd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x61,0x7c,0x82,0xc0]      
vpunpcklwd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpcklwd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x61,0x7c,0x82,0x40]      
vpunpcklwd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpunpcklwd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x61,0x74,0x82,0xc0]      
vpunpcklwd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpcklwd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x61,0x74,0x82,0x40]      
vpunpcklwd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpunpcklwd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x61,0x7c,0x02,0x40]      
vpunpcklwd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpunpcklwd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x61,0x74,0x02,0x40]      
vpunpcklwd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpunpcklwd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x61,0x7a,0x40]      
vpunpcklwd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpunpcklwd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x61,0x72,0x40]      
vpunpcklwd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpunpcklwd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x61,0x3a]      
vpunpcklwd (%rdx), %xmm15, %xmm15 

// CHECK: vpunpcklwd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x61,0x32]      
vpunpcklwd (%rdx), %xmm6, %xmm6 

// CHECK: vpunpcklwd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x61,0xff]      
vpunpcklwd %xmm15, %xmm15, %xmm15 

// CHECK: vpunpcklwd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x61,0xf6]      
vpunpcklwd %xmm6, %xmm6, %xmm6 

// CHECK: vpxor 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xef,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096, %xmm15, %xmm15 

// CHECK: vpxor 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xef,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096, %xmm6, %xmm6 

// CHECK: vpxor -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xef,0x7c,0x82,0xc0]      
vpxor -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpxor 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xef,0x7c,0x82,0x40]      
vpxor 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpxor -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xef,0x74,0x82,0xc0]      
vpxor -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpxor 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xef,0x74,0x82,0x40]      
vpxor 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpxor 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xef,0x7c,0x02,0x40]      
vpxor 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpxor 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xef,0x74,0x02,0x40]      
vpxor 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpxor 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xef,0x7a,0x40]      
vpxor 64(%rdx), %xmm15, %xmm15 

// CHECK: vpxor 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xef,0x72,0x40]      
vpxor 64(%rdx), %xmm6, %xmm6 

// CHECK: vpxor (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xef,0x3a]      
vpxor (%rdx), %xmm15, %xmm15 

// CHECK: vpxor (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xef,0x32]      
vpxor (%rdx), %xmm6, %xmm6 

// CHECK: vpxor %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xef,0xff]      
vpxor %xmm15, %xmm15, %xmm15 

// CHECK: vpxor %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xef,0xf6]      
vpxor %xmm6, %xmm6, %xmm6 

// CHECK: vrcpps 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x53,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096, %xmm15 

// CHECK: vrcpps 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x53,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096, %xmm6 

// CHECK: vrcpps 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x53,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096, %ymm7 

// CHECK: vrcpps 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x53,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vrcpps 485498096, %ymm9 

// CHECK: vrcpps -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x53,0x7c,0x82,0xc0]       
vrcpps -64(%rdx,%rax,4), %xmm15 

// CHECK: vrcpps 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x53,0x7c,0x82,0x40]       
vrcpps 64(%rdx,%rax,4), %xmm15 

// CHECK: vrcpps -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x53,0x74,0x82,0xc0]       
vrcpps -64(%rdx,%rax,4), %xmm6 

// CHECK: vrcpps 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x53,0x74,0x82,0x40]       
vrcpps 64(%rdx,%rax,4), %xmm6 

// CHECK: vrcpps -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x53,0x7c,0x82,0xc0]       
vrcpps -64(%rdx,%rax,4), %ymm7 

// CHECK: vrcpps 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x53,0x7c,0x82,0x40]       
vrcpps 64(%rdx,%rax,4), %ymm7 

// CHECK: vrcpps -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x53,0x4c,0x82,0xc0]       
vrcpps -64(%rdx,%rax,4), %ymm9 

// CHECK: vrcpps 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x53,0x4c,0x82,0x40]       
vrcpps 64(%rdx,%rax,4), %ymm9 

// CHECK: vrcpps 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x53,0x7c,0x02,0x40]       
vrcpps 64(%rdx,%rax), %xmm15 

// CHECK: vrcpps 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x53,0x74,0x02,0x40]       
vrcpps 64(%rdx,%rax), %xmm6 

// CHECK: vrcpps 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x53,0x7c,0x02,0x40]       
vrcpps 64(%rdx,%rax), %ymm7 

// CHECK: vrcpps 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x53,0x4c,0x02,0x40]       
vrcpps 64(%rdx,%rax), %ymm9 

// CHECK: vrcpps 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x53,0x7a,0x40]       
vrcpps 64(%rdx), %xmm15 

// CHECK: vrcpps 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x53,0x72,0x40]       
vrcpps 64(%rdx), %xmm6 

// CHECK: vrcpps 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x53,0x7a,0x40]       
vrcpps 64(%rdx), %ymm7 

// CHECK: vrcpps 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x53,0x4a,0x40]       
vrcpps 64(%rdx), %ymm9 

// CHECK: vrcpps (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x53,0x3a]       
vrcpps (%rdx), %xmm15 

// CHECK: vrcpps (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x53,0x32]       
vrcpps (%rdx), %xmm6 

// CHECK: vrcpps (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x53,0x3a]       
vrcpps (%rdx), %ymm7 

// CHECK: vrcpps (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x53,0x0a]       
vrcpps (%rdx), %ymm9 

// CHECK: vrcpps %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x53,0xff]       
vrcpps %xmm15, %xmm15 

// CHECK: vrcpps %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x53,0xf6]       
vrcpps %xmm6, %xmm6 

// CHECK: vrcpps %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x53,0xff]       
vrcpps %ymm7, %ymm7 

// CHECK: vrcpps %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7c,0x53,0xc9]       
vrcpps %ymm9, %ymm9 

// CHECK: vrcpss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x53,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vrcpss 485498096, %xmm15, %xmm15 

// CHECK: vrcpss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x53,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vrcpss 485498096, %xmm6, %xmm6 

// CHECK: vrcpss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x53,0x7c,0x82,0xc0]      
vrcpss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vrcpss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x53,0x7c,0x82,0x40]      
vrcpss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vrcpss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x53,0x74,0x82,0xc0]      
vrcpss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vrcpss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x53,0x74,0x82,0x40]      
vrcpss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vrcpss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x53,0x7c,0x02,0x40]      
vrcpss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vrcpss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x53,0x74,0x02,0x40]      
vrcpss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vrcpss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x53,0x7a,0x40]      
vrcpss 64(%rdx), %xmm15, %xmm15 

// CHECK: vrcpss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x53,0x72,0x40]      
vrcpss 64(%rdx), %xmm6, %xmm6 

// CHECK: vrcpss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x53,0x3a]      
vrcpss (%rdx), %xmm15, %xmm15 

// CHECK: vrcpss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x53,0x32]      
vrcpss (%rdx), %xmm6, %xmm6 

// CHECK: vrcpss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x53,0xff]      
vrcpss %xmm15, %xmm15, %xmm15 

// CHECK: vrcpss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x53,0xf6]      
vrcpss %xmm6, %xmm6, %xmm6 

// CHECK: vroundpd $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x09,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096, %xmm15 

// CHECK: vroundpd $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096, %xmm6 

// CHECK: vroundpd $0, 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096, %ymm7 

// CHECK: vroundpd $0, 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x09,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundpd $0, 485498096, %ymm9 

// CHECK: vroundpd $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x09,0x7c,0x82,0xc0,0x00]      
vroundpd $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vroundpd $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x09,0x7c,0x82,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vroundpd $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x74,0x82,0xc0,0x00]      
vroundpd $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vroundpd $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x74,0x82,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vroundpd $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x7c,0x82,0xc0,0x00]      
vroundpd $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vroundpd $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x7c,0x82,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vroundpd $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x09,0x4c,0x82,0xc0,0x00]      
vroundpd $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vroundpd $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x09,0x4c,0x82,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vroundpd $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x09,0x7c,0x02,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax), %xmm15 

// CHECK: vroundpd $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x74,0x02,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax), %xmm6 

// CHECK: vroundpd $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x7c,0x02,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax), %ymm7 

// CHECK: vroundpd $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x09,0x4c,0x02,0x40,0x00]      
vroundpd $0, 64(%rdx,%rax), %ymm9 

// CHECK: vroundpd $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x09,0x7a,0x40,0x00]      
vroundpd $0, 64(%rdx), %xmm15 

// CHECK: vroundpd $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x72,0x40,0x00]      
vroundpd $0, 64(%rdx), %xmm6 

// CHECK: vroundpd $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x7a,0x40,0x00]      
vroundpd $0, 64(%rdx), %ymm7 

// CHECK: vroundpd $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x09,0x4a,0x40,0x00]      
vroundpd $0, 64(%rdx), %ymm9 

// CHECK: vroundpd $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x09,0x3a,0x00]      
vroundpd $0, (%rdx), %xmm15 

// CHECK: vroundpd $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x32,0x00]      
vroundpd $0, (%rdx), %xmm6 

// CHECK: vroundpd $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x3a,0x00]      
vroundpd $0, (%rdx), %ymm7 

// CHECK: vroundpd $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x09,0x0a,0x00]      
vroundpd $0, (%rdx), %ymm9 

// CHECK: vroundpd $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x09,0xff,0x00]      
vroundpd $0, %xmm15, %xmm15 

// CHECK: vroundpd $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0xf6,0x00]      
vroundpd $0, %xmm6, %xmm6 

// CHECK: vroundpd $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0xff,0x00]      
vroundpd $0, %ymm7, %ymm7 

// CHECK: vroundpd $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x7d,0x09,0xc9,0x00]      
vroundpd $0, %ymm9, %ymm9 

// CHECK: vroundps $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x08,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096, %xmm15 

// CHECK: vroundps $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096, %xmm6 

// CHECK: vroundps $0, 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096, %ymm7 

// CHECK: vroundps $0, 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x08,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vroundps $0, 485498096, %ymm9 

// CHECK: vroundps $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x08,0x7c,0x82,0xc0,0x00]      
vroundps $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vroundps $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x08,0x7c,0x82,0x40,0x00]      
vroundps $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vroundps $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x74,0x82,0xc0,0x00]      
vroundps $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vroundps $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x74,0x82,0x40,0x00]      
vroundps $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vroundps $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x7c,0x82,0xc0,0x00]      
vroundps $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vroundps $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x7c,0x82,0x40,0x00]      
vroundps $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vroundps $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x08,0x4c,0x82,0xc0,0x00]      
vroundps $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vroundps $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x08,0x4c,0x82,0x40,0x00]      
vroundps $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vroundps $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x08,0x7c,0x02,0x40,0x00]      
vroundps $0, 64(%rdx,%rax), %xmm15 

// CHECK: vroundps $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x74,0x02,0x40,0x00]      
vroundps $0, 64(%rdx,%rax), %xmm6 

// CHECK: vroundps $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x7c,0x02,0x40,0x00]      
vroundps $0, 64(%rdx,%rax), %ymm7 

// CHECK: vroundps $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x08,0x4c,0x02,0x40,0x00]      
vroundps $0, 64(%rdx,%rax), %ymm9 

// CHECK: vroundps $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x08,0x7a,0x40,0x00]      
vroundps $0, 64(%rdx), %xmm15 

// CHECK: vroundps $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x72,0x40,0x00]      
vroundps $0, 64(%rdx), %xmm6 

// CHECK: vroundps $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x7a,0x40,0x00]      
vroundps $0, 64(%rdx), %ymm7 

// CHECK: vroundps $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x08,0x4a,0x40,0x00]      
vroundps $0, 64(%rdx), %ymm9 

// CHECK: vroundps $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0x08,0x3a,0x00]      
vroundps $0, (%rdx), %xmm15 

// CHECK: vroundps $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x32,0x00]      
vroundps $0, (%rdx), %xmm6 

// CHECK: vroundps $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x3a,0x00]      
vroundps $0, (%rdx), %ymm7 

// CHECK: vroundps $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0x7d,0x08,0x0a,0x00]      
vroundps $0, (%rdx), %ymm9 

// CHECK: vroundps $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0x08,0xff,0x00]      
vroundps $0, %xmm15, %xmm15 

// CHECK: vroundps $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0xf6,0x00]      
vroundps $0, %xmm6, %xmm6 

// CHECK: vroundps $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0xff,0x00]      
vroundps $0, %ymm7, %ymm7 

// CHECK: vroundps $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x7d,0x08,0xc9,0x00]      
vroundps $0, %ymm9, %ymm9 

// CHECK: vroundsd $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundsd $0, 485498096, %xmm15, %xmm15 

// CHECK: vroundsd $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0b,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundsd $0, 485498096, %xmm6, %xmm6 

// CHECK: vroundsd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0b,0x7c,0x82,0xc0,0x00]     
vroundsd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vroundsd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0b,0x7c,0x82,0x40,0x00]     
vroundsd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vroundsd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0b,0x74,0x82,0xc0,0x00]     
vroundsd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vroundsd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0b,0x74,0x82,0x40,0x00]     
vroundsd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vroundsd $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0b,0x7c,0x02,0x40,0x00]     
vroundsd $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vroundsd $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0b,0x74,0x02,0x40,0x00]     
vroundsd $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vroundsd $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0b,0x7a,0x40,0x00]     
vroundsd $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vroundsd $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0b,0x72,0x40,0x00]     
vroundsd $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vroundsd $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0b,0x3a,0x00]     
vroundsd $0, (%rdx), %xmm15, %xmm15 

// CHECK: vroundsd $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0b,0x32,0x00]     
vroundsd $0, (%rdx), %xmm6, %xmm6 

// CHECK: vroundsd $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x0b,0xff,0x00]     
vroundsd $0, %xmm15, %xmm15, %xmm15 

// CHECK: vroundsd $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0b,0xf6,0x00]     
vroundsd $0, %xmm6, %xmm6, %xmm6 

// CHECK: vroundss $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundss $0, 485498096, %xmm15, %xmm15 

// CHECK: vroundss $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0a,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vroundss $0, 485498096, %xmm6, %xmm6 

// CHECK: vroundss $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0a,0x7c,0x82,0xc0,0x00]     
vroundss $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vroundss $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0a,0x7c,0x82,0x40,0x00]     
vroundss $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vroundss $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0a,0x74,0x82,0xc0,0x00]     
vroundss $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vroundss $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0a,0x74,0x82,0x40,0x00]     
vroundss $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vroundss $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0a,0x7c,0x02,0x40,0x00]     
vroundss $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vroundss $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0a,0x74,0x02,0x40,0x00]     
vroundss $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vroundss $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0a,0x7a,0x40,0x00]     
vroundss $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vroundss $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0a,0x72,0x40,0x00]     
vroundss $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vroundss $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x0a,0x3a,0x00]     
vroundss $0, (%rdx), %xmm15, %xmm15 

// CHECK: vroundss $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0a,0x32,0x00]     
vroundss $0, (%rdx), %xmm6, %xmm6 

// CHECK: vroundss $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x0a,0xff,0x00]     
vroundss $0, %xmm15, %xmm15, %xmm15 

// CHECK: vroundss $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x0a,0xf6,0x00]     
vroundss $0, %xmm6, %xmm6, %xmm6 

// CHECK: vrsqrtps 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x52,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096, %xmm15 

// CHECK: vrsqrtps 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x52,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096, %xmm6 

// CHECK: vrsqrtps 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x52,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096, %ymm7 

// CHECK: vrsqrtps 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x52,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vrsqrtps 485498096, %ymm9 

// CHECK: vrsqrtps -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x52,0x7c,0x82,0xc0]       
vrsqrtps -64(%rdx,%rax,4), %xmm15 

// CHECK: vrsqrtps 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x52,0x7c,0x82,0x40]       
vrsqrtps 64(%rdx,%rax,4), %xmm15 

// CHECK: vrsqrtps -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x52,0x74,0x82,0xc0]       
vrsqrtps -64(%rdx,%rax,4), %xmm6 

// CHECK: vrsqrtps 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x52,0x74,0x82,0x40]       
vrsqrtps 64(%rdx,%rax,4), %xmm6 

// CHECK: vrsqrtps -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x52,0x7c,0x82,0xc0]       
vrsqrtps -64(%rdx,%rax,4), %ymm7 

// CHECK: vrsqrtps 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x52,0x7c,0x82,0x40]       
vrsqrtps 64(%rdx,%rax,4), %ymm7 

// CHECK: vrsqrtps -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x52,0x4c,0x82,0xc0]       
vrsqrtps -64(%rdx,%rax,4), %ymm9 

// CHECK: vrsqrtps 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x52,0x4c,0x82,0x40]       
vrsqrtps 64(%rdx,%rax,4), %ymm9 

// CHECK: vrsqrtps 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x52,0x7c,0x02,0x40]       
vrsqrtps 64(%rdx,%rax), %xmm15 

// CHECK: vrsqrtps 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x52,0x74,0x02,0x40]       
vrsqrtps 64(%rdx,%rax), %xmm6 

// CHECK: vrsqrtps 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x52,0x7c,0x02,0x40]       
vrsqrtps 64(%rdx,%rax), %ymm7 

// CHECK: vrsqrtps 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x52,0x4c,0x02,0x40]       
vrsqrtps 64(%rdx,%rax), %ymm9 

// CHECK: vrsqrtps 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x52,0x7a,0x40]       
vrsqrtps 64(%rdx), %xmm15 

// CHECK: vrsqrtps 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x52,0x72,0x40]       
vrsqrtps 64(%rdx), %xmm6 

// CHECK: vrsqrtps 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x52,0x7a,0x40]       
vrsqrtps 64(%rdx), %ymm7 

// CHECK: vrsqrtps 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x52,0x4a,0x40]       
vrsqrtps 64(%rdx), %ymm9 

// CHECK: vrsqrtps (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x52,0x3a]       
vrsqrtps (%rdx), %xmm15 

// CHECK: vrsqrtps (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x52,0x32]       
vrsqrtps (%rdx), %xmm6 

// CHECK: vrsqrtps (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x52,0x3a]       
vrsqrtps (%rdx), %ymm7 

// CHECK: vrsqrtps (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x52,0x0a]       
vrsqrtps (%rdx), %ymm9 

// CHECK: vrsqrtps %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x52,0xff]       
vrsqrtps %xmm15, %xmm15 

// CHECK: vrsqrtps %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x52,0xf6]       
vrsqrtps %xmm6, %xmm6 

// CHECK: vrsqrtps %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x52,0xff]       
vrsqrtps %ymm7, %ymm7 

// CHECK: vrsqrtps %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7c,0x52,0xc9]       
vrsqrtps %ymm9, %ymm9 

// CHECK: vrsqrtss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x52,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vrsqrtss 485498096, %xmm15, %xmm15 

// CHECK: vrsqrtss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x52,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vrsqrtss 485498096, %xmm6, %xmm6 

// CHECK: vrsqrtss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x52,0x7c,0x82,0xc0]      
vrsqrtss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vrsqrtss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x52,0x7c,0x82,0x40]      
vrsqrtss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vrsqrtss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x52,0x74,0x82,0xc0]      
vrsqrtss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vrsqrtss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x52,0x74,0x82,0x40]      
vrsqrtss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vrsqrtss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x52,0x7c,0x02,0x40]      
vrsqrtss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vrsqrtss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x52,0x74,0x02,0x40]      
vrsqrtss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vrsqrtss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x52,0x7a,0x40]      
vrsqrtss 64(%rdx), %xmm15, %xmm15 

// CHECK: vrsqrtss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x52,0x72,0x40]      
vrsqrtss 64(%rdx), %xmm6, %xmm6 

// CHECK: vrsqrtss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x52,0x3a]      
vrsqrtss (%rdx), %xmm15, %xmm15 

// CHECK: vrsqrtss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x52,0x32]      
vrsqrtss (%rdx), %xmm6, %xmm6 

// CHECK: vrsqrtss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x52,0xff]      
vrsqrtss %xmm15, %xmm15, %xmm15 

// CHECK: vrsqrtss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x52,0xf6]      
vrsqrtss %xmm6, %xmm6, %xmm6 

// CHECK: vshufpd $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096, %xmm15, %xmm15 

// CHECK: vshufpd $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc6,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096, %xmm6, %xmm6 

// CHECK: vshufpd $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096, %ymm7, %ymm7 

// CHECK: vshufpd $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufpd $0, 485498096, %ymm9, %ymm9 

// CHECK: vshufpd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc6,0x7c,0x82,0xc0,0x00]     
vshufpd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vshufpd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc6,0x7c,0x82,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vshufpd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc6,0x74,0x82,0xc0,0x00]     
vshufpd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vshufpd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc6,0x74,0x82,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vshufpd $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc6,0x7c,0x82,0xc0,0x00]     
vshufpd $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vshufpd $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc6,0x7c,0x82,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vshufpd $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc6,0x4c,0x82,0xc0,0x00]     
vshufpd $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vshufpd $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc6,0x4c,0x82,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vshufpd $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc6,0x7c,0x02,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vshufpd $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc6,0x74,0x02,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vshufpd $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc6,0x7c,0x02,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vshufpd $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc6,0x4c,0x02,0x40,0x00]     
vshufpd $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vshufpd $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc6,0x7a,0x40,0x00]     
vshufpd $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vshufpd $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc6,0x72,0x40,0x00]     
vshufpd $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vshufpd $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc6,0x7a,0x40,0x00]     
vshufpd $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vshufpd $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc6,0x4a,0x40,0x00]     
vshufpd $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vshufpd $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0xc6,0x3a,0x00]     
vshufpd $0, (%rdx), %xmm15, %xmm15 

// CHECK: vshufpd $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc6,0x32,0x00]     
vshufpd $0, (%rdx), %xmm6, %xmm6 

// CHECK: vshufpd $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc6,0x3a,0x00]     
vshufpd $0, (%rdx), %ymm7, %ymm7 

// CHECK: vshufpd $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xc6,0x0a,0x00]     
vshufpd $0, (%rdx), %ymm9, %ymm9 

// CHECK: vshufpd $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0xc6,0xff,0x00]     
vshufpd $0, %xmm15, %xmm15, %xmm15 

// CHECK: vshufpd $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0xc6,0xf6,0x00]     
vshufpd $0, %xmm6, %xmm6, %xmm6 

// CHECK: vshufpd $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xc6,0xff,0x00]     
vshufpd $0, %ymm7, %ymm7, %ymm7 

// CHECK: vshufpd $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xc6,0xc9,0x00]     
vshufpd $0, %ymm9, %ymm9, %ymm9 

// CHECK: vshufps $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096, %xmm15, %xmm15 

// CHECK: vshufps $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc6,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096, %xmm6, %xmm6 

// CHECK: vshufps $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096, %ymm7, %ymm7 

// CHECK: vshufps $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vshufps $0, 485498096, %ymm9, %ymm9 

// CHECK: vshufps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc6,0x7c,0x82,0xc0,0x00]     
vshufps $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vshufps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc6,0x7c,0x82,0x40,0x00]     
vshufps $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vshufps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc6,0x74,0x82,0xc0,0x00]     
vshufps $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vshufps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc6,0x74,0x82,0x40,0x00]     
vshufps $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vshufps $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc6,0x7c,0x82,0xc0,0x00]     
vshufps $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vshufps $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc6,0x7c,0x82,0x40,0x00]     
vshufps $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vshufps $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc6,0x4c,0x82,0xc0,0x00]     
vshufps $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vshufps $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc6,0x4c,0x82,0x40,0x00]     
vshufps $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vshufps $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc6,0x7c,0x02,0x40,0x00]     
vshufps $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vshufps $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc6,0x74,0x02,0x40,0x00]     
vshufps $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vshufps $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc6,0x7c,0x02,0x40,0x00]     
vshufps $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vshufps $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc6,0x4c,0x02,0x40,0x00]     
vshufps $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vshufps $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc6,0x7a,0x40,0x00]     
vshufps $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vshufps $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc6,0x72,0x40,0x00]     
vshufps $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vshufps $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc6,0x7a,0x40,0x00]     
vshufps $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vshufps $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc6,0x4a,0x40,0x00]     
vshufps $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vshufps $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0xc6,0x3a,0x00]     
vshufps $0, (%rdx), %xmm15, %xmm15 

// CHECK: vshufps $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc6,0x32,0x00]     
vshufps $0, (%rdx), %xmm6, %xmm6 

// CHECK: vshufps $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc6,0x3a,0x00]     
vshufps $0, (%rdx), %ymm7, %ymm7 

// CHECK: vshufps $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0xc6,0x0a,0x00]     
vshufps $0, (%rdx), %ymm9, %ymm9 

// CHECK: vshufps $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0xc6,0xff,0x00]     
vshufps $0, %xmm15, %xmm15, %xmm15 

// CHECK: vshufps $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0xc6,0xf6,0x00]     
vshufps $0, %xmm6, %xmm6, %xmm6 

// CHECK: vshufps $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0xc6,0xff,0x00]     
vshufps $0, %ymm7, %ymm7, %ymm7 

// CHECK: vshufps $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0xc6,0xc9,0x00]     
vshufps $0, %ymm9, %ymm9, %ymm9 

// CHECK: vsqrtpd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096, %xmm15 

// CHECK: vsqrtpd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x51,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096, %xmm6 

// CHECK: vsqrtpd 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096, %ymm7 

// CHECK: vsqrtpd 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtpd 485498096, %ymm9 

// CHECK: vsqrtpd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x51,0x7c,0x82,0xc0]       
vsqrtpd -64(%rdx,%rax,4), %xmm15 

// CHECK: vsqrtpd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x51,0x7c,0x82,0x40]       
vsqrtpd 64(%rdx,%rax,4), %xmm15 

// CHECK: vsqrtpd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x51,0x74,0x82,0xc0]       
vsqrtpd -64(%rdx,%rax,4), %xmm6 

// CHECK: vsqrtpd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x51,0x74,0x82,0x40]       
vsqrtpd 64(%rdx,%rax,4), %xmm6 

// CHECK: vsqrtpd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x51,0x7c,0x82,0xc0]       
vsqrtpd -64(%rdx,%rax,4), %ymm7 

// CHECK: vsqrtpd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x51,0x7c,0x82,0x40]       
vsqrtpd 64(%rdx,%rax,4), %ymm7 

// CHECK: vsqrtpd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x51,0x4c,0x82,0xc0]       
vsqrtpd -64(%rdx,%rax,4), %ymm9 

// CHECK: vsqrtpd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x51,0x4c,0x82,0x40]       
vsqrtpd 64(%rdx,%rax,4), %ymm9 

// CHECK: vsqrtpd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x51,0x7c,0x02,0x40]       
vsqrtpd 64(%rdx,%rax), %xmm15 

// CHECK: vsqrtpd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x51,0x74,0x02,0x40]       
vsqrtpd 64(%rdx,%rax), %xmm6 

// CHECK: vsqrtpd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x51,0x7c,0x02,0x40]       
vsqrtpd 64(%rdx,%rax), %ymm7 

// CHECK: vsqrtpd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x51,0x4c,0x02,0x40]       
vsqrtpd 64(%rdx,%rax), %ymm9 

// CHECK: vsqrtpd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x51,0x7a,0x40]       
vsqrtpd 64(%rdx), %xmm15 

// CHECK: vsqrtpd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x51,0x72,0x40]       
vsqrtpd 64(%rdx), %xmm6 

// CHECK: vsqrtpd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x51,0x7a,0x40]       
vsqrtpd 64(%rdx), %ymm7 

// CHECK: vsqrtpd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x51,0x4a,0x40]       
vsqrtpd 64(%rdx), %ymm9 

// CHECK: vsqrtpd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x51,0x3a]       
vsqrtpd (%rdx), %xmm15 

// CHECK: vsqrtpd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x51,0x32]       
vsqrtpd (%rdx), %xmm6 

// CHECK: vsqrtpd (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x51,0x3a]       
vsqrtpd (%rdx), %ymm7 

// CHECK: vsqrtpd (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x51,0x0a]       
vsqrtpd (%rdx), %ymm9 

// CHECK: vsqrtpd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x51,0xff]       
vsqrtpd %xmm15, %xmm15 

// CHECK: vsqrtpd %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x51,0xf6]       
vsqrtpd %xmm6, %xmm6 

// CHECK: vsqrtpd %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x51,0xff]       
vsqrtpd %ymm7, %ymm7 

// CHECK: vsqrtpd %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7d,0x51,0xc9]       
vsqrtpd %ymm9, %ymm9 

// CHECK: vsqrtps 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096, %xmm15 

// CHECK: vsqrtps 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x51,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096, %xmm6 

// CHECK: vsqrtps 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096, %ymm7 

// CHECK: vsqrtps 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vsqrtps 485498096, %ymm9 

// CHECK: vsqrtps -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x51,0x7c,0x82,0xc0]       
vsqrtps -64(%rdx,%rax,4), %xmm15 

// CHECK: vsqrtps 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x51,0x7c,0x82,0x40]       
vsqrtps 64(%rdx,%rax,4), %xmm15 

// CHECK: vsqrtps -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x51,0x74,0x82,0xc0]       
vsqrtps -64(%rdx,%rax,4), %xmm6 

// CHECK: vsqrtps 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x51,0x74,0x82,0x40]       
vsqrtps 64(%rdx,%rax,4), %xmm6 

// CHECK: vsqrtps -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x51,0x7c,0x82,0xc0]       
vsqrtps -64(%rdx,%rax,4), %ymm7 

// CHECK: vsqrtps 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x51,0x7c,0x82,0x40]       
vsqrtps 64(%rdx,%rax,4), %ymm7 

// CHECK: vsqrtps -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x51,0x4c,0x82,0xc0]       
vsqrtps -64(%rdx,%rax,4), %ymm9 

// CHECK: vsqrtps 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x51,0x4c,0x82,0x40]       
vsqrtps 64(%rdx,%rax,4), %ymm9 

// CHECK: vsqrtps 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x51,0x7c,0x02,0x40]       
vsqrtps 64(%rdx,%rax), %xmm15 

// CHECK: vsqrtps 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x51,0x74,0x02,0x40]       
vsqrtps 64(%rdx,%rax), %xmm6 

// CHECK: vsqrtps 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x51,0x7c,0x02,0x40]       
vsqrtps 64(%rdx,%rax), %ymm7 

// CHECK: vsqrtps 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x51,0x4c,0x02,0x40]       
vsqrtps 64(%rdx,%rax), %ymm9 

// CHECK: vsqrtps 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x51,0x7a,0x40]       
vsqrtps 64(%rdx), %xmm15 

// CHECK: vsqrtps 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x51,0x72,0x40]       
vsqrtps 64(%rdx), %xmm6 

// CHECK: vsqrtps 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x51,0x7a,0x40]       
vsqrtps 64(%rdx), %ymm7 

// CHECK: vsqrtps 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x51,0x4a,0x40]       
vsqrtps 64(%rdx), %ymm9 

// CHECK: vsqrtps (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x51,0x3a]       
vsqrtps (%rdx), %xmm15 

// CHECK: vsqrtps (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x51,0x32]       
vsqrtps (%rdx), %xmm6 

// CHECK: vsqrtps (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x51,0x3a]       
vsqrtps (%rdx), %ymm7 

// CHECK: vsqrtps (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7c,0x51,0x0a]       
vsqrtps (%rdx), %ymm9 

// CHECK: vsqrtps %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x51,0xff]       
vsqrtps %xmm15, %xmm15 

// CHECK: vsqrtps %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x51,0xf6]       
vsqrtps %xmm6, %xmm6 

// CHECK: vsqrtps %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfc,0x51,0xff]       
vsqrtps %ymm7, %ymm7 

// CHECK: vsqrtps %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7c,0x51,0xc9]       
vsqrtps %ymm9, %ymm9 

// CHECK: vsqrtsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsqrtsd 485498096, %xmm15, %xmm15 

// CHECK: vsqrtsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x51,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vsqrtsd 485498096, %xmm6, %xmm6 

// CHECK: vsqrtsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x51,0x7c,0x82,0xc0]      
vsqrtsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsqrtsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x51,0x7c,0x82,0x40]      
vsqrtsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsqrtsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x51,0x74,0x82,0xc0]      
vsqrtsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsqrtsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x51,0x74,0x82,0x40]      
vsqrtsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsqrtsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x51,0x7c,0x02,0x40]      
vsqrtsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vsqrtsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x51,0x74,0x02,0x40]      
vsqrtsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vsqrtsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x51,0x7a,0x40]      
vsqrtsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vsqrtsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x51,0x72,0x40]      
vsqrtsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vsqrtsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x51,0x3a]      
vsqrtsd (%rdx), %xmm15, %xmm15 

// CHECK: vsqrtsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x51,0x32]      
vsqrtsd (%rdx), %xmm6, %xmm6 

// CHECK: vsqrtsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x51,0xff]      
vsqrtsd %xmm15, %xmm15, %xmm15 

// CHECK: vsqrtsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x51,0xf6]      
vsqrtsd %xmm6, %xmm6, %xmm6 

// CHECK: vsqrtss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsqrtss 485498096, %xmm15, %xmm15 

// CHECK: vsqrtss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x51,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vsqrtss 485498096, %xmm6, %xmm6 

// CHECK: vsqrtss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x51,0x7c,0x82,0xc0]      
vsqrtss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsqrtss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x51,0x7c,0x82,0x40]      
vsqrtss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsqrtss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x51,0x74,0x82,0xc0]      
vsqrtss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsqrtss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x51,0x74,0x82,0x40]      
vsqrtss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsqrtss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x51,0x7c,0x02,0x40]      
vsqrtss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vsqrtss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x51,0x74,0x02,0x40]      
vsqrtss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vsqrtss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x51,0x7a,0x40]      
vsqrtss 64(%rdx), %xmm15, %xmm15 

// CHECK: vsqrtss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x51,0x72,0x40]      
vsqrtss 64(%rdx), %xmm6, %xmm6 

// CHECK: vsqrtss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x51,0x3a]      
vsqrtss (%rdx), %xmm15, %xmm15 

// CHECK: vsqrtss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x51,0x32]      
vsqrtss (%rdx), %xmm6, %xmm6 

// CHECK: vsqrtss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x51,0xff]      
vsqrtss %xmm15, %xmm15, %xmm15 

// CHECK: vsqrtss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x51,0xf6]      
vsqrtss %xmm6, %xmm6, %xmm6 

// CHECK: vstmxcsr 485498096 
// CHECK: encoding: [0xc5,0xf8,0xae,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
vstmxcsr 485498096 

// CHECK: vstmxcsr 64(%rdx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x5a,0x40]        
vstmxcsr 64(%rdx) 

// CHECK: vstmxcsr -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x5c,0x82,0xc0]        
vstmxcsr -64(%rdx,%rax,4) 

// CHECK: vstmxcsr 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x5c,0x82,0x40]        
vstmxcsr 64(%rdx,%rax,4) 

// CHECK: vstmxcsr 64(%rdx,%rax) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x5c,0x02,0x40]        
vstmxcsr 64(%rdx,%rax) 

// CHECK: vstmxcsr (%rdx) 
// CHECK: encoding: [0xc5,0xf8,0xae,0x1a]        
vstmxcsr (%rdx) 

// CHECK: vsubpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096, %xmm15, %xmm15 

// CHECK: vsubpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096, %xmm6, %xmm6 

// CHECK: vsubpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096, %ymm7, %ymm7 

// CHECK: vsubpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubpd 485498096, %ymm9, %ymm9 

// CHECK: vsubpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5c,0x7c,0x82,0xc0]      
vsubpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5c,0x7c,0x82,0x40]      
vsubpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5c,0x74,0x82,0xc0]      
vsubpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5c,0x74,0x82,0x40]      
vsubpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5c,0x7c,0x82,0xc0]      
vsubpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vsubpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5c,0x7c,0x82,0x40]      
vsubpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vsubpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5c,0x4c,0x82,0xc0]      
vsubpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vsubpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5c,0x4c,0x82,0x40]      
vsubpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vsubpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5c,0x7c,0x02,0x40]      
vsubpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vsubpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5c,0x74,0x02,0x40]      
vsubpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vsubpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5c,0x7c,0x02,0x40]      
vsubpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vsubpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5c,0x4c,0x02,0x40]      
vsubpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vsubpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5c,0x7a,0x40]      
vsubpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vsubpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5c,0x72,0x40]      
vsubpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vsubpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5c,0x7a,0x40]      
vsubpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vsubpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5c,0x4a,0x40]      
vsubpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vsubpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x5c,0x3a]      
vsubpd (%rdx), %xmm15, %xmm15 

// CHECK: vsubpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5c,0x32]      
vsubpd (%rdx), %xmm6, %xmm6 

// CHECK: vsubpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5c,0x3a]      
vsubpd (%rdx), %ymm7, %ymm7 

// CHECK: vsubpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x5c,0x0a]      
vsubpd (%rdx), %ymm9, %ymm9 

// CHECK: vsubpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x5c,0xff]      
vsubpd %xmm15, %xmm15, %xmm15 

// CHECK: vsubpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x5c,0xf6]      
vsubpd %xmm6, %xmm6, %xmm6 

// CHECK: vsubpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x5c,0xff]      
vsubpd %ymm7, %ymm7, %ymm7 

// CHECK: vsubpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x5c,0xc9]      
vsubpd %ymm9, %ymm9, %ymm9 

// CHECK: vsubps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096, %xmm15, %xmm15 

// CHECK: vsubps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096, %xmm6, %xmm6 

// CHECK: vsubps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096, %ymm7, %ymm7 

// CHECK: vsubps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubps 485498096, %ymm9, %ymm9 

// CHECK: vsubps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5c,0x7c,0x82,0xc0]      
vsubps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5c,0x7c,0x82,0x40]      
vsubps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5c,0x74,0x82,0xc0]      
vsubps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5c,0x74,0x82,0x40]      
vsubps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5c,0x7c,0x82,0xc0]      
vsubps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vsubps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5c,0x7c,0x82,0x40]      
vsubps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vsubps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5c,0x4c,0x82,0xc0]      
vsubps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vsubps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5c,0x4c,0x82,0x40]      
vsubps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vsubps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5c,0x7c,0x02,0x40]      
vsubps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vsubps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5c,0x74,0x02,0x40]      
vsubps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vsubps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5c,0x7c,0x02,0x40]      
vsubps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vsubps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5c,0x4c,0x02,0x40]      
vsubps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vsubps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5c,0x7a,0x40]      
vsubps 64(%rdx), %xmm15, %xmm15 

// CHECK: vsubps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5c,0x72,0x40]      
vsubps 64(%rdx), %xmm6, %xmm6 

// CHECK: vsubps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5c,0x7a,0x40]      
vsubps 64(%rdx), %ymm7, %ymm7 

// CHECK: vsubps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5c,0x4a,0x40]      
vsubps 64(%rdx), %ymm9, %ymm9 

// CHECK: vsubps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x5c,0x3a]      
vsubps (%rdx), %xmm15, %xmm15 

// CHECK: vsubps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5c,0x32]      
vsubps (%rdx), %xmm6, %xmm6 

// CHECK: vsubps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5c,0x3a]      
vsubps (%rdx), %ymm7, %ymm7 

// CHECK: vsubps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x5c,0x0a]      
vsubps (%rdx), %ymm9, %ymm9 

// CHECK: vsubps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x5c,0xff]      
vsubps %xmm15, %xmm15, %xmm15 

// CHECK: vsubps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x5c,0xf6]      
vsubps %xmm6, %xmm6, %xmm6 

// CHECK: vsubps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x5c,0xff]      
vsubps %ymm7, %ymm7, %ymm7 

// CHECK: vsubps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x5c,0xc9]      
vsubps %ymm9, %ymm9, %ymm9 

// CHECK: vsubsd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubsd 485498096, %xmm15, %xmm15 

// CHECK: vsubsd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubsd 485498096, %xmm6, %xmm6 

// CHECK: vsubsd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5c,0x7c,0x82,0xc0]      
vsubsd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubsd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5c,0x7c,0x82,0x40]      
vsubsd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubsd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5c,0x74,0x82,0xc0]      
vsubsd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubsd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5c,0x74,0x82,0x40]      
vsubsd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubsd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5c,0x7c,0x02,0x40]      
vsubsd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vsubsd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5c,0x74,0x02,0x40]      
vsubsd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vsubsd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5c,0x7a,0x40]      
vsubsd 64(%rdx), %xmm15, %xmm15 

// CHECK: vsubsd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5c,0x72,0x40]      
vsubsd 64(%rdx), %xmm6, %xmm6 

// CHECK: vsubsd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x03,0x5c,0x3a]      
vsubsd (%rdx), %xmm15, %xmm15 

// CHECK: vsubsd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5c,0x32]      
vsubsd (%rdx), %xmm6, %xmm6 

// CHECK: vsubsd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x03,0x5c,0xff]      
vsubsd %xmm15, %xmm15, %xmm15 

// CHECK: vsubsd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xcb,0x5c,0xf6]      
vsubsd %xmm6, %xmm6, %xmm6 

// CHECK: vsubss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubss 485498096, %xmm15, %xmm15 

// CHECK: vsubss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vsubss 485498096, %xmm6, %xmm6 

// CHECK: vsubss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5c,0x7c,0x82,0xc0]      
vsubss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5c,0x7c,0x82,0x40]      
vsubss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vsubss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5c,0x74,0x82,0xc0]      
vsubss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5c,0x74,0x82,0x40]      
vsubss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vsubss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5c,0x7c,0x02,0x40]      
vsubss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vsubss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5c,0x74,0x02,0x40]      
vsubss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vsubss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5c,0x7a,0x40]      
vsubss 64(%rdx), %xmm15, %xmm15 

// CHECK: vsubss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5c,0x72,0x40]      
vsubss 64(%rdx), %xmm6, %xmm6 

// CHECK: vsubss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x02,0x5c,0x3a]      
vsubss (%rdx), %xmm15, %xmm15 

// CHECK: vsubss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5c,0x32]      
vsubss (%rdx), %xmm6, %xmm6 

// CHECK: vsubss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x02,0x5c,0xff]      
vsubss %xmm15, %xmm15, %xmm15 

// CHECK: vsubss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xca,0x5c,0xf6]      
vsubss %xmm6, %xmm6, %xmm6 

// CHECK: vtestpd 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096, %xmm15 

// CHECK: vtestpd 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096, %xmm6 

// CHECK: vtestpd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096, %ymm7 

// CHECK: vtestpd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestpd 485498096, %ymm9 

// CHECK: vtestpd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0f,0x7c,0x82,0xc0]       
vtestpd -64(%rdx,%rax,4), %xmm15 

// CHECK: vtestpd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0f,0x7c,0x82,0x40]       
vtestpd 64(%rdx,%rax,4), %xmm15 

// CHECK: vtestpd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x74,0x82,0xc0]       
vtestpd -64(%rdx,%rax,4), %xmm6 

// CHECK: vtestpd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x74,0x82,0x40]       
vtestpd 64(%rdx,%rax,4), %xmm6 

// CHECK: vtestpd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x7c,0x82,0xc0]       
vtestpd -64(%rdx,%rax,4), %ymm7 

// CHECK: vtestpd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x7c,0x82,0x40]       
vtestpd 64(%rdx,%rax,4), %ymm7 

// CHECK: vtestpd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0f,0x4c,0x82,0xc0]       
vtestpd -64(%rdx,%rax,4), %ymm9 

// CHECK: vtestpd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0f,0x4c,0x82,0x40]       
vtestpd 64(%rdx,%rax,4), %ymm9 

// CHECK: vtestpd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0f,0x7c,0x02,0x40]       
vtestpd 64(%rdx,%rax), %xmm15 

// CHECK: vtestpd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x74,0x02,0x40]       
vtestpd 64(%rdx,%rax), %xmm6 

// CHECK: vtestpd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x7c,0x02,0x40]       
vtestpd 64(%rdx,%rax), %ymm7 

// CHECK: vtestpd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0f,0x4c,0x02,0x40]       
vtestpd 64(%rdx,%rax), %ymm9 

// CHECK: vtestpd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0f,0x7a,0x40]       
vtestpd 64(%rdx), %xmm15 

// CHECK: vtestpd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x72,0x40]       
vtestpd 64(%rdx), %xmm6 

// CHECK: vtestpd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x7a,0x40]       
vtestpd 64(%rdx), %ymm7 

// CHECK: vtestpd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0f,0x4a,0x40]       
vtestpd 64(%rdx), %ymm9 

// CHECK: vtestpd (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0f,0x3a]       
vtestpd (%rdx), %xmm15 

// CHECK: vtestpd (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x32]       
vtestpd (%rdx), %xmm6 

// CHECK: vtestpd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x3a]       
vtestpd (%rdx), %ymm7 

// CHECK: vtestpd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0f,0x0a]       
vtestpd (%rdx), %ymm9 

// CHECK: vtestpd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x0f,0xff]       
vtestpd %xmm15, %xmm15 

// CHECK: vtestpd %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0xf6]       
vtestpd %xmm6, %xmm6 

// CHECK: vtestpd %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0xff]       
vtestpd %ymm7, %ymm7 

// CHECK: vtestpd %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x0f,0xc9]       
vtestpd %ymm9, %ymm9 

// CHECK: vtestps 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096, %xmm15 

// CHECK: vtestps 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096, %xmm6 

// CHECK: vtestps 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096, %ymm7 

// CHECK: vtestps 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vtestps 485498096, %ymm9 

// CHECK: vtestps -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0e,0x7c,0x82,0xc0]       
vtestps -64(%rdx,%rax,4), %xmm15 

// CHECK: vtestps 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0e,0x7c,0x82,0x40]       
vtestps 64(%rdx,%rax,4), %xmm15 

// CHECK: vtestps -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x74,0x82,0xc0]       
vtestps -64(%rdx,%rax,4), %xmm6 

// CHECK: vtestps 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x74,0x82,0x40]       
vtestps 64(%rdx,%rax,4), %xmm6 

// CHECK: vtestps -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x7c,0x82,0xc0]       
vtestps -64(%rdx,%rax,4), %ymm7 

// CHECK: vtestps 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x7c,0x82,0x40]       
vtestps 64(%rdx,%rax,4), %ymm7 

// CHECK: vtestps -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0e,0x4c,0x82,0xc0]       
vtestps -64(%rdx,%rax,4), %ymm9 

// CHECK: vtestps 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0e,0x4c,0x82,0x40]       
vtestps 64(%rdx,%rax,4), %ymm9 

// CHECK: vtestps 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0e,0x7c,0x02,0x40]       
vtestps 64(%rdx,%rax), %xmm15 

// CHECK: vtestps 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x74,0x02,0x40]       
vtestps 64(%rdx,%rax), %xmm6 

// CHECK: vtestps 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x7c,0x02,0x40]       
vtestps 64(%rdx,%rax), %ymm7 

// CHECK: vtestps 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0e,0x4c,0x02,0x40]       
vtestps 64(%rdx,%rax), %ymm9 

// CHECK: vtestps 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0e,0x7a,0x40]       
vtestps 64(%rdx), %xmm15 

// CHECK: vtestps 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x72,0x40]       
vtestps 64(%rdx), %xmm6 

// CHECK: vtestps 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x7a,0x40]       
vtestps 64(%rdx), %ymm7 

// CHECK: vtestps 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0e,0x4a,0x40]       
vtestps 64(%rdx), %ymm9 

// CHECK: vtestps (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x0e,0x3a]       
vtestps (%rdx), %xmm15 

// CHECK: vtestps (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x32]       
vtestps (%rdx), %xmm6 

// CHECK: vtestps (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x3a]       
vtestps (%rdx), %ymm7 

// CHECK: vtestps (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x0e,0x0a]       
vtestps (%rdx), %ymm9 

// CHECK: vtestps %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x0e,0xff]       
vtestps %xmm15, %xmm15 

// CHECK: vtestps %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0xf6]       
vtestps %xmm6, %xmm6 

// CHECK: vtestps %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0xff]       
vtestps %ymm7, %ymm7 

// CHECK: vtestps %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x0e,0xc9]       
vtestps %ymm9, %ymm9 

// CHECK: vucomisd 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vucomisd 485498096, %xmm15 

// CHECK: vucomisd 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vucomisd 485498096, %xmm6 

// CHECK: vucomisd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2e,0x7c,0x82,0xc0]       
vucomisd -64(%rdx,%rax,4), %xmm15 

// CHECK: vucomisd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2e,0x7c,0x82,0x40]       
vucomisd 64(%rdx,%rax,4), %xmm15 

// CHECK: vucomisd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x74,0x82,0xc0]       
vucomisd -64(%rdx,%rax,4), %xmm6 

// CHECK: vucomisd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x74,0x82,0x40]       
vucomisd 64(%rdx,%rax,4), %xmm6 

// CHECK: vucomisd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2e,0x7c,0x02,0x40]       
vucomisd 64(%rdx,%rax), %xmm15 

// CHECK: vucomisd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x74,0x02,0x40]       
vucomisd 64(%rdx,%rax), %xmm6 

// CHECK: vucomisd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2e,0x7a,0x40]       
vucomisd 64(%rdx), %xmm15 

// CHECK: vucomisd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x72,0x40]       
vucomisd 64(%rdx), %xmm6 

// CHECK: vucomisd (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x79,0x2e,0x3a]       
vucomisd (%rdx), %xmm15 

// CHECK: vucomisd (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2e,0x32]       
vucomisd (%rdx), %xmm6 

// CHECK: vucomisd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x79,0x2e,0xff]       
vucomisd %xmm15, %xmm15 

// CHECK: vucomisd %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf9,0x2e,0xf6]       
vucomisd %xmm6, %xmm6 

// CHECK: vucomiss 485498096, %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vucomiss 485498096, %xmm15 

// CHECK: vucomiss 485498096, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vucomiss 485498096, %xmm6 

// CHECK: vucomiss -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2e,0x7c,0x82,0xc0]       
vucomiss -64(%rdx,%rax,4), %xmm15 

// CHECK: vucomiss 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2e,0x7c,0x82,0x40]       
vucomiss 64(%rdx,%rax,4), %xmm15 

// CHECK: vucomiss -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x74,0x82,0xc0]       
vucomiss -64(%rdx,%rax,4), %xmm6 

// CHECK: vucomiss 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x74,0x82,0x40]       
vucomiss 64(%rdx,%rax,4), %xmm6 

// CHECK: vucomiss 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2e,0x7c,0x02,0x40]       
vucomiss 64(%rdx,%rax), %xmm15 

// CHECK: vucomiss 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x74,0x02,0x40]       
vucomiss 64(%rdx,%rax), %xmm6 

// CHECK: vucomiss 64(%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2e,0x7a,0x40]       
vucomiss 64(%rdx), %xmm15 

// CHECK: vucomiss 64(%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x72,0x40]       
vucomiss 64(%rdx), %xmm6 

// CHECK: vucomiss (%rdx), %xmm15 
// CHECK: encoding: [0xc5,0x78,0x2e,0x3a]       
vucomiss (%rdx), %xmm15 

// CHECK: vucomiss (%rdx), %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2e,0x32]       
vucomiss (%rdx), %xmm6 

// CHECK: vucomiss %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x78,0x2e,0xff]       
vucomiss %xmm15, %xmm15 

// CHECK: vucomiss %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xf8,0x2e,0xf6]       
vucomiss %xmm6, %xmm6 

// CHECK: vunpckhpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x15,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096, %xmm15, %xmm15 

// CHECK: vunpckhpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x15,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096, %xmm6, %xmm6 

// CHECK: vunpckhpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x15,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096, %ymm7, %ymm7 

// CHECK: vunpckhpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x15,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhpd 485498096, %ymm9, %ymm9 

// CHECK: vunpckhpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x15,0x7c,0x82,0xc0]      
vunpckhpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpckhpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x15,0x7c,0x82,0x40]      
vunpckhpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpckhpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x15,0x74,0x82,0xc0]      
vunpckhpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpckhpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x15,0x74,0x82,0x40]      
vunpckhpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpckhpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x15,0x7c,0x82,0xc0]      
vunpckhpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpckhpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x15,0x7c,0x82,0x40]      
vunpckhpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpckhpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x15,0x4c,0x82,0xc0]      
vunpckhpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpckhpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x15,0x4c,0x82,0x40]      
vunpckhpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpckhpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x15,0x7c,0x02,0x40]      
vunpckhpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vunpckhpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x15,0x74,0x02,0x40]      
vunpckhpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vunpckhpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x15,0x7c,0x02,0x40]      
vunpckhpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vunpckhpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x15,0x4c,0x02,0x40]      
vunpckhpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vunpckhpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x15,0x7a,0x40]      
vunpckhpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vunpckhpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x15,0x72,0x40]      
vunpckhpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vunpckhpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x15,0x7a,0x40]      
vunpckhpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vunpckhpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x15,0x4a,0x40]      
vunpckhpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vunpckhpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x15,0x3a]      
vunpckhpd (%rdx), %xmm15, %xmm15 

// CHECK: vunpckhpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x15,0x32]      
vunpckhpd (%rdx), %xmm6, %xmm6 

// CHECK: vunpckhpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x15,0x3a]      
vunpckhpd (%rdx), %ymm7, %ymm7 

// CHECK: vunpckhpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x15,0x0a]      
vunpckhpd (%rdx), %ymm9, %ymm9 

// CHECK: vunpckhpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x15,0xff]      
vunpckhpd %xmm15, %xmm15, %xmm15 

// CHECK: vunpckhpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x15,0xf6]      
vunpckhpd %xmm6, %xmm6, %xmm6 

// CHECK: vunpckhpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x15,0xff]      
vunpckhpd %ymm7, %ymm7, %ymm7 

// CHECK: vunpckhpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x15,0xc9]      
vunpckhpd %ymm9, %ymm9, %ymm9 

// CHECK: vunpckhps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x15,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096, %xmm15, %xmm15 

// CHECK: vunpckhps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x15,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096, %xmm6, %xmm6 

// CHECK: vunpckhps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x15,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096, %ymm7, %ymm7 

// CHECK: vunpckhps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x15,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpckhps 485498096, %ymm9, %ymm9 

// CHECK: vunpckhps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x15,0x7c,0x82,0xc0]      
vunpckhps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpckhps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x15,0x7c,0x82,0x40]      
vunpckhps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpckhps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x15,0x74,0x82,0xc0]      
vunpckhps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpckhps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x15,0x74,0x82,0x40]      
vunpckhps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpckhps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x15,0x7c,0x82,0xc0]      
vunpckhps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpckhps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x15,0x7c,0x82,0x40]      
vunpckhps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpckhps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x15,0x4c,0x82,0xc0]      
vunpckhps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpckhps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x15,0x4c,0x82,0x40]      
vunpckhps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpckhps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x15,0x7c,0x02,0x40]      
vunpckhps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vunpckhps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x15,0x74,0x02,0x40]      
vunpckhps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vunpckhps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x15,0x7c,0x02,0x40]      
vunpckhps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vunpckhps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x15,0x4c,0x02,0x40]      
vunpckhps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vunpckhps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x15,0x7a,0x40]      
vunpckhps 64(%rdx), %xmm15, %xmm15 

// CHECK: vunpckhps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x15,0x72,0x40]      
vunpckhps 64(%rdx), %xmm6, %xmm6 

// CHECK: vunpckhps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x15,0x7a,0x40]      
vunpckhps 64(%rdx), %ymm7, %ymm7 

// CHECK: vunpckhps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x15,0x4a,0x40]      
vunpckhps 64(%rdx), %ymm9, %ymm9 

// CHECK: vunpckhps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x15,0x3a]      
vunpckhps (%rdx), %xmm15, %xmm15 

// CHECK: vunpckhps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x15,0x32]      
vunpckhps (%rdx), %xmm6, %xmm6 

// CHECK: vunpckhps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x15,0x3a]      
vunpckhps (%rdx), %ymm7, %ymm7 

// CHECK: vunpckhps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x15,0x0a]      
vunpckhps (%rdx), %ymm9, %ymm9 

// CHECK: vunpckhps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x15,0xff]      
vunpckhps %xmm15, %xmm15, %xmm15 

// CHECK: vunpckhps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x15,0xf6]      
vunpckhps %xmm6, %xmm6, %xmm6 

// CHECK: vunpckhps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x15,0xff]      
vunpckhps %ymm7, %ymm7, %ymm7 

// CHECK: vunpckhps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x15,0xc9]      
vunpckhps %ymm9, %ymm9, %ymm9 

// CHECK: vunpcklpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x14,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096, %xmm15, %xmm15 

// CHECK: vunpcklpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x14,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096, %xmm6, %xmm6 

// CHECK: vunpcklpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x14,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096, %ymm7, %ymm7 

// CHECK: vunpcklpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x14,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklpd 485498096, %ymm9, %ymm9 

// CHECK: vunpcklpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x14,0x7c,0x82,0xc0]      
vunpcklpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpcklpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x14,0x7c,0x82,0x40]      
vunpcklpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpcklpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x14,0x74,0x82,0xc0]      
vunpcklpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpcklpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x14,0x74,0x82,0x40]      
vunpcklpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpcklpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x14,0x7c,0x82,0xc0]      
vunpcklpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpcklpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x14,0x7c,0x82,0x40]      
vunpcklpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpcklpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x14,0x4c,0x82,0xc0]      
vunpcklpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpcklpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x14,0x4c,0x82,0x40]      
vunpcklpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpcklpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x14,0x7c,0x02,0x40]      
vunpcklpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vunpcklpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x14,0x74,0x02,0x40]      
vunpcklpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vunpcklpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x14,0x7c,0x02,0x40]      
vunpcklpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vunpcklpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x14,0x4c,0x02,0x40]      
vunpcklpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vunpcklpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x14,0x7a,0x40]      
vunpcklpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vunpcklpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x14,0x72,0x40]      
vunpcklpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vunpcklpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x14,0x7a,0x40]      
vunpcklpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vunpcklpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x14,0x4a,0x40]      
vunpcklpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vunpcklpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x14,0x3a]      
vunpcklpd (%rdx), %xmm15, %xmm15 

// CHECK: vunpcklpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x14,0x32]      
vunpcklpd (%rdx), %xmm6, %xmm6 

// CHECK: vunpcklpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x14,0x3a]      
vunpcklpd (%rdx), %ymm7, %ymm7 

// CHECK: vunpcklpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x14,0x0a]      
vunpcklpd (%rdx), %ymm9, %ymm9 

// CHECK: vunpcklpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x14,0xff]      
vunpcklpd %xmm15, %xmm15, %xmm15 

// CHECK: vunpcklpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x14,0xf6]      
vunpcklpd %xmm6, %xmm6, %xmm6 

// CHECK: vunpcklpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x14,0xff]      
vunpcklpd %ymm7, %ymm7, %ymm7 

// CHECK: vunpcklpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x14,0xc9]      
vunpcklpd %ymm9, %ymm9, %ymm9 

// CHECK: vunpcklps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x14,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096, %xmm15, %xmm15 

// CHECK: vunpcklps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x14,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096, %xmm6, %xmm6 

// CHECK: vunpcklps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x14,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096, %ymm7, %ymm7 

// CHECK: vunpcklps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x14,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vunpcklps 485498096, %ymm9, %ymm9 

// CHECK: vunpcklps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x14,0x7c,0x82,0xc0]      
vunpcklps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpcklps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x14,0x7c,0x82,0x40]      
vunpcklps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vunpcklps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x14,0x74,0x82,0xc0]      
vunpcklps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpcklps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x14,0x74,0x82,0x40]      
vunpcklps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vunpcklps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x14,0x7c,0x82,0xc0]      
vunpcklps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpcklps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x14,0x7c,0x82,0x40]      
vunpcklps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vunpcklps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x14,0x4c,0x82,0xc0]      
vunpcklps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpcklps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x14,0x4c,0x82,0x40]      
vunpcklps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vunpcklps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x14,0x7c,0x02,0x40]      
vunpcklps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vunpcklps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x14,0x74,0x02,0x40]      
vunpcklps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vunpcklps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x14,0x7c,0x02,0x40]      
vunpcklps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vunpcklps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x14,0x4c,0x02,0x40]      
vunpcklps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vunpcklps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x14,0x7a,0x40]      
vunpcklps 64(%rdx), %xmm15, %xmm15 

// CHECK: vunpcklps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x14,0x72,0x40]      
vunpcklps 64(%rdx), %xmm6, %xmm6 

// CHECK: vunpcklps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x14,0x7a,0x40]      
vunpcklps 64(%rdx), %ymm7, %ymm7 

// CHECK: vunpcklps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x14,0x4a,0x40]      
vunpcklps 64(%rdx), %ymm9, %ymm9 

// CHECK: vunpcklps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x14,0x3a]      
vunpcklps (%rdx), %xmm15, %xmm15 

// CHECK: vunpcklps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x14,0x32]      
vunpcklps (%rdx), %xmm6, %xmm6 

// CHECK: vunpcklps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x14,0x3a]      
vunpcklps (%rdx), %ymm7, %ymm7 

// CHECK: vunpcklps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x14,0x0a]      
vunpcklps (%rdx), %ymm9, %ymm9 

// CHECK: vunpcklps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x14,0xff]      
vunpcklps %xmm15, %xmm15, %xmm15 

// CHECK: vunpcklps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x14,0xf6]      
vunpcklps %xmm6, %xmm6, %xmm6 

// CHECK: vunpcklps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x14,0xff]      
vunpcklps %ymm7, %ymm7, %ymm7 

// CHECK: vunpcklps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x14,0xc9]      
vunpcklps %ymm9, %ymm9, %ymm9 

// CHECK: vxorpd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x57,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096, %xmm15, %xmm15 

// CHECK: vxorpd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x57,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096, %xmm6, %xmm6 

// CHECK: vxorpd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x57,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096, %ymm7, %ymm7 

// CHECK: vxorpd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x57,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorpd 485498096, %ymm9, %ymm9 

// CHECK: vxorpd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x57,0x7c,0x82,0xc0]      
vxorpd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vxorpd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x57,0x7c,0x82,0x40]      
vxorpd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vxorpd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x57,0x74,0x82,0xc0]      
vxorpd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vxorpd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x57,0x74,0x82,0x40]      
vxorpd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vxorpd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x57,0x7c,0x82,0xc0]      
vxorpd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vxorpd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x57,0x7c,0x82,0x40]      
vxorpd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vxorpd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x57,0x4c,0x82,0xc0]      
vxorpd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vxorpd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x57,0x4c,0x82,0x40]      
vxorpd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vxorpd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x57,0x7c,0x02,0x40]      
vxorpd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vxorpd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x57,0x74,0x02,0x40]      
vxorpd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vxorpd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x57,0x7c,0x02,0x40]      
vxorpd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vxorpd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x57,0x4c,0x02,0x40]      
vxorpd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vxorpd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x57,0x7a,0x40]      
vxorpd 64(%rdx), %xmm15, %xmm15 

// CHECK: vxorpd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x57,0x72,0x40]      
vxorpd 64(%rdx), %xmm6, %xmm6 

// CHECK: vxorpd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x57,0x7a,0x40]      
vxorpd 64(%rdx), %ymm7, %ymm7 

// CHECK: vxorpd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x57,0x4a,0x40]      
vxorpd 64(%rdx), %ymm9, %ymm9 

// CHECK: vxorpd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x01,0x57,0x3a]      
vxorpd (%rdx), %xmm15, %xmm15 

// CHECK: vxorpd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x57,0x32]      
vxorpd (%rdx), %xmm6, %xmm6 

// CHECK: vxorpd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x57,0x3a]      
vxorpd (%rdx), %ymm7, %ymm7 

// CHECK: vxorpd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x57,0x0a]      
vxorpd (%rdx), %ymm9, %ymm9 

// CHECK: vxorpd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x01,0x57,0xff]      
vxorpd %xmm15, %xmm15, %xmm15 

// CHECK: vxorpd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc9,0x57,0xf6]      
vxorpd %xmm6, %xmm6, %xmm6 

// CHECK: vxorpd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x57,0xff]      
vxorpd %ymm7, %ymm7, %ymm7 

// CHECK: vxorpd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x57,0xc9]      
vxorpd %ymm9, %ymm9, %ymm9 

// CHECK: vxorps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x57,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096, %xmm15, %xmm15 

// CHECK: vxorps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x57,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096, %xmm6, %xmm6 

// CHECK: vxorps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x57,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096, %ymm7, %ymm7 

// CHECK: vxorps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x57,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vxorps 485498096, %ymm9, %ymm9 

// CHECK: vxorps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x57,0x7c,0x82,0xc0]      
vxorps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vxorps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x57,0x7c,0x82,0x40]      
vxorps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vxorps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x57,0x74,0x82,0xc0]      
vxorps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vxorps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x57,0x74,0x82,0x40]      
vxorps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vxorps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x57,0x7c,0x82,0xc0]      
vxorps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vxorps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x57,0x7c,0x82,0x40]      
vxorps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vxorps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x57,0x4c,0x82,0xc0]      
vxorps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vxorps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x57,0x4c,0x82,0x40]      
vxorps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vxorps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x57,0x7c,0x02,0x40]      
vxorps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vxorps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x57,0x74,0x02,0x40]      
vxorps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vxorps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x57,0x7c,0x02,0x40]      
vxorps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vxorps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x57,0x4c,0x02,0x40]      
vxorps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vxorps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x57,0x7a,0x40]      
vxorps 64(%rdx), %xmm15, %xmm15 

// CHECK: vxorps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x57,0x72,0x40]      
vxorps 64(%rdx), %xmm6, %xmm6 

// CHECK: vxorps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x57,0x7a,0x40]      
vxorps 64(%rdx), %ymm7, %ymm7 

// CHECK: vxorps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x57,0x4a,0x40]      
vxorps 64(%rdx), %ymm9, %ymm9 

// CHECK: vxorps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc5,0x00,0x57,0x3a]      
vxorps (%rdx), %xmm15, %xmm15 

// CHECK: vxorps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x57,0x32]      
vxorps (%rdx), %xmm6, %xmm6 

// CHECK: vxorps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x57,0x3a]      
vxorps (%rdx), %ymm7, %ymm7 

// CHECK: vxorps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x34,0x57,0x0a]      
vxorps (%rdx), %ymm9, %ymm9 

// CHECK: vxorps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x41,0x00,0x57,0xff]      
vxorps %xmm15, %xmm15, %xmm15 

// CHECK: vxorps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc5,0xc8,0x57,0xf6]      
vxorps %xmm6, %xmm6, %xmm6 

// CHECK: vxorps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc4,0x57,0xff]      
vxorps %ymm7, %ymm7, %ymm7 

// CHECK: vxorps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x34,0x57,0xc9]      
vxorps %ymm9, %ymm9, %ymm9 

// CHECK: vzeroall 
// CHECK: encoding: [0xc5,0xfc,0x77]         
vzeroall 

// CHECK: vzeroupper 
// CHECK: encoding: [0xc5,0xf8,0x77]         
vzeroupper 

