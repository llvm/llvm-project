// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vfmadd132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x98,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x98,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x98,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x98,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd132pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x98,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd132pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x98,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmadd132pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x98,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096, %xmm1, %xmm1 

// CHECK: vfmadd132pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x98,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096, %ymm4, %ymm4 

// CHECK: vfmadd132pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x98,0x4c,0x02,0x40]      
vfmadd132pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd132pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x98,0x64,0x02,0x40]      
vfmadd132pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmadd132pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x98,0x0a]      
vfmadd132pd (%edx), %xmm1, %xmm1 

// CHECK: vfmadd132pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x98,0x22]      
vfmadd132pd (%edx), %ymm4, %ymm4 

// CHECK: vfmadd132pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x98,0xc9]      
vfmadd132pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd132pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x98,0xe4]      
vfmadd132pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmadd132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x98,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x98,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x98,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x98,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd132ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x98,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd132ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x98,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmadd132ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x98,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096, %xmm1, %xmm1 

// CHECK: vfmadd132ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x98,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096, %ymm4, %ymm4 

// CHECK: vfmadd132ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x98,0x4c,0x02,0x40]      
vfmadd132ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd132ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x98,0x64,0x02,0x40]      
vfmadd132ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmadd132ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x98,0x0a]      
vfmadd132ps (%edx), %xmm1, %xmm1 

// CHECK: vfmadd132ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x98,0x22]      
vfmadd132ps (%edx), %ymm4, %ymm4 

// CHECK: vfmadd132ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x98,0xc9]      
vfmadd132ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd132ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x98,0xe4]      
vfmadd132ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd132sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd132sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd132sd 485498096, %xmm1, %xmm1 

// CHECK: vfmadd132sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x4c,0x02,0x40]      
vfmadd132sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd132sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x0a]      
vfmadd132sd (%edx), %xmm1, %xmm1 

// CHECK: vfmadd132sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0xc9]      
vfmadd132sd %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd132ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd132ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ss 485498096, %xmm1, %xmm1 

// CHECK: vfmadd132ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x4c,0x02,0x40]      
vfmadd132ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd132ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x0a]      
vfmadd132ss (%edx), %xmm1, %xmm1 

// CHECK: vfmadd132ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0xc9]      
vfmadd132ss %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd213pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa8,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd213pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa8,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmadd213pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa8,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096, %xmm1, %xmm1 

// CHECK: vfmadd213pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa8,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096, %ymm4, %ymm4 

// CHECK: vfmadd213pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa8,0x4c,0x02,0x40]      
vfmadd213pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd213pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa8,0x64,0x02,0x40]      
vfmadd213pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmadd213pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa8,0x0a]      
vfmadd213pd (%edx), %xmm1, %xmm1 

// CHECK: vfmadd213pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa8,0x22]      
vfmadd213pd (%edx), %ymm4, %ymm4 

// CHECK: vfmadd213pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa8,0xc9]      
vfmadd213pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd213pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa8,0xe4]      
vfmadd213pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmadd213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd213ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa8,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd213ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa8,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmadd213ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa8,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096, %xmm1, %xmm1 

// CHECK: vfmadd213ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa8,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096, %ymm4, %ymm4 

// CHECK: vfmadd213ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa8,0x4c,0x02,0x40]      
vfmadd213ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd213ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa8,0x64,0x02,0x40]      
vfmadd213ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmadd213ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa8,0x0a]      
vfmadd213ps (%edx), %xmm1, %xmm1 

// CHECK: vfmadd213ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa8,0x22]      
vfmadd213ps (%edx), %ymm4, %ymm4 

// CHECK: vfmadd213ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa8,0xc9]      
vfmadd213ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd213ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa8,0xe4]      
vfmadd213ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd213sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd213sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd213sd 485498096, %xmm1, %xmm1 

// CHECK: vfmadd213sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x4c,0x02,0x40]      
vfmadd213sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd213sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x0a]      
vfmadd213sd (%edx), %xmm1, %xmm1 

// CHECK: vfmadd213sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0xc9]      
vfmadd213sd %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd213ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd213ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ss 485498096, %xmm1, %xmm1 

// CHECK: vfmadd213ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x4c,0x02,0x40]      
vfmadd213ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd213ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x0a]      
vfmadd213ss (%edx), %xmm1, %xmm1 

// CHECK: vfmadd213ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0xc9]      
vfmadd213ss %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd231pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb8,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd231pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb8,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmadd231pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb8,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096, %xmm1, %xmm1 

// CHECK: vfmadd231pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb8,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096, %ymm4, %ymm4 

// CHECK: vfmadd231pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb8,0x4c,0x02,0x40]      
vfmadd231pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd231pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb8,0x64,0x02,0x40]      
vfmadd231pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmadd231pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb8,0x0a]      
vfmadd231pd (%edx), %xmm1, %xmm1 

// CHECK: vfmadd231pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb8,0x22]      
vfmadd231pd (%edx), %ymm4, %ymm4 

// CHECK: vfmadd231pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb8,0xc9]      
vfmadd231pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd231pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb8,0xe4]      
vfmadd231pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmadd231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmadd231ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb8,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd231ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb8,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmadd231ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb8,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096, %xmm1, %xmm1 

// CHECK: vfmadd231ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb8,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096, %ymm4, %ymm4 

// CHECK: vfmadd231ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb8,0x4c,0x02,0x40]      
vfmadd231ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd231ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb8,0x64,0x02,0x40]      
vfmadd231ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmadd231ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb8,0x0a]      
vfmadd231ps (%edx), %xmm1, %xmm1 

// CHECK: vfmadd231ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb8,0x22]      
vfmadd231ps (%edx), %ymm4, %ymm4 

// CHECK: vfmadd231ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb8,0xc9]      
vfmadd231ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd231ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb8,0xe4]      
vfmadd231ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd231sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd231sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd231sd 485498096, %xmm1, %xmm1 

// CHECK: vfmadd231sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x4c,0x02,0x40]      
vfmadd231sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd231sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x0a]      
vfmadd231sd (%edx), %xmm1, %xmm1 

// CHECK: vfmadd231sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0xc9]      
vfmadd231sd %xmm1, %xmm1, %xmm1 

// CHECK: vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmadd231ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmadd231ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ss 485498096, %xmm1, %xmm1 

// CHECK: vfmadd231ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x4c,0x02,0x40]      
vfmadd231ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmadd231ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x0a]      
vfmadd231ss (%edx), %xmm1, %xmm1 

// CHECK: vfmadd231ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0xc9]      
vfmadd231ss %xmm1, %xmm1, %xmm1 

// CHECK: vfmaddsub132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x96,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x96,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x96,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x96,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub132pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x96,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub132pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x96,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub132pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x96,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096, %xmm1, %xmm1 

// CHECK: vfmaddsub132pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x96,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096, %ymm4, %ymm4 

// CHECK: vfmaddsub132pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x96,0x4c,0x02,0x40]      
vfmaddsub132pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmaddsub132pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x96,0x64,0x02,0x40]      
vfmaddsub132pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmaddsub132pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x96,0x0a]      
vfmaddsub132pd (%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub132pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x96,0x22]      
vfmaddsub132pd (%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub132pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x96,0xc9]      
vfmaddsub132pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmaddsub132pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x96,0xe4]      
vfmaddsub132pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmaddsub132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x96,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x96,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x96,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x96,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub132ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x96,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub132ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x96,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub132ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x96,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096, %xmm1, %xmm1 

// CHECK: vfmaddsub132ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x96,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096, %ymm4, %ymm4 

// CHECK: vfmaddsub132ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x96,0x4c,0x02,0x40]      
vfmaddsub132ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmaddsub132ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x96,0x64,0x02,0x40]      
vfmaddsub132ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmaddsub132ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x96,0x0a]      
vfmaddsub132ps (%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub132ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x96,0x22]      
vfmaddsub132ps (%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub132ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x96,0xc9]      
vfmaddsub132ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmaddsub132ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x96,0xe4]      
vfmaddsub132ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmaddsub213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub213pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa6,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub213pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa6,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub213pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa6,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096, %xmm1, %xmm1 

// CHECK: vfmaddsub213pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa6,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096, %ymm4, %ymm4 

// CHECK: vfmaddsub213pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa6,0x4c,0x02,0x40]      
vfmaddsub213pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmaddsub213pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa6,0x64,0x02,0x40]      
vfmaddsub213pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmaddsub213pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa6,0x0a]      
vfmaddsub213pd (%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub213pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa6,0x22]      
vfmaddsub213pd (%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub213pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa6,0xc9]      
vfmaddsub213pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmaddsub213pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa6,0xe4]      
vfmaddsub213pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmaddsub213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub213ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa6,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub213ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa6,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub213ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa6,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096, %xmm1, %xmm1 

// CHECK: vfmaddsub213ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa6,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096, %ymm4, %ymm4 

// CHECK: vfmaddsub213ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa6,0x4c,0x02,0x40]      
vfmaddsub213ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmaddsub213ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa6,0x64,0x02,0x40]      
vfmaddsub213ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmaddsub213ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa6,0x0a]      
vfmaddsub213ps (%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub213ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa6,0x22]      
vfmaddsub213ps (%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub213ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa6,0xc9]      
vfmaddsub213ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmaddsub213ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa6,0xe4]      
vfmaddsub213ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmaddsub231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub231pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb6,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub231pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb6,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub231pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb6,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096, %xmm1, %xmm1 

// CHECK: vfmaddsub231pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb6,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096, %ymm4, %ymm4 

// CHECK: vfmaddsub231pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb6,0x4c,0x02,0x40]      
vfmaddsub231pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmaddsub231pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb6,0x64,0x02,0x40]      
vfmaddsub231pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmaddsub231pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb6,0x0a]      
vfmaddsub231pd (%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub231pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb6,0x22]      
vfmaddsub231pd (%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub231pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb6,0xc9]      
vfmaddsub231pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmaddsub231pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb6,0xe4]      
vfmaddsub231pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmaddsub231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb6,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb6,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmaddsub231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmaddsub231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmaddsub231ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb6,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub231ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb6,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub231ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb6,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096, %xmm1, %xmm1 

// CHECK: vfmaddsub231ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb6,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096, %ymm4, %ymm4 

// CHECK: vfmaddsub231ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb6,0x4c,0x02,0x40]      
vfmaddsub231ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmaddsub231ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb6,0x64,0x02,0x40]      
vfmaddsub231ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmaddsub231ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb6,0x0a]      
vfmaddsub231ps (%edx), %xmm1, %xmm1 

// CHECK: vfmaddsub231ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb6,0x22]      
vfmaddsub231ps (%edx), %ymm4, %ymm4 

// CHECK: vfmaddsub231ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb6,0xc9]      
vfmaddsub231ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmaddsub231ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb6,0xe4]      
vfmaddsub231ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmsub132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub132pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub132pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9a,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsub132pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096, %xmm1, %xmm1 

// CHECK: vfmsub132pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9a,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096, %ymm4, %ymm4 

// CHECK: vfmsub132pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9a,0x4c,0x02,0x40]      
vfmsub132pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub132pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9a,0x64,0x02,0x40]      
vfmsub132pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsub132pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9a,0x0a]      
vfmsub132pd (%edx), %xmm1, %xmm1 

// CHECK: vfmsub132pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9a,0x22]      
vfmsub132pd (%edx), %ymm4, %ymm4 

// CHECK: vfmsub132pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9a,0xc9]      
vfmsub132pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub132pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9a,0xe4]      
vfmsub132pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmsub132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub132ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9a,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub132ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9a,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsub132ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9a,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096, %xmm1, %xmm1 

// CHECK: vfmsub132ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9a,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096, %ymm4, %ymm4 

// CHECK: vfmsub132ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9a,0x4c,0x02,0x40]      
vfmsub132ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub132ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9a,0x64,0x02,0x40]      
vfmsub132ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsub132ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9a,0x0a]      
vfmsub132ps (%edx), %xmm1, %xmm1 

// CHECK: vfmsub132ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9a,0x22]      
vfmsub132ps (%edx), %ymm4, %ymm4 

// CHECK: vfmsub132ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9a,0xc9]      
vfmsub132ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub132ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9a,0xe4]      
vfmsub132ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub132sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub132sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub132sd 485498096, %xmm1, %xmm1 

// CHECK: vfmsub132sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x4c,0x02,0x40]      
vfmsub132sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub132sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x0a]      
vfmsub132sd (%edx), %xmm1, %xmm1 

// CHECK: vfmsub132sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0xc9]      
vfmsub132sd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub132ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub132ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ss 485498096, %xmm1, %xmm1 

// CHECK: vfmsub132ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x4c,0x02,0x40]      
vfmsub132ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub132ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x0a]      
vfmsub132ss (%edx), %xmm1, %xmm1 

// CHECK: vfmsub132ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0xc9]      
vfmsub132ss %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaa,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaa,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xaa,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xaa,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub213pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaa,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub213pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xaa,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsub213pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaa,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096, %xmm1, %xmm1 

// CHECK: vfmsub213pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xaa,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096, %ymm4, %ymm4 

// CHECK: vfmsub213pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaa,0x4c,0x02,0x40]      
vfmsub213pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub213pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xaa,0x64,0x02,0x40]      
vfmsub213pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsub213pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaa,0x0a]      
vfmsub213pd (%edx), %xmm1, %xmm1 

// CHECK: vfmsub213pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xaa,0x22]      
vfmsub213pd (%edx), %ymm4, %ymm4 

// CHECK: vfmsub213pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaa,0xc9]      
vfmsub213pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub213pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xaa,0xe4]      
vfmsub213pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmsub213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaa,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaa,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xaa,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xaa,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub213ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaa,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub213ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xaa,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsub213ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaa,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096, %xmm1, %xmm1 

// CHECK: vfmsub213ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xaa,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096, %ymm4, %ymm4 

// CHECK: vfmsub213ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaa,0x4c,0x02,0x40]      
vfmsub213ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub213ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xaa,0x64,0x02,0x40]      
vfmsub213ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsub213ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaa,0x0a]      
vfmsub213ps (%edx), %xmm1, %xmm1 

// CHECK: vfmsub213ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xaa,0x22]      
vfmsub213ps (%edx), %ymm4, %ymm4 

// CHECK: vfmsub213ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaa,0xc9]      
vfmsub213ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub213ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xaa,0xe4]      
vfmsub213ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub213sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub213sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub213sd 485498096, %xmm1, %xmm1 

// CHECK: vfmsub213sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x4c,0x02,0x40]      
vfmsub213sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub213sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x0a]      
vfmsub213sd (%edx), %xmm1, %xmm1 

// CHECK: vfmsub213sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0xc9]      
vfmsub213sd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub213ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub213ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ss 485498096, %xmm1, %xmm1 

// CHECK: vfmsub213ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x4c,0x02,0x40]      
vfmsub213ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub213ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x0a]      
vfmsub213ss (%edx), %xmm1, %xmm1 

// CHECK: vfmsub213ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0xc9]      
vfmsub213ss %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xba,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xba,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xba,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xba,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub231pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xba,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub231pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xba,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsub231pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xba,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096, %xmm1, %xmm1 

// CHECK: vfmsub231pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xba,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096, %ymm4, %ymm4 

// CHECK: vfmsub231pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xba,0x4c,0x02,0x40]      
vfmsub231pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub231pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xba,0x64,0x02,0x40]      
vfmsub231pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsub231pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xba,0x0a]      
vfmsub231pd (%edx), %xmm1, %xmm1 

// CHECK: vfmsub231pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xba,0x22]      
vfmsub231pd (%edx), %ymm4, %ymm4 

// CHECK: vfmsub231pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xba,0xc9]      
vfmsub231pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub231pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xba,0xe4]      
vfmsub231pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmsub231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xba,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xba,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xba,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xba,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsub231ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xba,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub231ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xba,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsub231ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xba,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096, %xmm1, %xmm1 

// CHECK: vfmsub231ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xba,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096, %ymm4, %ymm4 

// CHECK: vfmsub231ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xba,0x4c,0x02,0x40]      
vfmsub231ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub231ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xba,0x64,0x02,0x40]      
vfmsub231ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsub231ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xba,0x0a]      
vfmsub231ps (%edx), %xmm1, %xmm1 

// CHECK: vfmsub231ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xba,0x22]      
vfmsub231ps (%edx), %ymm4, %ymm4 

// CHECK: vfmsub231ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xba,0xc9]      
vfmsub231ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub231ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xba,0xe4]      
vfmsub231ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub231sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub231sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub231sd 485498096, %xmm1, %xmm1 

// CHECK: vfmsub231sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x4c,0x02,0x40]      
vfmsub231sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub231sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x0a]      
vfmsub231sd (%edx), %xmm1, %xmm1 

// CHECK: vfmsub231sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0xc9]      
vfmsub231sd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsub231ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsub231ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ss 485498096, %xmm1, %xmm1 

// CHECK: vfmsub231ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x4c,0x02,0x40]      
vfmsub231ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsub231ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x0a]      
vfmsub231ss (%edx), %xmm1, %xmm1 

// CHECK: vfmsub231ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0xc9]      
vfmsub231ss %xmm1, %xmm1, %xmm1 

// CHECK: vfmsubadd132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x97,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x97,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x97,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x97,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd132pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x97,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd132pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x97,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd132pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x97,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096, %xmm1, %xmm1 

// CHECK: vfmsubadd132pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x97,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096, %ymm4, %ymm4 

// CHECK: vfmsubadd132pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x97,0x4c,0x02,0x40]      
vfmsubadd132pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsubadd132pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x97,0x64,0x02,0x40]      
vfmsubadd132pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsubadd132pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x97,0x0a]      
vfmsubadd132pd (%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd132pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x97,0x22]      
vfmsubadd132pd (%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd132pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x97,0xc9]      
vfmsubadd132pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsubadd132pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x97,0xe4]      
vfmsubadd132pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmsubadd132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x97,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x97,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x97,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x97,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd132ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x97,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd132ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x97,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd132ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x97,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096, %xmm1, %xmm1 

// CHECK: vfmsubadd132ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x97,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096, %ymm4, %ymm4 

// CHECK: vfmsubadd132ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x97,0x4c,0x02,0x40]      
vfmsubadd132ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsubadd132ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x97,0x64,0x02,0x40]      
vfmsubadd132ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsubadd132ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x97,0x0a]      
vfmsubadd132ps (%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd132ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x97,0x22]      
vfmsubadd132ps (%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd132ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x97,0xc9]      
vfmsubadd132ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmsubadd132ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x97,0xe4]      
vfmsubadd132ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmsubadd213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa7,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa7,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd213pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa7,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd213pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa7,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd213pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa7,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096, %xmm1, %xmm1 

// CHECK: vfmsubadd213pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa7,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096, %ymm4, %ymm4 

// CHECK: vfmsubadd213pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa7,0x4c,0x02,0x40]      
vfmsubadd213pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsubadd213pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa7,0x64,0x02,0x40]      
vfmsubadd213pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsubadd213pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa7,0x0a]      
vfmsubadd213pd (%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd213pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa7,0x22]      
vfmsubadd213pd (%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd213pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa7,0xc9]      
vfmsubadd213pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsubadd213pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xa7,0xe4]      
vfmsubadd213pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmsubadd213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa7,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa7,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd213ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa7,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd213ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa7,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd213ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa7,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096, %xmm1, %xmm1 

// CHECK: vfmsubadd213ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa7,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096, %ymm4, %ymm4 

// CHECK: vfmsubadd213ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa7,0x4c,0x02,0x40]      
vfmsubadd213ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsubadd213ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa7,0x64,0x02,0x40]      
vfmsubadd213ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsubadd213ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa7,0x0a]      
vfmsubadd213ps (%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd213ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa7,0x22]      
vfmsubadd213ps (%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd213ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xa7,0xc9]      
vfmsubadd213ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmsubadd213ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xa7,0xe4]      
vfmsubadd213ps %ymm4, %ymm4, %ymm4 

// CHECK: vfmsubadd231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb7,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb7,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd231pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb7,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd231pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb7,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd231pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb7,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096, %xmm1, %xmm1 

// CHECK: vfmsubadd231pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb7,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096, %ymm4, %ymm4 

// CHECK: vfmsubadd231pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb7,0x4c,0x02,0x40]      
vfmsubadd231pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsubadd231pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb7,0x64,0x02,0x40]      
vfmsubadd231pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsubadd231pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb7,0x0a]      
vfmsubadd231pd (%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd231pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb7,0x22]      
vfmsubadd231pd (%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd231pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb7,0xc9]      
vfmsubadd231pd %xmm1, %xmm1, %xmm1 

// CHECK: vfmsubadd231pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xb7,0xe4]      
vfmsubadd231pd %ymm4, %ymm4, %ymm4 

// CHECK: vfmsubadd231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb7,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb7,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfmsubadd231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfmsubadd231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfmsubadd231ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb7,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd231ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb7,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd231ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb7,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096, %xmm1, %xmm1 

// CHECK: vfmsubadd231ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb7,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096, %ymm4, %ymm4 

// CHECK: vfmsubadd231ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb7,0x4c,0x02,0x40]      
vfmsubadd231ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfmsubadd231ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb7,0x64,0x02,0x40]      
vfmsubadd231ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfmsubadd231ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb7,0x0a]      
vfmsubadd231ps (%edx), %xmm1, %xmm1 

// CHECK: vfmsubadd231ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb7,0x22]      
vfmsubadd231ps (%edx), %ymm4, %ymm4 

// CHECK: vfmsubadd231ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xb7,0xc9]      
vfmsubadd231ps %xmm1, %xmm1, %xmm1 

// CHECK: vfmsubadd231ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xb7,0xe4]      
vfmsubadd231ps %ymm4, %ymm4, %ymm4 

// CHECK: vfnmadd132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd132pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmadd132pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd132pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096, %ymm4, %ymm4 

// CHECK: vfnmadd132pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9c,0x4c,0x02,0x40]      
vfnmadd132pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd132pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9c,0x64,0x02,0x40]      
vfnmadd132pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmadd132pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9c,0x0a]      
vfnmadd132pd (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9c,0x22]      
vfnmadd132pd (%edx), %ymm4, %ymm4 

// CHECK: vfnmadd132pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9c,0xc9]      
vfnmadd132pd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd132pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9c,0xe4]      
vfnmadd132pd %ymm4, %ymm4, %ymm4 

// CHECK: vfnmadd132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd132ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmadd132ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd132ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096, %ymm4, %ymm4 

// CHECK: vfnmadd132ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9c,0x4c,0x02,0x40]      
vfnmadd132ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd132ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9c,0x64,0x02,0x40]      
vfnmadd132ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmadd132ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9c,0x0a]      
vfnmadd132ps (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9c,0x22]      
vfnmadd132ps (%edx), %ymm4, %ymm4 

// CHECK: vfnmadd132ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9c,0xc9]      
vfnmadd132ps %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd132ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9c,0xe4]      
vfnmadd132ps %ymm4, %ymm4, %ymm4 

// CHECK: vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132sd 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd132sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x4c,0x02,0x40]      
vfnmadd132sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd132sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x0a]      
vfnmadd132sd (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0xc9]      
vfnmadd132sd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd132ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ss 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd132ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x4c,0x02,0x40]      
vfnmadd132ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd132ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x0a]      
vfnmadd132ss (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd132ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0xc9]      
vfnmadd132ss %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xac,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xac,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xac,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xac,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd213pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xac,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xac,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmadd213pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xac,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd213pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xac,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096, %ymm4, %ymm4 

// CHECK: vfnmadd213pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xac,0x4c,0x02,0x40]      
vfnmadd213pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd213pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xac,0x64,0x02,0x40]      
vfnmadd213pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmadd213pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xac,0x0a]      
vfnmadd213pd (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xac,0x22]      
vfnmadd213pd (%edx), %ymm4, %ymm4 

// CHECK: vfnmadd213pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xac,0xc9]      
vfnmadd213pd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd213pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xac,0xe4]      
vfnmadd213pd %ymm4, %ymm4, %ymm4 

// CHECK: vfnmadd213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xac,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xac,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xac,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xac,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd213ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xac,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xac,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmadd213ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xac,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd213ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xac,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096, %ymm4, %ymm4 

// CHECK: vfnmadd213ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xac,0x4c,0x02,0x40]      
vfnmadd213ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd213ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xac,0x64,0x02,0x40]      
vfnmadd213ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmadd213ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xac,0x0a]      
vfnmadd213ps (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xac,0x22]      
vfnmadd213ps (%edx), %ymm4, %ymm4 

// CHECK: vfnmadd213ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xac,0xc9]      
vfnmadd213ps %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd213ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xac,0xe4]      
vfnmadd213ps %ymm4, %ymm4, %ymm4 

// CHECK: vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213sd 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd213sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x4c,0x02,0x40]      
vfnmadd213sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd213sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x0a]      
vfnmadd213sd (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0xc9]      
vfnmadd213sd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd213ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ss 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd213ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x4c,0x02,0x40]      
vfnmadd213ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd213ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x0a]      
vfnmadd213ss (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd213ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0xc9]      
vfnmadd213ss %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbc,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbc,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbc,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbc,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd231pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbc,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbc,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmadd231pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbc,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd231pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbc,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096, %ymm4, %ymm4 

// CHECK: vfnmadd231pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbc,0x4c,0x02,0x40]      
vfnmadd231pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd231pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbc,0x64,0x02,0x40]      
vfnmadd231pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmadd231pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbc,0x0a]      
vfnmadd231pd (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbc,0x22]      
vfnmadd231pd (%edx), %ymm4, %ymm4 

// CHECK: vfnmadd231pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbc,0xc9]      
vfnmadd231pd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd231pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbc,0xe4]      
vfnmadd231pd %ymm4, %ymm4, %ymm4 

// CHECK: vfnmadd231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbc,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbc,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbc,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbc,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmadd231ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbc,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbc,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmadd231ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbc,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd231ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbc,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096, %ymm4, %ymm4 

// CHECK: vfnmadd231ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbc,0x4c,0x02,0x40]      
vfnmadd231ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd231ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbc,0x64,0x02,0x40]      
vfnmadd231ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmadd231ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbc,0x0a]      
vfnmadd231ps (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbc,0x22]      
vfnmadd231ps (%edx), %ymm4, %ymm4 

// CHECK: vfnmadd231ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbc,0xc9]      
vfnmadd231ps %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd231ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbc,0xe4]      
vfnmadd231ps %ymm4, %ymm4, %ymm4 

// CHECK: vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231sd 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd231sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x4c,0x02,0x40]      
vfnmadd231sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd231sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x0a]      
vfnmadd231sd (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0xc9]      
vfnmadd231sd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmadd231ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ss 485498096, %xmm1, %xmm1 

// CHECK: vfnmadd231ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x4c,0x02,0x40]      
vfnmadd231ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmadd231ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x0a]      
vfnmadd231ss (%edx), %xmm1, %xmm1 

// CHECK: vfnmadd231ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0xc9]      
vfnmadd231ss %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub132pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub132pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub132pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmsub132pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub132pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9e,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096, %ymm4, %ymm4 

// CHECK: vfnmsub132pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9e,0x4c,0x02,0x40]      
vfnmsub132pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub132pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9e,0x64,0x02,0x40]      
vfnmsub132pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmsub132pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9e,0x0a]      
vfnmsub132pd (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9e,0x22]      
vfnmsub132pd (%edx), %ymm4, %ymm4 

// CHECK: vfnmsub132pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9e,0xc9]      
vfnmsub132pd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub132pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x9e,0xe4]      
vfnmsub132pd %ymm4, %ymm4, %ymm4 

// CHECK: vfnmsub132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub132ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub132ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub132ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmsub132ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub132ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9e,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096, %ymm4, %ymm4 

// CHECK: vfnmsub132ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9e,0x4c,0x02,0x40]      
vfnmsub132ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub132ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9e,0x64,0x02,0x40]      
vfnmsub132ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmsub132ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9e,0x0a]      
vfnmsub132ps (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9e,0x22]      
vfnmsub132ps (%edx), %ymm4, %ymm4 

// CHECK: vfnmsub132ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9e,0xc9]      
vfnmsub132ps %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub132ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x9e,0xe4]      
vfnmsub132ps %ymm4, %ymm4, %ymm4 

// CHECK: vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132sd 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub132sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x4c,0x02,0x40]      
vfnmsub132sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub132sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x0a]      
vfnmsub132sd (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0xc9]      
vfnmsub132sd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub132ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ss 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub132ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x4c,0x02,0x40]      
vfnmsub132ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub132ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x0a]      
vfnmsub132ss (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub132ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0xc9]      
vfnmsub132ss %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xae,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub213pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xae,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xae,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub213pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xae,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub213pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xae,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xae,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmsub213pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xae,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub213pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xae,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096, %ymm4, %ymm4 

// CHECK: vfnmsub213pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xae,0x4c,0x02,0x40]      
vfnmsub213pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub213pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xae,0x64,0x02,0x40]      
vfnmsub213pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmsub213pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xae,0x0a]      
vfnmsub213pd (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xae,0x22]      
vfnmsub213pd (%edx), %ymm4, %ymm4 

// CHECK: vfnmsub213pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xae,0xc9]      
vfnmsub213pd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub213pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xae,0xe4]      
vfnmsub213pd %ymm4, %ymm4, %ymm4 

// CHECK: vfnmsub213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xae,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub213ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xae,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xae,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub213ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xae,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub213ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xae,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xae,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmsub213ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xae,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub213ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xae,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096, %ymm4, %ymm4 

// CHECK: vfnmsub213ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xae,0x4c,0x02,0x40]      
vfnmsub213ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub213ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xae,0x64,0x02,0x40]      
vfnmsub213ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmsub213ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xae,0x0a]      
vfnmsub213ps (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xae,0x22]      
vfnmsub213ps (%edx), %ymm4, %ymm4 

// CHECK: vfnmsub213ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xae,0xc9]      
vfnmsub213ps %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub213ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xae,0xe4]      
vfnmsub213ps %ymm4, %ymm4, %ymm4 

// CHECK: vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213sd 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub213sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x4c,0x02,0x40]      
vfnmsub213sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub213sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x0a]      
vfnmsub213sd (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0xc9]      
vfnmsub213sd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub213ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ss 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub213ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x4c,0x02,0x40]      
vfnmsub213ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub213ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x0a]      
vfnmsub213ss (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub213ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0xc9]      
vfnmsub213ss %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbe,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub231pd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbe,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbe,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub231pd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbe,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub231pd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbe,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231pd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbe,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmsub231pd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbe,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub231pd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbe,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096, %ymm4, %ymm4 

// CHECK: vfnmsub231pd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbe,0x4c,0x02,0x40]      
vfnmsub231pd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub231pd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbe,0x64,0x02,0x40]      
vfnmsub231pd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmsub231pd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbe,0x0a]      
vfnmsub231pd (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231pd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbe,0x22]      
vfnmsub231pd (%edx), %ymm4, %ymm4 

// CHECK: vfnmsub231pd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbe,0xc9]      
vfnmsub231pd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub231pd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0xbe,0xe4]      
vfnmsub231pd %ymm4, %ymm4, %ymm4 

// CHECK: vfnmsub231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbe,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub231ps -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbe,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbe,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub231ps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbe,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vfnmsub231ps 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbe,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231ps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbe,0xa2,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vfnmsub231ps 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbe,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub231ps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbe,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096, %ymm4, %ymm4 

// CHECK: vfnmsub231ps 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbe,0x4c,0x02,0x40]      
vfnmsub231ps 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub231ps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbe,0x64,0x02,0x40]      
vfnmsub231ps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vfnmsub231ps (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbe,0x0a]      
vfnmsub231ps (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231ps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbe,0x22]      
vfnmsub231ps (%edx), %ymm4, %ymm4 

// CHECK: vfnmsub231ps %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbe,0xc9]      
vfnmsub231ps %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub231ps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0xbe,0xe4]      
vfnmsub231ps %ymm4, %ymm4, %ymm4 

// CHECK: vfnmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231sd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbf,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231sd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231sd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbf,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231sd 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub231sd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbf,0x4c,0x02,0x40]      
vfnmsub231sd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub231sd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbf,0x0a]      
vfnmsub231sd (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231sd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbf,0xc9]      
vfnmsub231sd %xmm1, %xmm1, %xmm1 

// CHECK: vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vfnmsub231ss 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x8a,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ss 485498096(%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231ss 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x0d,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ss 485498096, %xmm1, %xmm1 

// CHECK: vfnmsub231ss 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x4c,0x02,0x40]      
vfnmsub231ss 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vfnmsub231ss (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x0a]      
vfnmsub231ss (%edx), %xmm1, %xmm1 

// CHECK: vfnmsub231ss %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0xc9]      
vfnmsub231ss %xmm1, %xmm1, %xmm1 

