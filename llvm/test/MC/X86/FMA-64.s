// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vfmadd132pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x98,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096, %xmm15, %xmm15 

// CHECK: vfmadd132pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x98,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096, %xmm6, %xmm6 

// CHECK: vfmadd132pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x98,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096, %ymm7, %ymm7 

// CHECK: vfmadd132pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x98,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132pd 485498096, %ymm9, %ymm9 

// CHECK: vfmadd132pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x98,0x7c,0x82,0xc0]      
vfmadd132pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x98,0x7c,0x82,0x40]      
vfmadd132pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x98,0x74,0x82,0xc0]      
vfmadd132pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x98,0x74,0x82,0x40]      
vfmadd132pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x98,0x7c,0x82,0xc0]      
vfmadd132pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd132pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x98,0x7c,0x82,0x40]      
vfmadd132pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd132pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x98,0x4c,0x82,0xc0]      
vfmadd132pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd132pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x98,0x4c,0x82,0x40]      
vfmadd132pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd132pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x98,0x7c,0x02,0x40]      
vfmadd132pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd132pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x98,0x74,0x02,0x40]      
vfmadd132pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd132pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x98,0x7c,0x02,0x40]      
vfmadd132pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmadd132pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x98,0x4c,0x02,0x40]      
vfmadd132pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmadd132pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x98,0x7a,0x40]      
vfmadd132pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x98,0x72,0x40]      
vfmadd132pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x98,0x7a,0x40]      
vfmadd132pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmadd132pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x98,0x4a,0x40]      
vfmadd132pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmadd132pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x98,0x3a]      
vfmadd132pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x98,0x32]      
vfmadd132pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x98,0x3a]      
vfmadd132pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmadd132pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x98,0x0a]      
vfmadd132pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmadd132pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x98,0xff]      
vfmadd132pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd132pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x98,0xf6]      
vfmadd132pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd132pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x98,0xff]      
vfmadd132pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmadd132pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x98,0xc9]      
vfmadd132pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmadd132ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x98,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096, %xmm15, %xmm15 

// CHECK: vfmadd132ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x98,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096, %xmm6, %xmm6 

// CHECK: vfmadd132ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x98,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096, %ymm7, %ymm7 

// CHECK: vfmadd132ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x98,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ps 485498096, %ymm9, %ymm9 

// CHECK: vfmadd132ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x98,0x7c,0x82,0xc0]      
vfmadd132ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x98,0x7c,0x82,0x40]      
vfmadd132ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x98,0x74,0x82,0xc0]      
vfmadd132ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x98,0x74,0x82,0x40]      
vfmadd132ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x98,0x7c,0x82,0xc0]      
vfmadd132ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd132ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x98,0x7c,0x82,0x40]      
vfmadd132ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd132ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x98,0x4c,0x82,0xc0]      
vfmadd132ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd132ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x98,0x4c,0x82,0x40]      
vfmadd132ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd132ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x98,0x7c,0x02,0x40]      
vfmadd132ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd132ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x98,0x74,0x02,0x40]      
vfmadd132ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd132ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x98,0x7c,0x02,0x40]      
vfmadd132ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmadd132ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x98,0x4c,0x02,0x40]      
vfmadd132ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmadd132ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x98,0x7a,0x40]      
vfmadd132ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x98,0x72,0x40]      
vfmadd132ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x98,0x7a,0x40]      
vfmadd132ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmadd132ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x98,0x4a,0x40]      
vfmadd132ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmadd132ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x98,0x3a]      
vfmadd132ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x98,0x32]      
vfmadd132ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x98,0x3a]      
vfmadd132ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmadd132ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x98,0x0a]      
vfmadd132ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmadd132ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x98,0xff]      
vfmadd132ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd132ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x98,0xf6]      
vfmadd132ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd132ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x98,0xff]      
vfmadd132ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmadd132ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x98,0xc9]      
vfmadd132ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmadd132sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132sd 485498096, %xmm15, %xmm15 

// CHECK: vfmadd132sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x99,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132sd 485498096, %xmm6, %xmm6 

// CHECK: vfmadd132sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x7c,0x82,0xc0]      
vfmadd132sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x7c,0x82,0x40]      
vfmadd132sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x99,0x74,0x82,0xc0]      
vfmadd132sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x99,0x74,0x82,0x40]      
vfmadd132sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x7c,0x02,0x40]      
vfmadd132sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd132sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x99,0x74,0x02,0x40]      
vfmadd132sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd132sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x7a,0x40]      
vfmadd132sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x99,0x72,0x40]      
vfmadd132sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x3a]      
vfmadd132sd (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x99,0x32]      
vfmadd132sd (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x99,0xff]      
vfmadd132sd %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd132sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x99,0xf6]      
vfmadd132sd %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd132ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ss 485498096, %xmm15, %xmm15 

// CHECK: vfmadd132ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x99,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd132ss 485498096, %xmm6, %xmm6 

// CHECK: vfmadd132ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x7c,0x82,0xc0]      
vfmadd132ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x7c,0x82,0x40]      
vfmadd132ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd132ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x99,0x74,0x82,0xc0]      
vfmadd132ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x99,0x74,0x82,0x40]      
vfmadd132ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd132ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x7c,0x02,0x40]      
vfmadd132ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd132ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x99,0x74,0x02,0x40]      
vfmadd132ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd132ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x7a,0x40]      
vfmadd132ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x99,0x72,0x40]      
vfmadd132ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x3a]      
vfmadd132ss (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd132ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x99,0x32]      
vfmadd132ss (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd132ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x99,0xff]      
vfmadd132ss %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd132ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x99,0xf6]      
vfmadd132ss %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd213pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096, %xmm15, %xmm15 

// CHECK: vfmadd213pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096, %xmm6, %xmm6 

// CHECK: vfmadd213pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096, %ymm7, %ymm7 

// CHECK: vfmadd213pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213pd 485498096, %ymm9, %ymm9 

// CHECK: vfmadd213pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa8,0x7c,0x82,0xc0]      
vfmadd213pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa8,0x7c,0x82,0x40]      
vfmadd213pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa8,0x74,0x82,0xc0]      
vfmadd213pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa8,0x74,0x82,0x40]      
vfmadd213pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa8,0x7c,0x82,0xc0]      
vfmadd213pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd213pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa8,0x7c,0x82,0x40]      
vfmadd213pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd213pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa8,0x4c,0x82,0xc0]      
vfmadd213pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd213pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa8,0x4c,0x82,0x40]      
vfmadd213pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd213pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa8,0x7c,0x02,0x40]      
vfmadd213pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd213pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa8,0x74,0x02,0x40]      
vfmadd213pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd213pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa8,0x7c,0x02,0x40]      
vfmadd213pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmadd213pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa8,0x4c,0x02,0x40]      
vfmadd213pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmadd213pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa8,0x7a,0x40]      
vfmadd213pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa8,0x72,0x40]      
vfmadd213pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa8,0x7a,0x40]      
vfmadd213pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmadd213pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa8,0x4a,0x40]      
vfmadd213pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmadd213pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa8,0x3a]      
vfmadd213pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa8,0x32]      
vfmadd213pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa8,0x3a]      
vfmadd213pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmadd213pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa8,0x0a]      
vfmadd213pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmadd213pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xa8,0xff]      
vfmadd213pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd213pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa8,0xf6]      
vfmadd213pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd213pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa8,0xff]      
vfmadd213pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmadd213pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xa8,0xc9]      
vfmadd213pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmadd213ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096, %xmm15, %xmm15 

// CHECK: vfmadd213ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096, %xmm6, %xmm6 

// CHECK: vfmadd213ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096, %ymm7, %ymm7 

// CHECK: vfmadd213ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ps 485498096, %ymm9, %ymm9 

// CHECK: vfmadd213ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa8,0x7c,0x82,0xc0]      
vfmadd213ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa8,0x7c,0x82,0x40]      
vfmadd213ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa8,0x74,0x82,0xc0]      
vfmadd213ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa8,0x74,0x82,0x40]      
vfmadd213ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa8,0x7c,0x82,0xc0]      
vfmadd213ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd213ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa8,0x7c,0x82,0x40]      
vfmadd213ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd213ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa8,0x4c,0x82,0xc0]      
vfmadd213ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd213ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa8,0x4c,0x82,0x40]      
vfmadd213ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd213ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa8,0x7c,0x02,0x40]      
vfmadd213ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd213ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa8,0x74,0x02,0x40]      
vfmadd213ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd213ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa8,0x7c,0x02,0x40]      
vfmadd213ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmadd213ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa8,0x4c,0x02,0x40]      
vfmadd213ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmadd213ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa8,0x7a,0x40]      
vfmadd213ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa8,0x72,0x40]      
vfmadd213ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa8,0x7a,0x40]      
vfmadd213ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmadd213ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa8,0x4a,0x40]      
vfmadd213ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmadd213ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa8,0x3a]      
vfmadd213ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa8,0x32]      
vfmadd213ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa8,0x3a]      
vfmadd213ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmadd213ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa8,0x0a]      
vfmadd213ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmadd213ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xa8,0xff]      
vfmadd213ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd213ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa8,0xf6]      
vfmadd213ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd213ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa8,0xff]      
vfmadd213ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmadd213ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xa8,0xc9]      
vfmadd213ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmadd213sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213sd 485498096, %xmm15, %xmm15 

// CHECK: vfmadd213sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213sd 485498096, %xmm6, %xmm6 

// CHECK: vfmadd213sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x7c,0x82,0xc0]      
vfmadd213sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x7c,0x82,0x40]      
vfmadd213sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa9,0x74,0x82,0xc0]      
vfmadd213sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa9,0x74,0x82,0x40]      
vfmadd213sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x7c,0x02,0x40]      
vfmadd213sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd213sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa9,0x74,0x02,0x40]      
vfmadd213sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd213sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x7a,0x40]      
vfmadd213sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa9,0x72,0x40]      
vfmadd213sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x3a]      
vfmadd213sd (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa9,0x32]      
vfmadd213sd (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xa9,0xff]      
vfmadd213sd %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd213sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa9,0xf6]      
vfmadd213sd %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd213ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ss 485498096, %xmm15, %xmm15 

// CHECK: vfmadd213ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd213ss 485498096, %xmm6, %xmm6 

// CHECK: vfmadd213ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x7c,0x82,0xc0]      
vfmadd213ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x7c,0x82,0x40]      
vfmadd213ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd213ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa9,0x74,0x82,0xc0]      
vfmadd213ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa9,0x74,0x82,0x40]      
vfmadd213ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd213ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x7c,0x02,0x40]      
vfmadd213ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd213ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa9,0x74,0x02,0x40]      
vfmadd213ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd213ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x7a,0x40]      
vfmadd213ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa9,0x72,0x40]      
vfmadd213ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x3a]      
vfmadd213ss (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd213ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa9,0x32]      
vfmadd213ss (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd213ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xa9,0xff]      
vfmadd213ss %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd213ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa9,0xf6]      
vfmadd213ss %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd231pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096, %xmm15, %xmm15 

// CHECK: vfmadd231pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096, %xmm6, %xmm6 

// CHECK: vfmadd231pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096, %ymm7, %ymm7 

// CHECK: vfmadd231pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231pd 485498096, %ymm9, %ymm9 

// CHECK: vfmadd231pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb8,0x7c,0x82,0xc0]      
vfmadd231pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb8,0x7c,0x82,0x40]      
vfmadd231pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb8,0x74,0x82,0xc0]      
vfmadd231pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb8,0x74,0x82,0x40]      
vfmadd231pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb8,0x7c,0x82,0xc0]      
vfmadd231pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd231pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb8,0x7c,0x82,0x40]      
vfmadd231pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd231pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb8,0x4c,0x82,0xc0]      
vfmadd231pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd231pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb8,0x4c,0x82,0x40]      
vfmadd231pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd231pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb8,0x7c,0x02,0x40]      
vfmadd231pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd231pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb8,0x74,0x02,0x40]      
vfmadd231pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd231pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb8,0x7c,0x02,0x40]      
vfmadd231pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmadd231pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb8,0x4c,0x02,0x40]      
vfmadd231pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmadd231pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb8,0x7a,0x40]      
vfmadd231pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb8,0x72,0x40]      
vfmadd231pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb8,0x7a,0x40]      
vfmadd231pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmadd231pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb8,0x4a,0x40]      
vfmadd231pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmadd231pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb8,0x3a]      
vfmadd231pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb8,0x32]      
vfmadd231pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb8,0x3a]      
vfmadd231pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmadd231pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb8,0x0a]      
vfmadd231pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmadd231pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xb8,0xff]      
vfmadd231pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd231pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb8,0xf6]      
vfmadd231pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd231pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb8,0xff]      
vfmadd231pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmadd231pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xb8,0xc9]      
vfmadd231pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmadd231ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096, %xmm15, %xmm15 

// CHECK: vfmadd231ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096, %xmm6, %xmm6 

// CHECK: vfmadd231ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096, %ymm7, %ymm7 

// CHECK: vfmadd231ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ps 485498096, %ymm9, %ymm9 

// CHECK: vfmadd231ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb8,0x7c,0x82,0xc0]      
vfmadd231ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb8,0x7c,0x82,0x40]      
vfmadd231ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb8,0x74,0x82,0xc0]      
vfmadd231ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb8,0x74,0x82,0x40]      
vfmadd231ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb8,0x7c,0x82,0xc0]      
vfmadd231ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd231ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb8,0x7c,0x82,0x40]      
vfmadd231ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmadd231ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb8,0x4c,0x82,0xc0]      
vfmadd231ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd231ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb8,0x4c,0x82,0x40]      
vfmadd231ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmadd231ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb8,0x7c,0x02,0x40]      
vfmadd231ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd231ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb8,0x74,0x02,0x40]      
vfmadd231ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd231ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb8,0x7c,0x02,0x40]      
vfmadd231ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmadd231ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb8,0x4c,0x02,0x40]      
vfmadd231ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmadd231ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb8,0x7a,0x40]      
vfmadd231ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb8,0x72,0x40]      
vfmadd231ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb8,0x7a,0x40]      
vfmadd231ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmadd231ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb8,0x4a,0x40]      
vfmadd231ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmadd231ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb8,0x3a]      
vfmadd231ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb8,0x32]      
vfmadd231ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb8,0x3a]      
vfmadd231ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmadd231ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb8,0x0a]      
vfmadd231ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmadd231ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xb8,0xff]      
vfmadd231ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd231ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb8,0xf6]      
vfmadd231ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd231ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb8,0xff]      
vfmadd231ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmadd231ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xb8,0xc9]      
vfmadd231ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmadd231sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231sd 485498096, %xmm15, %xmm15 

// CHECK: vfmadd231sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231sd 485498096, %xmm6, %xmm6 

// CHECK: vfmadd231sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x7c,0x82,0xc0]      
vfmadd231sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x7c,0x82,0x40]      
vfmadd231sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb9,0x74,0x82,0xc0]      
vfmadd231sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb9,0x74,0x82,0x40]      
vfmadd231sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x7c,0x02,0x40]      
vfmadd231sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd231sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb9,0x74,0x02,0x40]      
vfmadd231sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd231sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x7a,0x40]      
vfmadd231sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb9,0x72,0x40]      
vfmadd231sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x3a]      
vfmadd231sd (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb9,0x32]      
vfmadd231sd (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xb9,0xff]      
vfmadd231sd %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd231sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb9,0xf6]      
vfmadd231sd %xmm6, %xmm6, %xmm6 

// CHECK: vfmadd231ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ss 485498096, %xmm15, %xmm15 

// CHECK: vfmadd231ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmadd231ss 485498096, %xmm6, %xmm6 

// CHECK: vfmadd231ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x7c,0x82,0xc0]      
vfmadd231ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x7c,0x82,0x40]      
vfmadd231ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmadd231ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb9,0x74,0x82,0xc0]      
vfmadd231ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb9,0x74,0x82,0x40]      
vfmadd231ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmadd231ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x7c,0x02,0x40]      
vfmadd231ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmadd231ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb9,0x74,0x02,0x40]      
vfmadd231ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmadd231ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x7a,0x40]      
vfmadd231ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb9,0x72,0x40]      
vfmadd231ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x3a]      
vfmadd231ss (%rdx), %xmm15, %xmm15 

// CHECK: vfmadd231ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb9,0x32]      
vfmadd231ss (%rdx), %xmm6, %xmm6 

// CHECK: vfmadd231ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xb9,0xff]      
vfmadd231ss %xmm15, %xmm15, %xmm15 

// CHECK: vfmadd231ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb9,0xf6]      
vfmadd231ss %xmm6, %xmm6, %xmm6 

// CHECK: vfmaddsub132pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x96,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096, %xmm15, %xmm15 

// CHECK: vfmaddsub132pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x96,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096, %xmm6, %xmm6 

// CHECK: vfmaddsub132pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x96,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096, %ymm7, %ymm7 

// CHECK: vfmaddsub132pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x96,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132pd 485498096, %ymm9, %ymm9 

// CHECK: vfmaddsub132pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x96,0x7c,0x82,0xc0]      
vfmaddsub132pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub132pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x96,0x7c,0x82,0x40]      
vfmaddsub132pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub132pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x96,0x74,0x82,0xc0]      
vfmaddsub132pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub132pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x96,0x74,0x82,0x40]      
vfmaddsub132pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub132pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x96,0x7c,0x82,0xc0]      
vfmaddsub132pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub132pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x96,0x7c,0x82,0x40]      
vfmaddsub132pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub132pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x96,0x4c,0x82,0xc0]      
vfmaddsub132pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub132pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x96,0x4c,0x82,0x40]      
vfmaddsub132pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub132pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x96,0x7c,0x02,0x40]      
vfmaddsub132pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmaddsub132pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x96,0x74,0x02,0x40]      
vfmaddsub132pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmaddsub132pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x96,0x7c,0x02,0x40]      
vfmaddsub132pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmaddsub132pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x96,0x4c,0x02,0x40]      
vfmaddsub132pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmaddsub132pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x96,0x7a,0x40]      
vfmaddsub132pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub132pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x96,0x72,0x40]      
vfmaddsub132pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub132pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x96,0x7a,0x40]      
vfmaddsub132pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub132pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x96,0x4a,0x40]      
vfmaddsub132pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub132pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x96,0x3a]      
vfmaddsub132pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub132pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x96,0x32]      
vfmaddsub132pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub132pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x96,0x3a]      
vfmaddsub132pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub132pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x96,0x0a]      
vfmaddsub132pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub132pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x96,0xff]      
vfmaddsub132pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmaddsub132pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x96,0xf6]      
vfmaddsub132pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmaddsub132pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x96,0xff]      
vfmaddsub132pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmaddsub132pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x96,0xc9]      
vfmaddsub132pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmaddsub132ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x96,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096, %xmm15, %xmm15 

// CHECK: vfmaddsub132ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x96,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096, %xmm6, %xmm6 

// CHECK: vfmaddsub132ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x96,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096, %ymm7, %ymm7 

// CHECK: vfmaddsub132ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x96,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub132ps 485498096, %ymm9, %ymm9 

// CHECK: vfmaddsub132ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x96,0x7c,0x82,0xc0]      
vfmaddsub132ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub132ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x96,0x7c,0x82,0x40]      
vfmaddsub132ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub132ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x96,0x74,0x82,0xc0]      
vfmaddsub132ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub132ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x96,0x74,0x82,0x40]      
vfmaddsub132ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub132ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x96,0x7c,0x82,0xc0]      
vfmaddsub132ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub132ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x96,0x7c,0x82,0x40]      
vfmaddsub132ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub132ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x96,0x4c,0x82,0xc0]      
vfmaddsub132ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub132ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x96,0x4c,0x82,0x40]      
vfmaddsub132ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub132ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x96,0x7c,0x02,0x40]      
vfmaddsub132ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmaddsub132ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x96,0x74,0x02,0x40]      
vfmaddsub132ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmaddsub132ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x96,0x7c,0x02,0x40]      
vfmaddsub132ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmaddsub132ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x96,0x4c,0x02,0x40]      
vfmaddsub132ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmaddsub132ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x96,0x7a,0x40]      
vfmaddsub132ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub132ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x96,0x72,0x40]      
vfmaddsub132ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub132ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x96,0x7a,0x40]      
vfmaddsub132ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub132ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x96,0x4a,0x40]      
vfmaddsub132ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub132ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x96,0x3a]      
vfmaddsub132ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub132ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x96,0x32]      
vfmaddsub132ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub132ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x96,0x3a]      
vfmaddsub132ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub132ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x96,0x0a]      
vfmaddsub132ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub132ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x96,0xff]      
vfmaddsub132ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmaddsub132ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x96,0xf6]      
vfmaddsub132ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmaddsub132ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x96,0xff]      
vfmaddsub132ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmaddsub132ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x96,0xc9]      
vfmaddsub132ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmaddsub213pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096, %xmm15, %xmm15 

// CHECK: vfmaddsub213pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096, %xmm6, %xmm6 

// CHECK: vfmaddsub213pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096, %ymm7, %ymm7 

// CHECK: vfmaddsub213pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213pd 485498096, %ymm9, %ymm9 

// CHECK: vfmaddsub213pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa6,0x7c,0x82,0xc0]      
vfmaddsub213pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub213pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa6,0x7c,0x82,0x40]      
vfmaddsub213pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub213pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa6,0x74,0x82,0xc0]      
vfmaddsub213pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub213pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa6,0x74,0x82,0x40]      
vfmaddsub213pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub213pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa6,0x7c,0x82,0xc0]      
vfmaddsub213pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub213pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa6,0x7c,0x82,0x40]      
vfmaddsub213pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub213pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa6,0x4c,0x82,0xc0]      
vfmaddsub213pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub213pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa6,0x4c,0x82,0x40]      
vfmaddsub213pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub213pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa6,0x7c,0x02,0x40]      
vfmaddsub213pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmaddsub213pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa6,0x74,0x02,0x40]      
vfmaddsub213pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmaddsub213pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa6,0x7c,0x02,0x40]      
vfmaddsub213pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmaddsub213pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa6,0x4c,0x02,0x40]      
vfmaddsub213pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmaddsub213pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa6,0x7a,0x40]      
vfmaddsub213pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub213pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa6,0x72,0x40]      
vfmaddsub213pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub213pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa6,0x7a,0x40]      
vfmaddsub213pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub213pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa6,0x4a,0x40]      
vfmaddsub213pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub213pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa6,0x3a]      
vfmaddsub213pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub213pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa6,0x32]      
vfmaddsub213pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub213pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa6,0x3a]      
vfmaddsub213pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub213pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa6,0x0a]      
vfmaddsub213pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub213pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xa6,0xff]      
vfmaddsub213pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmaddsub213pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa6,0xf6]      
vfmaddsub213pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmaddsub213pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa6,0xff]      
vfmaddsub213pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmaddsub213pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xa6,0xc9]      
vfmaddsub213pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmaddsub213ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096, %xmm15, %xmm15 

// CHECK: vfmaddsub213ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096, %xmm6, %xmm6 

// CHECK: vfmaddsub213ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096, %ymm7, %ymm7 

// CHECK: vfmaddsub213ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub213ps 485498096, %ymm9, %ymm9 

// CHECK: vfmaddsub213ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa6,0x7c,0x82,0xc0]      
vfmaddsub213ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub213ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa6,0x7c,0x82,0x40]      
vfmaddsub213ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub213ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa6,0x74,0x82,0xc0]      
vfmaddsub213ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub213ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa6,0x74,0x82,0x40]      
vfmaddsub213ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub213ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa6,0x7c,0x82,0xc0]      
vfmaddsub213ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub213ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa6,0x7c,0x82,0x40]      
vfmaddsub213ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub213ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa6,0x4c,0x82,0xc0]      
vfmaddsub213ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub213ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa6,0x4c,0x82,0x40]      
vfmaddsub213ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub213ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa6,0x7c,0x02,0x40]      
vfmaddsub213ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmaddsub213ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa6,0x74,0x02,0x40]      
vfmaddsub213ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmaddsub213ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa6,0x7c,0x02,0x40]      
vfmaddsub213ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmaddsub213ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa6,0x4c,0x02,0x40]      
vfmaddsub213ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmaddsub213ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa6,0x7a,0x40]      
vfmaddsub213ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub213ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa6,0x72,0x40]      
vfmaddsub213ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub213ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa6,0x7a,0x40]      
vfmaddsub213ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub213ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa6,0x4a,0x40]      
vfmaddsub213ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub213ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa6,0x3a]      
vfmaddsub213ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub213ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa6,0x32]      
vfmaddsub213ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub213ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa6,0x3a]      
vfmaddsub213ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub213ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa6,0x0a]      
vfmaddsub213ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub213ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xa6,0xff]      
vfmaddsub213ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmaddsub213ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa6,0xf6]      
vfmaddsub213ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmaddsub213ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa6,0xff]      
vfmaddsub213ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmaddsub213ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xa6,0xc9]      
vfmaddsub213ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmaddsub231pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096, %xmm15, %xmm15 

// CHECK: vfmaddsub231pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096, %xmm6, %xmm6 

// CHECK: vfmaddsub231pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096, %ymm7, %ymm7 

// CHECK: vfmaddsub231pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231pd 485498096, %ymm9, %ymm9 

// CHECK: vfmaddsub231pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb6,0x7c,0x82,0xc0]      
vfmaddsub231pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub231pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb6,0x7c,0x82,0x40]      
vfmaddsub231pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub231pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb6,0x74,0x82,0xc0]      
vfmaddsub231pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub231pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb6,0x74,0x82,0x40]      
vfmaddsub231pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub231pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb6,0x7c,0x82,0xc0]      
vfmaddsub231pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub231pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb6,0x7c,0x82,0x40]      
vfmaddsub231pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub231pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb6,0x4c,0x82,0xc0]      
vfmaddsub231pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub231pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb6,0x4c,0x82,0x40]      
vfmaddsub231pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub231pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb6,0x7c,0x02,0x40]      
vfmaddsub231pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmaddsub231pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb6,0x74,0x02,0x40]      
vfmaddsub231pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmaddsub231pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb6,0x7c,0x02,0x40]      
vfmaddsub231pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmaddsub231pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb6,0x4c,0x02,0x40]      
vfmaddsub231pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmaddsub231pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb6,0x7a,0x40]      
vfmaddsub231pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub231pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb6,0x72,0x40]      
vfmaddsub231pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub231pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb6,0x7a,0x40]      
vfmaddsub231pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub231pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb6,0x4a,0x40]      
vfmaddsub231pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub231pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb6,0x3a]      
vfmaddsub231pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub231pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb6,0x32]      
vfmaddsub231pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub231pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb6,0x3a]      
vfmaddsub231pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub231pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb6,0x0a]      
vfmaddsub231pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub231pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xb6,0xff]      
vfmaddsub231pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmaddsub231pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb6,0xf6]      
vfmaddsub231pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmaddsub231pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb6,0xff]      
vfmaddsub231pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmaddsub231pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xb6,0xc9]      
vfmaddsub231pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmaddsub231ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096, %xmm15, %xmm15 

// CHECK: vfmaddsub231ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096, %xmm6, %xmm6 

// CHECK: vfmaddsub231ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096, %ymm7, %ymm7 

// CHECK: vfmaddsub231ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmaddsub231ps 485498096, %ymm9, %ymm9 

// CHECK: vfmaddsub231ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb6,0x7c,0x82,0xc0]      
vfmaddsub231ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub231ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb6,0x7c,0x82,0x40]      
vfmaddsub231ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmaddsub231ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb6,0x74,0x82,0xc0]      
vfmaddsub231ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub231ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb6,0x74,0x82,0x40]      
vfmaddsub231ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmaddsub231ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb6,0x7c,0x82,0xc0]      
vfmaddsub231ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub231ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb6,0x7c,0x82,0x40]      
vfmaddsub231ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmaddsub231ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb6,0x4c,0x82,0xc0]      
vfmaddsub231ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub231ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb6,0x4c,0x82,0x40]      
vfmaddsub231ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmaddsub231ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb6,0x7c,0x02,0x40]      
vfmaddsub231ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmaddsub231ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb6,0x74,0x02,0x40]      
vfmaddsub231ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmaddsub231ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb6,0x7c,0x02,0x40]      
vfmaddsub231ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmaddsub231ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb6,0x4c,0x02,0x40]      
vfmaddsub231ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmaddsub231ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb6,0x7a,0x40]      
vfmaddsub231ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub231ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb6,0x72,0x40]      
vfmaddsub231ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub231ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb6,0x7a,0x40]      
vfmaddsub231ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub231ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb6,0x4a,0x40]      
vfmaddsub231ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub231ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb6,0x3a]      
vfmaddsub231ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmaddsub231ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb6,0x32]      
vfmaddsub231ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmaddsub231ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb6,0x3a]      
vfmaddsub231ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmaddsub231ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb6,0x0a]      
vfmaddsub231ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmaddsub231ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xb6,0xff]      
vfmaddsub231ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmaddsub231ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb6,0xf6]      
vfmaddsub231ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmaddsub231ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb6,0xff]      
vfmaddsub231ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmaddsub231ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xb6,0xc9]      
vfmaddsub231ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmsub132pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096, %xmm15, %xmm15 

// CHECK: vfmsub132pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096, %xmm6, %xmm6 

// CHECK: vfmsub132pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096, %ymm7, %ymm7 

// CHECK: vfmsub132pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132pd 485498096, %ymm9, %ymm9 

// CHECK: vfmsub132pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9a,0x7c,0x82,0xc0]      
vfmsub132pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9a,0x7c,0x82,0x40]      
vfmsub132pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9a,0x74,0x82,0xc0]      
vfmsub132pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9a,0x74,0x82,0x40]      
vfmsub132pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9a,0x7c,0x82,0xc0]      
vfmsub132pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub132pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9a,0x7c,0x82,0x40]      
vfmsub132pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub132pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9a,0x4c,0x82,0xc0]      
vfmsub132pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub132pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9a,0x4c,0x82,0x40]      
vfmsub132pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub132pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9a,0x7c,0x02,0x40]      
vfmsub132pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub132pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9a,0x74,0x02,0x40]      
vfmsub132pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub132pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9a,0x7c,0x02,0x40]      
vfmsub132pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsub132pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9a,0x4c,0x02,0x40]      
vfmsub132pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsub132pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9a,0x7a,0x40]      
vfmsub132pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9a,0x72,0x40]      
vfmsub132pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9a,0x7a,0x40]      
vfmsub132pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsub132pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9a,0x4a,0x40]      
vfmsub132pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsub132pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9a,0x3a]      
vfmsub132pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9a,0x32]      
vfmsub132pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9a,0x3a]      
vfmsub132pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmsub132pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9a,0x0a]      
vfmsub132pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmsub132pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x9a,0xff]      
vfmsub132pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub132pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9a,0xf6]      
vfmsub132pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub132pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9a,0xff]      
vfmsub132pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmsub132pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x9a,0xc9]      
vfmsub132pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmsub132ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096, %xmm15, %xmm15 

// CHECK: vfmsub132ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096, %xmm6, %xmm6 

// CHECK: vfmsub132ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096, %ymm7, %ymm7 

// CHECK: vfmsub132ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ps 485498096, %ymm9, %ymm9 

// CHECK: vfmsub132ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9a,0x7c,0x82,0xc0]      
vfmsub132ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9a,0x7c,0x82,0x40]      
vfmsub132ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9a,0x74,0x82,0xc0]      
vfmsub132ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9a,0x74,0x82,0x40]      
vfmsub132ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9a,0x7c,0x82,0xc0]      
vfmsub132ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub132ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9a,0x7c,0x82,0x40]      
vfmsub132ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub132ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9a,0x4c,0x82,0xc0]      
vfmsub132ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub132ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9a,0x4c,0x82,0x40]      
vfmsub132ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub132ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9a,0x7c,0x02,0x40]      
vfmsub132ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub132ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9a,0x74,0x02,0x40]      
vfmsub132ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub132ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9a,0x7c,0x02,0x40]      
vfmsub132ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsub132ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9a,0x4c,0x02,0x40]      
vfmsub132ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsub132ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9a,0x7a,0x40]      
vfmsub132ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9a,0x72,0x40]      
vfmsub132ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9a,0x7a,0x40]      
vfmsub132ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsub132ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9a,0x4a,0x40]      
vfmsub132ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsub132ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9a,0x3a]      
vfmsub132ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9a,0x32]      
vfmsub132ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9a,0x3a]      
vfmsub132ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmsub132ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9a,0x0a]      
vfmsub132ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmsub132ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x9a,0xff]      
vfmsub132ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub132ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9a,0xf6]      
vfmsub132ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub132ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9a,0xff]      
vfmsub132ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmsub132ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x9a,0xc9]      
vfmsub132ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmsub132sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132sd 485498096, %xmm15, %xmm15 

// CHECK: vfmsub132sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132sd 485498096, %xmm6, %xmm6 

// CHECK: vfmsub132sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x7c,0x82,0xc0]      
vfmsub132sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x7c,0x82,0x40]      
vfmsub132sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9b,0x74,0x82,0xc0]      
vfmsub132sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9b,0x74,0x82,0x40]      
vfmsub132sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x7c,0x02,0x40]      
vfmsub132sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub132sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9b,0x74,0x02,0x40]      
vfmsub132sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub132sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x7a,0x40]      
vfmsub132sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9b,0x72,0x40]      
vfmsub132sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x3a]      
vfmsub132sd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9b,0x32]      
vfmsub132sd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x9b,0xff]      
vfmsub132sd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub132sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9b,0xf6]      
vfmsub132sd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub132ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ss 485498096, %xmm15, %xmm15 

// CHECK: vfmsub132ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub132ss 485498096, %xmm6, %xmm6 

// CHECK: vfmsub132ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x7c,0x82,0xc0]      
vfmsub132ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x7c,0x82,0x40]      
vfmsub132ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub132ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9b,0x74,0x82,0xc0]      
vfmsub132ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9b,0x74,0x82,0x40]      
vfmsub132ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub132ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x7c,0x02,0x40]      
vfmsub132ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub132ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9b,0x74,0x02,0x40]      
vfmsub132ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub132ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x7a,0x40]      
vfmsub132ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9b,0x72,0x40]      
vfmsub132ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x3a]      
vfmsub132ss (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub132ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9b,0x32]      
vfmsub132ss (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub132ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x9b,0xff]      
vfmsub132ss %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub132ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9b,0xf6]      
vfmsub132ss %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub213pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaa,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096, %xmm15, %xmm15 

// CHECK: vfmsub213pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaa,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096, %xmm6, %xmm6 

// CHECK: vfmsub213pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xaa,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096, %ymm7, %ymm7 

// CHECK: vfmsub213pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xaa,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213pd 485498096, %ymm9, %ymm9 

// CHECK: vfmsub213pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaa,0x7c,0x82,0xc0]      
vfmsub213pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaa,0x7c,0x82,0x40]      
vfmsub213pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaa,0x74,0x82,0xc0]      
vfmsub213pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaa,0x74,0x82,0x40]      
vfmsub213pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xaa,0x7c,0x82,0xc0]      
vfmsub213pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub213pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xaa,0x7c,0x82,0x40]      
vfmsub213pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub213pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xaa,0x4c,0x82,0xc0]      
vfmsub213pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub213pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xaa,0x4c,0x82,0x40]      
vfmsub213pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub213pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaa,0x7c,0x02,0x40]      
vfmsub213pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub213pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaa,0x74,0x02,0x40]      
vfmsub213pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub213pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xaa,0x7c,0x02,0x40]      
vfmsub213pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsub213pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xaa,0x4c,0x02,0x40]      
vfmsub213pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsub213pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaa,0x7a,0x40]      
vfmsub213pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaa,0x72,0x40]      
vfmsub213pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xaa,0x7a,0x40]      
vfmsub213pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsub213pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xaa,0x4a,0x40]      
vfmsub213pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsub213pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaa,0x3a]      
vfmsub213pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaa,0x32]      
vfmsub213pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xaa,0x3a]      
vfmsub213pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmsub213pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xaa,0x0a]      
vfmsub213pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmsub213pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xaa,0xff]      
vfmsub213pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub213pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaa,0xf6]      
vfmsub213pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub213pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xaa,0xff]      
vfmsub213pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmsub213pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xaa,0xc9]      
vfmsub213pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmsub213ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaa,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096, %xmm15, %xmm15 

// CHECK: vfmsub213ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaa,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096, %xmm6, %xmm6 

// CHECK: vfmsub213ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xaa,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096, %ymm7, %ymm7 

// CHECK: vfmsub213ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xaa,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ps 485498096, %ymm9, %ymm9 

// CHECK: vfmsub213ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaa,0x7c,0x82,0xc0]      
vfmsub213ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaa,0x7c,0x82,0x40]      
vfmsub213ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaa,0x74,0x82,0xc0]      
vfmsub213ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaa,0x74,0x82,0x40]      
vfmsub213ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xaa,0x7c,0x82,0xc0]      
vfmsub213ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub213ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xaa,0x7c,0x82,0x40]      
vfmsub213ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub213ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xaa,0x4c,0x82,0xc0]      
vfmsub213ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub213ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xaa,0x4c,0x82,0x40]      
vfmsub213ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub213ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaa,0x7c,0x02,0x40]      
vfmsub213ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub213ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaa,0x74,0x02,0x40]      
vfmsub213ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub213ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xaa,0x7c,0x02,0x40]      
vfmsub213ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsub213ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xaa,0x4c,0x02,0x40]      
vfmsub213ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsub213ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaa,0x7a,0x40]      
vfmsub213ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaa,0x72,0x40]      
vfmsub213ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xaa,0x7a,0x40]      
vfmsub213ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsub213ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xaa,0x4a,0x40]      
vfmsub213ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsub213ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaa,0x3a]      
vfmsub213ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaa,0x32]      
vfmsub213ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xaa,0x3a]      
vfmsub213ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmsub213ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xaa,0x0a]      
vfmsub213ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmsub213ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xaa,0xff]      
vfmsub213ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub213ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaa,0xf6]      
vfmsub213ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub213ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xaa,0xff]      
vfmsub213ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmsub213ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xaa,0xc9]      
vfmsub213ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmsub213sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213sd 485498096, %xmm15, %xmm15 

// CHECK: vfmsub213sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xab,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213sd 485498096, %xmm6, %xmm6 

// CHECK: vfmsub213sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x7c,0x82,0xc0]      
vfmsub213sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x7c,0x82,0x40]      
vfmsub213sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xab,0x74,0x82,0xc0]      
vfmsub213sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xab,0x74,0x82,0x40]      
vfmsub213sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x7c,0x02,0x40]      
vfmsub213sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub213sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xab,0x74,0x02,0x40]      
vfmsub213sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub213sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x7a,0x40]      
vfmsub213sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xab,0x72,0x40]      
vfmsub213sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x3a]      
vfmsub213sd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xab,0x32]      
vfmsub213sd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xab,0xff]      
vfmsub213sd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub213sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xab,0xf6]      
vfmsub213sd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub213ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ss 485498096, %xmm15, %xmm15 

// CHECK: vfmsub213ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xab,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub213ss 485498096, %xmm6, %xmm6 

// CHECK: vfmsub213ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x7c,0x82,0xc0]      
vfmsub213ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x7c,0x82,0x40]      
vfmsub213ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub213ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xab,0x74,0x82,0xc0]      
vfmsub213ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xab,0x74,0x82,0x40]      
vfmsub213ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub213ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x7c,0x02,0x40]      
vfmsub213ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub213ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xab,0x74,0x02,0x40]      
vfmsub213ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub213ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x7a,0x40]      
vfmsub213ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xab,0x72,0x40]      
vfmsub213ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x3a]      
vfmsub213ss (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub213ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xab,0x32]      
vfmsub213ss (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub213ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xab,0xff]      
vfmsub213ss %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub213ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xab,0xf6]      
vfmsub213ss %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub231pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xba,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096, %xmm15, %xmm15 

// CHECK: vfmsub231pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xba,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096, %xmm6, %xmm6 

// CHECK: vfmsub231pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xba,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096, %ymm7, %ymm7 

// CHECK: vfmsub231pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xba,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231pd 485498096, %ymm9, %ymm9 

// CHECK: vfmsub231pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xba,0x7c,0x82,0xc0]      
vfmsub231pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xba,0x7c,0x82,0x40]      
vfmsub231pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xba,0x74,0x82,0xc0]      
vfmsub231pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xba,0x74,0x82,0x40]      
vfmsub231pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xba,0x7c,0x82,0xc0]      
vfmsub231pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub231pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xba,0x7c,0x82,0x40]      
vfmsub231pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub231pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xba,0x4c,0x82,0xc0]      
vfmsub231pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub231pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xba,0x4c,0x82,0x40]      
vfmsub231pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub231pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xba,0x7c,0x02,0x40]      
vfmsub231pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub231pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xba,0x74,0x02,0x40]      
vfmsub231pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub231pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xba,0x7c,0x02,0x40]      
vfmsub231pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsub231pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xba,0x4c,0x02,0x40]      
vfmsub231pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsub231pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xba,0x7a,0x40]      
vfmsub231pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xba,0x72,0x40]      
vfmsub231pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xba,0x7a,0x40]      
vfmsub231pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsub231pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xba,0x4a,0x40]      
vfmsub231pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsub231pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xba,0x3a]      
vfmsub231pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xba,0x32]      
vfmsub231pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xba,0x3a]      
vfmsub231pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmsub231pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xba,0x0a]      
vfmsub231pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmsub231pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xba,0xff]      
vfmsub231pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub231pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xba,0xf6]      
vfmsub231pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub231pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xba,0xff]      
vfmsub231pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmsub231pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xba,0xc9]      
vfmsub231pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmsub231ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xba,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096, %xmm15, %xmm15 

// CHECK: vfmsub231ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xba,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096, %xmm6, %xmm6 

// CHECK: vfmsub231ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xba,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096, %ymm7, %ymm7 

// CHECK: vfmsub231ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xba,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ps 485498096, %ymm9, %ymm9 

// CHECK: vfmsub231ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xba,0x7c,0x82,0xc0]      
vfmsub231ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xba,0x7c,0x82,0x40]      
vfmsub231ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xba,0x74,0x82,0xc0]      
vfmsub231ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xba,0x74,0x82,0x40]      
vfmsub231ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xba,0x7c,0x82,0xc0]      
vfmsub231ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub231ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xba,0x7c,0x82,0x40]      
vfmsub231ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsub231ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xba,0x4c,0x82,0xc0]      
vfmsub231ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub231ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xba,0x4c,0x82,0x40]      
vfmsub231ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsub231ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xba,0x7c,0x02,0x40]      
vfmsub231ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub231ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xba,0x74,0x02,0x40]      
vfmsub231ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub231ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xba,0x7c,0x02,0x40]      
vfmsub231ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsub231ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xba,0x4c,0x02,0x40]      
vfmsub231ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsub231ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xba,0x7a,0x40]      
vfmsub231ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xba,0x72,0x40]      
vfmsub231ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xba,0x7a,0x40]      
vfmsub231ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsub231ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xba,0x4a,0x40]      
vfmsub231ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsub231ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xba,0x3a]      
vfmsub231ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xba,0x32]      
vfmsub231ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xba,0x3a]      
vfmsub231ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmsub231ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xba,0x0a]      
vfmsub231ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmsub231ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xba,0xff]      
vfmsub231ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub231ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xba,0xf6]      
vfmsub231ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub231ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xba,0xff]      
vfmsub231ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmsub231ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xba,0xc9]      
vfmsub231ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmsub231sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231sd 485498096, %xmm15, %xmm15 

// CHECK: vfmsub231sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbb,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231sd 485498096, %xmm6, %xmm6 

// CHECK: vfmsub231sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x7c,0x82,0xc0]      
vfmsub231sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x7c,0x82,0x40]      
vfmsub231sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbb,0x74,0x82,0xc0]      
vfmsub231sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbb,0x74,0x82,0x40]      
vfmsub231sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x7c,0x02,0x40]      
vfmsub231sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub231sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbb,0x74,0x02,0x40]      
vfmsub231sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub231sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x7a,0x40]      
vfmsub231sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbb,0x72,0x40]      
vfmsub231sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x3a]      
vfmsub231sd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbb,0x32]      
vfmsub231sd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xbb,0xff]      
vfmsub231sd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub231sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbb,0xf6]      
vfmsub231sd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsub231ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ss 485498096, %xmm15, %xmm15 

// CHECK: vfmsub231ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbb,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsub231ss 485498096, %xmm6, %xmm6 

// CHECK: vfmsub231ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x7c,0x82,0xc0]      
vfmsub231ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x7c,0x82,0x40]      
vfmsub231ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsub231ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbb,0x74,0x82,0xc0]      
vfmsub231ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbb,0x74,0x82,0x40]      
vfmsub231ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsub231ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x7c,0x02,0x40]      
vfmsub231ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsub231ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbb,0x74,0x02,0x40]      
vfmsub231ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsub231ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x7a,0x40]      
vfmsub231ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbb,0x72,0x40]      
vfmsub231ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x3a]      
vfmsub231ss (%rdx), %xmm15, %xmm15 

// CHECK: vfmsub231ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbb,0x32]      
vfmsub231ss (%rdx), %xmm6, %xmm6 

// CHECK: vfmsub231ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xbb,0xff]      
vfmsub231ss %xmm15, %xmm15, %xmm15 

// CHECK: vfmsub231ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbb,0xf6]      
vfmsub231ss %xmm6, %xmm6, %xmm6 

// CHECK: vfmsubadd132pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x97,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096, %xmm15, %xmm15 

// CHECK: vfmsubadd132pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x97,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096, %xmm6, %xmm6 

// CHECK: vfmsubadd132pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x97,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096, %ymm7, %ymm7 

// CHECK: vfmsubadd132pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x97,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132pd 485498096, %ymm9, %ymm9 

// CHECK: vfmsubadd132pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x97,0x7c,0x82,0xc0]      
vfmsubadd132pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd132pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x97,0x7c,0x82,0x40]      
vfmsubadd132pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd132pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x97,0x74,0x82,0xc0]      
vfmsubadd132pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd132pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x97,0x74,0x82,0x40]      
vfmsubadd132pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd132pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x97,0x7c,0x82,0xc0]      
vfmsubadd132pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd132pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x97,0x7c,0x82,0x40]      
vfmsubadd132pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd132pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x97,0x4c,0x82,0xc0]      
vfmsubadd132pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd132pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x97,0x4c,0x82,0x40]      
vfmsubadd132pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd132pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x97,0x7c,0x02,0x40]      
vfmsubadd132pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsubadd132pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x97,0x74,0x02,0x40]      
vfmsubadd132pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsubadd132pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x97,0x7c,0x02,0x40]      
vfmsubadd132pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsubadd132pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x97,0x4c,0x02,0x40]      
vfmsubadd132pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsubadd132pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x97,0x7a,0x40]      
vfmsubadd132pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd132pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x97,0x72,0x40]      
vfmsubadd132pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd132pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x97,0x7a,0x40]      
vfmsubadd132pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd132pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x97,0x4a,0x40]      
vfmsubadd132pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd132pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x97,0x3a]      
vfmsubadd132pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd132pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x97,0x32]      
vfmsubadd132pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd132pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x97,0x3a]      
vfmsubadd132pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd132pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x97,0x0a]      
vfmsubadd132pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd132pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x97,0xff]      
vfmsubadd132pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsubadd132pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x97,0xf6]      
vfmsubadd132pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsubadd132pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x97,0xff]      
vfmsubadd132pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmsubadd132pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x97,0xc9]      
vfmsubadd132pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmsubadd132ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x97,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096, %xmm15, %xmm15 

// CHECK: vfmsubadd132ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x97,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096, %xmm6, %xmm6 

// CHECK: vfmsubadd132ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x97,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096, %ymm7, %ymm7 

// CHECK: vfmsubadd132ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x97,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd132ps 485498096, %ymm9, %ymm9 

// CHECK: vfmsubadd132ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x97,0x7c,0x82,0xc0]      
vfmsubadd132ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd132ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x97,0x7c,0x82,0x40]      
vfmsubadd132ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd132ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x97,0x74,0x82,0xc0]      
vfmsubadd132ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd132ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x97,0x74,0x82,0x40]      
vfmsubadd132ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd132ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x97,0x7c,0x82,0xc0]      
vfmsubadd132ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd132ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x97,0x7c,0x82,0x40]      
vfmsubadd132ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd132ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x97,0x4c,0x82,0xc0]      
vfmsubadd132ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd132ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x97,0x4c,0x82,0x40]      
vfmsubadd132ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd132ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x97,0x7c,0x02,0x40]      
vfmsubadd132ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsubadd132ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x97,0x74,0x02,0x40]      
vfmsubadd132ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsubadd132ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x97,0x7c,0x02,0x40]      
vfmsubadd132ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsubadd132ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x97,0x4c,0x02,0x40]      
vfmsubadd132ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsubadd132ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x97,0x7a,0x40]      
vfmsubadd132ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd132ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x97,0x72,0x40]      
vfmsubadd132ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd132ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x97,0x7a,0x40]      
vfmsubadd132ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd132ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x97,0x4a,0x40]      
vfmsubadd132ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd132ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x97,0x3a]      
vfmsubadd132ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd132ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x97,0x32]      
vfmsubadd132ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd132ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x97,0x3a]      
vfmsubadd132ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd132ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x97,0x0a]      
vfmsubadd132ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd132ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x97,0xff]      
vfmsubadd132ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmsubadd132ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x97,0xf6]      
vfmsubadd132ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmsubadd132ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x97,0xff]      
vfmsubadd132ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmsubadd132ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x97,0xc9]      
vfmsubadd132ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmsubadd213pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096, %xmm15, %xmm15 

// CHECK: vfmsubadd213pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096, %xmm6, %xmm6 

// CHECK: vfmsubadd213pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096, %ymm7, %ymm7 

// CHECK: vfmsubadd213pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa7,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213pd 485498096, %ymm9, %ymm9 

// CHECK: vfmsubadd213pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa7,0x7c,0x82,0xc0]      
vfmsubadd213pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd213pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa7,0x7c,0x82,0x40]      
vfmsubadd213pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd213pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa7,0x74,0x82,0xc0]      
vfmsubadd213pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd213pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa7,0x74,0x82,0x40]      
vfmsubadd213pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd213pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa7,0x7c,0x82,0xc0]      
vfmsubadd213pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd213pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa7,0x7c,0x82,0x40]      
vfmsubadd213pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd213pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa7,0x4c,0x82,0xc0]      
vfmsubadd213pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd213pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa7,0x4c,0x82,0x40]      
vfmsubadd213pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd213pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa7,0x7c,0x02,0x40]      
vfmsubadd213pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsubadd213pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa7,0x74,0x02,0x40]      
vfmsubadd213pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsubadd213pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa7,0x7c,0x02,0x40]      
vfmsubadd213pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsubadd213pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa7,0x4c,0x02,0x40]      
vfmsubadd213pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsubadd213pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa7,0x7a,0x40]      
vfmsubadd213pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd213pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa7,0x72,0x40]      
vfmsubadd213pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd213pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa7,0x7a,0x40]      
vfmsubadd213pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd213pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa7,0x4a,0x40]      
vfmsubadd213pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd213pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xa7,0x3a]      
vfmsubadd213pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd213pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa7,0x32]      
vfmsubadd213pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd213pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa7,0x3a]      
vfmsubadd213pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd213pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xa7,0x0a]      
vfmsubadd213pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd213pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xa7,0xff]      
vfmsubadd213pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsubadd213pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xa7,0xf6]      
vfmsubadd213pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsubadd213pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xa7,0xff]      
vfmsubadd213pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmsubadd213pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xa7,0xc9]      
vfmsubadd213pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmsubadd213ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096, %xmm15, %xmm15 

// CHECK: vfmsubadd213ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096, %xmm6, %xmm6 

// CHECK: vfmsubadd213ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096, %ymm7, %ymm7 

// CHECK: vfmsubadd213ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa7,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd213ps 485498096, %ymm9, %ymm9 

// CHECK: vfmsubadd213ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa7,0x7c,0x82,0xc0]      
vfmsubadd213ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd213ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa7,0x7c,0x82,0x40]      
vfmsubadd213ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd213ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa7,0x74,0x82,0xc0]      
vfmsubadd213ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd213ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa7,0x74,0x82,0x40]      
vfmsubadd213ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd213ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa7,0x7c,0x82,0xc0]      
vfmsubadd213ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd213ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa7,0x7c,0x82,0x40]      
vfmsubadd213ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd213ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa7,0x4c,0x82,0xc0]      
vfmsubadd213ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd213ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa7,0x4c,0x82,0x40]      
vfmsubadd213ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd213ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa7,0x7c,0x02,0x40]      
vfmsubadd213ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsubadd213ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa7,0x74,0x02,0x40]      
vfmsubadd213ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsubadd213ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa7,0x7c,0x02,0x40]      
vfmsubadd213ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsubadd213ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa7,0x4c,0x02,0x40]      
vfmsubadd213ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsubadd213ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa7,0x7a,0x40]      
vfmsubadd213ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd213ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa7,0x72,0x40]      
vfmsubadd213ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd213ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa7,0x7a,0x40]      
vfmsubadd213ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd213ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa7,0x4a,0x40]      
vfmsubadd213ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd213ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xa7,0x3a]      
vfmsubadd213ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd213ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa7,0x32]      
vfmsubadd213ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd213ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa7,0x3a]      
vfmsubadd213ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd213ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xa7,0x0a]      
vfmsubadd213ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd213ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xa7,0xff]      
vfmsubadd213ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmsubadd213ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xa7,0xf6]      
vfmsubadd213ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmsubadd213ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xa7,0xff]      
vfmsubadd213ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmsubadd213ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xa7,0xc9]      
vfmsubadd213ps %ymm9, %ymm9, %ymm9 

// CHECK: vfmsubadd231pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096, %xmm15, %xmm15 

// CHECK: vfmsubadd231pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096, %xmm6, %xmm6 

// CHECK: vfmsubadd231pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096, %ymm7, %ymm7 

// CHECK: vfmsubadd231pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb7,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231pd 485498096, %ymm9, %ymm9 

// CHECK: vfmsubadd231pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb7,0x7c,0x82,0xc0]      
vfmsubadd231pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd231pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb7,0x7c,0x82,0x40]      
vfmsubadd231pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd231pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb7,0x74,0x82,0xc0]      
vfmsubadd231pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd231pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb7,0x74,0x82,0x40]      
vfmsubadd231pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd231pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb7,0x7c,0x82,0xc0]      
vfmsubadd231pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd231pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb7,0x7c,0x82,0x40]      
vfmsubadd231pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd231pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb7,0x4c,0x82,0xc0]      
vfmsubadd231pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd231pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb7,0x4c,0x82,0x40]      
vfmsubadd231pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd231pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb7,0x7c,0x02,0x40]      
vfmsubadd231pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsubadd231pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb7,0x74,0x02,0x40]      
vfmsubadd231pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsubadd231pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb7,0x7c,0x02,0x40]      
vfmsubadd231pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsubadd231pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb7,0x4c,0x02,0x40]      
vfmsubadd231pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsubadd231pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb7,0x7a,0x40]      
vfmsubadd231pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd231pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb7,0x72,0x40]      
vfmsubadd231pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd231pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb7,0x7a,0x40]      
vfmsubadd231pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd231pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb7,0x4a,0x40]      
vfmsubadd231pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd231pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xb7,0x3a]      
vfmsubadd231pd (%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd231pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb7,0x32]      
vfmsubadd231pd (%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd231pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb7,0x3a]      
vfmsubadd231pd (%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd231pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xb7,0x0a]      
vfmsubadd231pd (%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd231pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xb7,0xff]      
vfmsubadd231pd %xmm15, %xmm15, %xmm15 

// CHECK: vfmsubadd231pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xb7,0xf6]      
vfmsubadd231pd %xmm6, %xmm6, %xmm6 

// CHECK: vfmsubadd231pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xb7,0xff]      
vfmsubadd231pd %ymm7, %ymm7, %ymm7 

// CHECK: vfmsubadd231pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xb7,0xc9]      
vfmsubadd231pd %ymm9, %ymm9, %ymm9 

// CHECK: vfmsubadd231ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096, %xmm15, %xmm15 

// CHECK: vfmsubadd231ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096, %xmm6, %xmm6 

// CHECK: vfmsubadd231ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096, %ymm7, %ymm7 

// CHECK: vfmsubadd231ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb7,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfmsubadd231ps 485498096, %ymm9, %ymm9 

// CHECK: vfmsubadd231ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb7,0x7c,0x82,0xc0]      
vfmsubadd231ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd231ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb7,0x7c,0x82,0x40]      
vfmsubadd231ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfmsubadd231ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb7,0x74,0x82,0xc0]      
vfmsubadd231ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd231ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb7,0x74,0x82,0x40]      
vfmsubadd231ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfmsubadd231ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb7,0x7c,0x82,0xc0]      
vfmsubadd231ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd231ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb7,0x7c,0x82,0x40]      
vfmsubadd231ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfmsubadd231ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb7,0x4c,0x82,0xc0]      
vfmsubadd231ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd231ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb7,0x4c,0x82,0x40]      
vfmsubadd231ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfmsubadd231ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb7,0x7c,0x02,0x40]      
vfmsubadd231ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfmsubadd231ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb7,0x74,0x02,0x40]      
vfmsubadd231ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfmsubadd231ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb7,0x7c,0x02,0x40]      
vfmsubadd231ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfmsubadd231ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb7,0x4c,0x02,0x40]      
vfmsubadd231ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfmsubadd231ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb7,0x7a,0x40]      
vfmsubadd231ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd231ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb7,0x72,0x40]      
vfmsubadd231ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd231ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb7,0x7a,0x40]      
vfmsubadd231ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd231ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb7,0x4a,0x40]      
vfmsubadd231ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd231ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xb7,0x3a]      
vfmsubadd231ps (%rdx), %xmm15, %xmm15 

// CHECK: vfmsubadd231ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb7,0x32]      
vfmsubadd231ps (%rdx), %xmm6, %xmm6 

// CHECK: vfmsubadd231ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb7,0x3a]      
vfmsubadd231ps (%rdx), %ymm7, %ymm7 

// CHECK: vfmsubadd231ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xb7,0x0a]      
vfmsubadd231ps (%rdx), %ymm9, %ymm9 

// CHECK: vfmsubadd231ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xb7,0xff]      
vfmsubadd231ps %xmm15, %xmm15, %xmm15 

// CHECK: vfmsubadd231ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xb7,0xf6]      
vfmsubadd231ps %xmm6, %xmm6, %xmm6 

// CHECK: vfmsubadd231ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xb7,0xff]      
vfmsubadd231ps %ymm7, %ymm7, %ymm7 

// CHECK: vfmsubadd231ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xb7,0xc9]      
vfmsubadd231ps %ymm9, %ymm9, %ymm9 

// CHECK: vfnmadd132pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd132pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd132pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096, %ymm7, %ymm7 

// CHECK: vfnmadd132pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132pd 485498096, %ymm9, %ymm9 

// CHECK: vfnmadd132pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9c,0x7c,0x82,0xc0]      
vfnmadd132pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9c,0x7c,0x82,0x40]      
vfnmadd132pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9c,0x74,0x82,0xc0]      
vfnmadd132pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9c,0x74,0x82,0x40]      
vfnmadd132pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9c,0x7c,0x82,0xc0]      
vfnmadd132pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd132pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9c,0x7c,0x82,0x40]      
vfnmadd132pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd132pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9c,0x4c,0x82,0xc0]      
vfnmadd132pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd132pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9c,0x4c,0x82,0x40]      
vfnmadd132pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd132pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9c,0x7c,0x02,0x40]      
vfnmadd132pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd132pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9c,0x74,0x02,0x40]      
vfnmadd132pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd132pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9c,0x7c,0x02,0x40]      
vfnmadd132pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmadd132pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9c,0x4c,0x02,0x40]      
vfnmadd132pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmadd132pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9c,0x7a,0x40]      
vfnmadd132pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9c,0x72,0x40]      
vfnmadd132pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9c,0x7a,0x40]      
vfnmadd132pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd132pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9c,0x4a,0x40]      
vfnmadd132pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd132pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9c,0x3a]      
vfnmadd132pd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9c,0x32]      
vfnmadd132pd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9c,0x3a]      
vfnmadd132pd (%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd132pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9c,0x0a]      
vfnmadd132pd (%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd132pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x9c,0xff]      
vfnmadd132pd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd132pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9c,0xf6]      
vfnmadd132pd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd132pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9c,0xff]      
vfnmadd132pd %ymm7, %ymm7, %ymm7 

// CHECK: vfnmadd132pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x9c,0xc9]      
vfnmadd132pd %ymm9, %ymm9, %ymm9 

// CHECK: vfnmadd132ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd132ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd132ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096, %ymm7, %ymm7 

// CHECK: vfnmadd132ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ps 485498096, %ymm9, %ymm9 

// CHECK: vfnmadd132ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9c,0x7c,0x82,0xc0]      
vfnmadd132ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9c,0x7c,0x82,0x40]      
vfnmadd132ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9c,0x74,0x82,0xc0]      
vfnmadd132ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9c,0x74,0x82,0x40]      
vfnmadd132ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9c,0x7c,0x82,0xc0]      
vfnmadd132ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd132ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9c,0x7c,0x82,0x40]      
vfnmadd132ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd132ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9c,0x4c,0x82,0xc0]      
vfnmadd132ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd132ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9c,0x4c,0x82,0x40]      
vfnmadd132ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd132ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9c,0x7c,0x02,0x40]      
vfnmadd132ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd132ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9c,0x74,0x02,0x40]      
vfnmadd132ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd132ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9c,0x7c,0x02,0x40]      
vfnmadd132ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmadd132ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9c,0x4c,0x02,0x40]      
vfnmadd132ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmadd132ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9c,0x7a,0x40]      
vfnmadd132ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9c,0x72,0x40]      
vfnmadd132ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9c,0x7a,0x40]      
vfnmadd132ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd132ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9c,0x4a,0x40]      
vfnmadd132ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd132ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9c,0x3a]      
vfnmadd132ps (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9c,0x32]      
vfnmadd132ps (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9c,0x3a]      
vfnmadd132ps (%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd132ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9c,0x0a]      
vfnmadd132ps (%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd132ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x9c,0xff]      
vfnmadd132ps %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd132ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9c,0xf6]      
vfnmadd132ps %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd132ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9c,0xff]      
vfnmadd132ps %ymm7, %ymm7, %ymm7 

// CHECK: vfnmadd132ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x9c,0xc9]      
vfnmadd132ps %ymm9, %ymm9, %ymm9 

// CHECK: vfnmadd132sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132sd 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd132sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132sd 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd132sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x7c,0x82,0xc0]      
vfnmadd132sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x7c,0x82,0x40]      
vfnmadd132sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9d,0x74,0x82,0xc0]      
vfnmadd132sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9d,0x74,0x82,0x40]      
vfnmadd132sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x7c,0x02,0x40]      
vfnmadd132sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd132sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9d,0x74,0x02,0x40]      
vfnmadd132sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd132sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x7a,0x40]      
vfnmadd132sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9d,0x72,0x40]      
vfnmadd132sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x3a]      
vfnmadd132sd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9d,0x32]      
vfnmadd132sd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x9d,0xff]      
vfnmadd132sd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd132sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9d,0xf6]      
vfnmadd132sd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd132ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ss 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd132ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd132ss 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd132ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x7c,0x82,0xc0]      
vfnmadd132ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x7c,0x82,0x40]      
vfnmadd132ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd132ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9d,0x74,0x82,0xc0]      
vfnmadd132ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9d,0x74,0x82,0x40]      
vfnmadd132ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd132ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x7c,0x02,0x40]      
vfnmadd132ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd132ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9d,0x74,0x02,0x40]      
vfnmadd132ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd132ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x7a,0x40]      
vfnmadd132ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9d,0x72,0x40]      
vfnmadd132ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x3a]      
vfnmadd132ss (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd132ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9d,0x32]      
vfnmadd132ss (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd132ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x9d,0xff]      
vfnmadd132ss %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd132ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9d,0xf6]      
vfnmadd132ss %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd213pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xac,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd213pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xac,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd213pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xac,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096, %ymm7, %ymm7 

// CHECK: vfnmadd213pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xac,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213pd 485498096, %ymm9, %ymm9 

// CHECK: vfnmadd213pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xac,0x7c,0x82,0xc0]      
vfnmadd213pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xac,0x7c,0x82,0x40]      
vfnmadd213pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xac,0x74,0x82,0xc0]      
vfnmadd213pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xac,0x74,0x82,0x40]      
vfnmadd213pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xac,0x7c,0x82,0xc0]      
vfnmadd213pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd213pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xac,0x7c,0x82,0x40]      
vfnmadd213pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd213pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xac,0x4c,0x82,0xc0]      
vfnmadd213pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd213pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xac,0x4c,0x82,0x40]      
vfnmadd213pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd213pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xac,0x7c,0x02,0x40]      
vfnmadd213pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd213pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xac,0x74,0x02,0x40]      
vfnmadd213pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd213pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xac,0x7c,0x02,0x40]      
vfnmadd213pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmadd213pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xac,0x4c,0x02,0x40]      
vfnmadd213pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmadd213pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xac,0x7a,0x40]      
vfnmadd213pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xac,0x72,0x40]      
vfnmadd213pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xac,0x7a,0x40]      
vfnmadd213pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd213pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xac,0x4a,0x40]      
vfnmadd213pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd213pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xac,0x3a]      
vfnmadd213pd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xac,0x32]      
vfnmadd213pd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xac,0x3a]      
vfnmadd213pd (%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd213pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xac,0x0a]      
vfnmadd213pd (%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd213pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xac,0xff]      
vfnmadd213pd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd213pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xac,0xf6]      
vfnmadd213pd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd213pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xac,0xff]      
vfnmadd213pd %ymm7, %ymm7, %ymm7 

// CHECK: vfnmadd213pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xac,0xc9]      
vfnmadd213pd %ymm9, %ymm9, %ymm9 

// CHECK: vfnmadd213ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xac,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd213ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xac,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd213ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xac,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096, %ymm7, %ymm7 

// CHECK: vfnmadd213ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xac,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ps 485498096, %ymm9, %ymm9 

// CHECK: vfnmadd213ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xac,0x7c,0x82,0xc0]      
vfnmadd213ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xac,0x7c,0x82,0x40]      
vfnmadd213ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xac,0x74,0x82,0xc0]      
vfnmadd213ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xac,0x74,0x82,0x40]      
vfnmadd213ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xac,0x7c,0x82,0xc0]      
vfnmadd213ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd213ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xac,0x7c,0x82,0x40]      
vfnmadd213ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd213ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xac,0x4c,0x82,0xc0]      
vfnmadd213ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd213ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xac,0x4c,0x82,0x40]      
vfnmadd213ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd213ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xac,0x7c,0x02,0x40]      
vfnmadd213ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd213ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xac,0x74,0x02,0x40]      
vfnmadd213ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd213ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xac,0x7c,0x02,0x40]      
vfnmadd213ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmadd213ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xac,0x4c,0x02,0x40]      
vfnmadd213ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmadd213ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xac,0x7a,0x40]      
vfnmadd213ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xac,0x72,0x40]      
vfnmadd213ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xac,0x7a,0x40]      
vfnmadd213ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd213ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xac,0x4a,0x40]      
vfnmadd213ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd213ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xac,0x3a]      
vfnmadd213ps (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xac,0x32]      
vfnmadd213ps (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xac,0x3a]      
vfnmadd213ps (%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd213ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xac,0x0a]      
vfnmadd213ps (%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd213ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xac,0xff]      
vfnmadd213ps %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd213ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xac,0xf6]      
vfnmadd213ps %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd213ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xac,0xff]      
vfnmadd213ps %ymm7, %ymm7, %ymm7 

// CHECK: vfnmadd213ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xac,0xc9]      
vfnmadd213ps %ymm9, %ymm9, %ymm9 

// CHECK: vfnmadd213sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213sd 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd213sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xad,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213sd 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd213sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x7c,0x82,0xc0]      
vfnmadd213sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x7c,0x82,0x40]      
vfnmadd213sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xad,0x74,0x82,0xc0]      
vfnmadd213sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xad,0x74,0x82,0x40]      
vfnmadd213sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x7c,0x02,0x40]      
vfnmadd213sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd213sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xad,0x74,0x02,0x40]      
vfnmadd213sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd213sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x7a,0x40]      
vfnmadd213sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xad,0x72,0x40]      
vfnmadd213sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x3a]      
vfnmadd213sd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xad,0x32]      
vfnmadd213sd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xad,0xff]      
vfnmadd213sd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd213sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xad,0xf6]      
vfnmadd213sd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd213ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ss 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd213ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xad,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd213ss 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd213ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x7c,0x82,0xc0]      
vfnmadd213ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x7c,0x82,0x40]      
vfnmadd213ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd213ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xad,0x74,0x82,0xc0]      
vfnmadd213ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xad,0x74,0x82,0x40]      
vfnmadd213ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd213ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x7c,0x02,0x40]      
vfnmadd213ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd213ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xad,0x74,0x02,0x40]      
vfnmadd213ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd213ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x7a,0x40]      
vfnmadd213ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xad,0x72,0x40]      
vfnmadd213ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x3a]      
vfnmadd213ss (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd213ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xad,0x32]      
vfnmadd213ss (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd213ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xad,0xff]      
vfnmadd213ss %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd213ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xad,0xf6]      
vfnmadd213ss %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd231pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd231pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbc,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd231pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096, %ymm7, %ymm7 

// CHECK: vfnmadd231pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbc,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231pd 485498096, %ymm9, %ymm9 

// CHECK: vfnmadd231pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbc,0x7c,0x82,0xc0]      
vfnmadd231pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbc,0x7c,0x82,0x40]      
vfnmadd231pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbc,0x74,0x82,0xc0]      
vfnmadd231pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbc,0x74,0x82,0x40]      
vfnmadd231pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbc,0x7c,0x82,0xc0]      
vfnmadd231pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd231pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbc,0x7c,0x82,0x40]      
vfnmadd231pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd231pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbc,0x4c,0x82,0xc0]      
vfnmadd231pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd231pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbc,0x4c,0x82,0x40]      
vfnmadd231pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd231pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbc,0x7c,0x02,0x40]      
vfnmadd231pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd231pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbc,0x74,0x02,0x40]      
vfnmadd231pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd231pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbc,0x7c,0x02,0x40]      
vfnmadd231pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmadd231pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbc,0x4c,0x02,0x40]      
vfnmadd231pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmadd231pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbc,0x7a,0x40]      
vfnmadd231pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbc,0x72,0x40]      
vfnmadd231pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbc,0x7a,0x40]      
vfnmadd231pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd231pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbc,0x4a,0x40]      
vfnmadd231pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd231pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbc,0x3a]      
vfnmadd231pd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbc,0x32]      
vfnmadd231pd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbc,0x3a]      
vfnmadd231pd (%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd231pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbc,0x0a]      
vfnmadd231pd (%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd231pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xbc,0xff]      
vfnmadd231pd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd231pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbc,0xf6]      
vfnmadd231pd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd231pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbc,0xff]      
vfnmadd231pd %ymm7, %ymm7, %ymm7 

// CHECK: vfnmadd231pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xbc,0xc9]      
vfnmadd231pd %ymm9, %ymm9, %ymm9 

// CHECK: vfnmadd231ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd231ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbc,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd231ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096, %ymm7, %ymm7 

// CHECK: vfnmadd231ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbc,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ps 485498096, %ymm9, %ymm9 

// CHECK: vfnmadd231ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbc,0x7c,0x82,0xc0]      
vfnmadd231ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbc,0x7c,0x82,0x40]      
vfnmadd231ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbc,0x74,0x82,0xc0]      
vfnmadd231ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbc,0x74,0x82,0x40]      
vfnmadd231ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbc,0x7c,0x82,0xc0]      
vfnmadd231ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd231ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbc,0x7c,0x82,0x40]      
vfnmadd231ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmadd231ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbc,0x4c,0x82,0xc0]      
vfnmadd231ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd231ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbc,0x4c,0x82,0x40]      
vfnmadd231ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmadd231ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbc,0x7c,0x02,0x40]      
vfnmadd231ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd231ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbc,0x74,0x02,0x40]      
vfnmadd231ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd231ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbc,0x7c,0x02,0x40]      
vfnmadd231ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmadd231ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbc,0x4c,0x02,0x40]      
vfnmadd231ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmadd231ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbc,0x7a,0x40]      
vfnmadd231ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbc,0x72,0x40]      
vfnmadd231ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbc,0x7a,0x40]      
vfnmadd231ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd231ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbc,0x4a,0x40]      
vfnmadd231ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd231ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbc,0x3a]      
vfnmadd231ps (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbc,0x32]      
vfnmadd231ps (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbc,0x3a]      
vfnmadd231ps (%rdx), %ymm7, %ymm7 

// CHECK: vfnmadd231ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbc,0x0a]      
vfnmadd231ps (%rdx), %ymm9, %ymm9 

// CHECK: vfnmadd231ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xbc,0xff]      
vfnmadd231ps %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd231ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbc,0xf6]      
vfnmadd231ps %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd231ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbc,0xff]      
vfnmadd231ps %ymm7, %ymm7, %ymm7 

// CHECK: vfnmadd231ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xbc,0xc9]      
vfnmadd231ps %ymm9, %ymm9, %ymm9 

// CHECK: vfnmadd231sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231sd 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd231sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbd,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231sd 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd231sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x7c,0x82,0xc0]      
vfnmadd231sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x7c,0x82,0x40]      
vfnmadd231sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbd,0x74,0x82,0xc0]      
vfnmadd231sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbd,0x74,0x82,0x40]      
vfnmadd231sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x7c,0x02,0x40]      
vfnmadd231sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd231sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbd,0x74,0x02,0x40]      
vfnmadd231sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd231sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x7a,0x40]      
vfnmadd231sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbd,0x72,0x40]      
vfnmadd231sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x3a]      
vfnmadd231sd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbd,0x32]      
vfnmadd231sd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xbd,0xff]      
vfnmadd231sd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd231sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbd,0xf6]      
vfnmadd231sd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmadd231ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ss 485498096, %xmm15, %xmm15 

// CHECK: vfnmadd231ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbd,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmadd231ss 485498096, %xmm6, %xmm6 

// CHECK: vfnmadd231ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x7c,0x82,0xc0]      
vfnmadd231ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x7c,0x82,0x40]      
vfnmadd231ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmadd231ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbd,0x74,0x82,0xc0]      
vfnmadd231ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbd,0x74,0x82,0x40]      
vfnmadd231ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmadd231ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x7c,0x02,0x40]      
vfnmadd231ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmadd231ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbd,0x74,0x02,0x40]      
vfnmadd231ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmadd231ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x7a,0x40]      
vfnmadd231ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbd,0x72,0x40]      
vfnmadd231ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x3a]      
vfnmadd231ss (%rdx), %xmm15, %xmm15 

// CHECK: vfnmadd231ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbd,0x32]      
vfnmadd231ss (%rdx), %xmm6, %xmm6 

// CHECK: vfnmadd231ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xbd,0xff]      
vfnmadd231ss %xmm15, %xmm15, %xmm15 

// CHECK: vfnmadd231ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbd,0xf6]      
vfnmadd231ss %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub132pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub132pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub132pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096, %ymm7, %ymm7 

// CHECK: vfnmsub132pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132pd 485498096, %ymm9, %ymm9 

// CHECK: vfnmsub132pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9e,0x7c,0x82,0xc0]      
vfnmsub132pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9e,0x7c,0x82,0x40]      
vfnmsub132pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9e,0x74,0x82,0xc0]      
vfnmsub132pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9e,0x74,0x82,0x40]      
vfnmsub132pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9e,0x7c,0x82,0xc0]      
vfnmsub132pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub132pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9e,0x7c,0x82,0x40]      
vfnmsub132pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub132pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9e,0x4c,0x82,0xc0]      
vfnmsub132pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub132pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9e,0x4c,0x82,0x40]      
vfnmsub132pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub132pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9e,0x7c,0x02,0x40]      
vfnmsub132pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub132pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9e,0x74,0x02,0x40]      
vfnmsub132pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub132pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9e,0x7c,0x02,0x40]      
vfnmsub132pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmsub132pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9e,0x4c,0x02,0x40]      
vfnmsub132pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmsub132pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9e,0x7a,0x40]      
vfnmsub132pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9e,0x72,0x40]      
vfnmsub132pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9e,0x7a,0x40]      
vfnmsub132pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub132pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9e,0x4a,0x40]      
vfnmsub132pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub132pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9e,0x3a]      
vfnmsub132pd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9e,0x32]      
vfnmsub132pd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9e,0x3a]      
vfnmsub132pd (%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub132pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x9e,0x0a]      
vfnmsub132pd (%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub132pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x9e,0xff]      
vfnmsub132pd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub132pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9e,0xf6]      
vfnmsub132pd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub132pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x9e,0xff]      
vfnmsub132pd %ymm7, %ymm7, %ymm7 

// CHECK: vfnmsub132pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x9e,0xc9]      
vfnmsub132pd %ymm9, %ymm9, %ymm9 

// CHECK: vfnmsub132ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub132ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub132ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096, %ymm7, %ymm7 

// CHECK: vfnmsub132ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ps 485498096, %ymm9, %ymm9 

// CHECK: vfnmsub132ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9e,0x7c,0x82,0xc0]      
vfnmsub132ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9e,0x7c,0x82,0x40]      
vfnmsub132ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9e,0x74,0x82,0xc0]      
vfnmsub132ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9e,0x74,0x82,0x40]      
vfnmsub132ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9e,0x7c,0x82,0xc0]      
vfnmsub132ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub132ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9e,0x7c,0x82,0x40]      
vfnmsub132ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub132ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9e,0x4c,0x82,0xc0]      
vfnmsub132ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub132ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9e,0x4c,0x82,0x40]      
vfnmsub132ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub132ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9e,0x7c,0x02,0x40]      
vfnmsub132ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub132ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9e,0x74,0x02,0x40]      
vfnmsub132ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub132ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9e,0x7c,0x02,0x40]      
vfnmsub132ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmsub132ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9e,0x4c,0x02,0x40]      
vfnmsub132ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmsub132ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9e,0x7a,0x40]      
vfnmsub132ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9e,0x72,0x40]      
vfnmsub132ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9e,0x7a,0x40]      
vfnmsub132ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub132ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9e,0x4a,0x40]      
vfnmsub132ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub132ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9e,0x3a]      
vfnmsub132ps (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9e,0x32]      
vfnmsub132ps (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9e,0x3a]      
vfnmsub132ps (%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub132ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x9e,0x0a]      
vfnmsub132ps (%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub132ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x9e,0xff]      
vfnmsub132ps %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub132ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9e,0xf6]      
vfnmsub132ps %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub132ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x9e,0xff]      
vfnmsub132ps %ymm7, %ymm7, %ymm7 

// CHECK: vfnmsub132ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x9e,0xc9]      
vfnmsub132ps %ymm9, %ymm9, %ymm9 

// CHECK: vfnmsub132sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132sd 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub132sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132sd 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub132sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x7c,0x82,0xc0]      
vfnmsub132sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x7c,0x82,0x40]      
vfnmsub132sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9f,0x74,0x82,0xc0]      
vfnmsub132sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9f,0x74,0x82,0x40]      
vfnmsub132sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x7c,0x02,0x40]      
vfnmsub132sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub132sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9f,0x74,0x02,0x40]      
vfnmsub132sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub132sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x7a,0x40]      
vfnmsub132sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9f,0x72,0x40]      
vfnmsub132sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x3a]      
vfnmsub132sd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9f,0x32]      
vfnmsub132sd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x9f,0xff]      
vfnmsub132sd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub132sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x9f,0xf6]      
vfnmsub132sd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub132ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ss 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub132ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub132ss 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub132ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x7c,0x82,0xc0]      
vfnmsub132ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x7c,0x82,0x40]      
vfnmsub132ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub132ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9f,0x74,0x82,0xc0]      
vfnmsub132ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9f,0x74,0x82,0x40]      
vfnmsub132ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub132ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x7c,0x02,0x40]      
vfnmsub132ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub132ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9f,0x74,0x02,0x40]      
vfnmsub132ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub132ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x7a,0x40]      
vfnmsub132ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9f,0x72,0x40]      
vfnmsub132ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x3a]      
vfnmsub132ss (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub132ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9f,0x32]      
vfnmsub132ss (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub132ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x9f,0xff]      
vfnmsub132ss %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub132ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x9f,0xf6]      
vfnmsub132ss %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub213pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xae,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub213pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xae,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub213pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xae,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096, %ymm7, %ymm7 

// CHECK: vfnmsub213pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xae,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213pd 485498096, %ymm9, %ymm9 

// CHECK: vfnmsub213pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xae,0x7c,0x82,0xc0]      
vfnmsub213pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xae,0x7c,0x82,0x40]      
vfnmsub213pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xae,0x74,0x82,0xc0]      
vfnmsub213pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xae,0x74,0x82,0x40]      
vfnmsub213pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xae,0x7c,0x82,0xc0]      
vfnmsub213pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub213pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xae,0x7c,0x82,0x40]      
vfnmsub213pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub213pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xae,0x4c,0x82,0xc0]      
vfnmsub213pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub213pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xae,0x4c,0x82,0x40]      
vfnmsub213pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub213pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xae,0x7c,0x02,0x40]      
vfnmsub213pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub213pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xae,0x74,0x02,0x40]      
vfnmsub213pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub213pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xae,0x7c,0x02,0x40]      
vfnmsub213pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmsub213pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xae,0x4c,0x02,0x40]      
vfnmsub213pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmsub213pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xae,0x7a,0x40]      
vfnmsub213pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xae,0x72,0x40]      
vfnmsub213pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xae,0x7a,0x40]      
vfnmsub213pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub213pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xae,0x4a,0x40]      
vfnmsub213pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub213pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xae,0x3a]      
vfnmsub213pd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xae,0x32]      
vfnmsub213pd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xae,0x3a]      
vfnmsub213pd (%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub213pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xae,0x0a]      
vfnmsub213pd (%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub213pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xae,0xff]      
vfnmsub213pd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub213pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xae,0xf6]      
vfnmsub213pd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub213pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xae,0xff]      
vfnmsub213pd %ymm7, %ymm7, %ymm7 

// CHECK: vfnmsub213pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xae,0xc9]      
vfnmsub213pd %ymm9, %ymm9, %ymm9 

// CHECK: vfnmsub213ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xae,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub213ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xae,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub213ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xae,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096, %ymm7, %ymm7 

// CHECK: vfnmsub213ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xae,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ps 485498096, %ymm9, %ymm9 

// CHECK: vfnmsub213ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xae,0x7c,0x82,0xc0]      
vfnmsub213ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xae,0x7c,0x82,0x40]      
vfnmsub213ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xae,0x74,0x82,0xc0]      
vfnmsub213ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xae,0x74,0x82,0x40]      
vfnmsub213ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xae,0x7c,0x82,0xc0]      
vfnmsub213ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub213ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xae,0x7c,0x82,0x40]      
vfnmsub213ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub213ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xae,0x4c,0x82,0xc0]      
vfnmsub213ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub213ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xae,0x4c,0x82,0x40]      
vfnmsub213ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub213ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xae,0x7c,0x02,0x40]      
vfnmsub213ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub213ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xae,0x74,0x02,0x40]      
vfnmsub213ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub213ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xae,0x7c,0x02,0x40]      
vfnmsub213ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmsub213ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xae,0x4c,0x02,0x40]      
vfnmsub213ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmsub213ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xae,0x7a,0x40]      
vfnmsub213ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xae,0x72,0x40]      
vfnmsub213ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xae,0x7a,0x40]      
vfnmsub213ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub213ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xae,0x4a,0x40]      
vfnmsub213ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub213ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xae,0x3a]      
vfnmsub213ps (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xae,0x32]      
vfnmsub213ps (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xae,0x3a]      
vfnmsub213ps (%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub213ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xae,0x0a]      
vfnmsub213ps (%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub213ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xae,0xff]      
vfnmsub213ps %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub213ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xae,0xf6]      
vfnmsub213ps %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub213ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xae,0xff]      
vfnmsub213ps %ymm7, %ymm7, %ymm7 

// CHECK: vfnmsub213ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xae,0xc9]      
vfnmsub213ps %ymm9, %ymm9, %ymm9 

// CHECK: vfnmsub213sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213sd 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub213sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaf,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213sd 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub213sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x7c,0x82,0xc0]      
vfnmsub213sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x7c,0x82,0x40]      
vfnmsub213sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaf,0x74,0x82,0xc0]      
vfnmsub213sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaf,0x74,0x82,0x40]      
vfnmsub213sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x7c,0x02,0x40]      
vfnmsub213sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub213sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaf,0x74,0x02,0x40]      
vfnmsub213sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub213sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x7a,0x40]      
vfnmsub213sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaf,0x72,0x40]      
vfnmsub213sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x3a]      
vfnmsub213sd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaf,0x32]      
vfnmsub213sd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xaf,0xff]      
vfnmsub213sd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub213sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xaf,0xf6]      
vfnmsub213sd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub213ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ss 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub213ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaf,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub213ss 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub213ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x7c,0x82,0xc0]      
vfnmsub213ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x7c,0x82,0x40]      
vfnmsub213ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub213ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaf,0x74,0x82,0xc0]      
vfnmsub213ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaf,0x74,0x82,0x40]      
vfnmsub213ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub213ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x7c,0x02,0x40]      
vfnmsub213ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub213ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaf,0x74,0x02,0x40]      
vfnmsub213ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub213ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x7a,0x40]      
vfnmsub213ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaf,0x72,0x40]      
vfnmsub213ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x3a]      
vfnmsub213ss (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub213ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaf,0x32]      
vfnmsub213ss (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub213ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xaf,0xff]      
vfnmsub213ss %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub213ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xaf,0xf6]      
vfnmsub213ss %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub231pd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbe,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub231pd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbe,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub231pd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbe,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096, %ymm7, %ymm7 

// CHECK: vfnmsub231pd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbe,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231pd 485498096, %ymm9, %ymm9 

// CHECK: vfnmsub231pd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbe,0x7c,0x82,0xc0]      
vfnmsub231pd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231pd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbe,0x7c,0x82,0x40]      
vfnmsub231pd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231pd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbe,0x74,0x82,0xc0]      
vfnmsub231pd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231pd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbe,0x74,0x82,0x40]      
vfnmsub231pd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231pd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbe,0x7c,0x82,0xc0]      
vfnmsub231pd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub231pd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbe,0x7c,0x82,0x40]      
vfnmsub231pd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub231pd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbe,0x4c,0x82,0xc0]      
vfnmsub231pd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub231pd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbe,0x4c,0x82,0x40]      
vfnmsub231pd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub231pd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbe,0x7c,0x02,0x40]      
vfnmsub231pd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub231pd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbe,0x74,0x02,0x40]      
vfnmsub231pd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub231pd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbe,0x7c,0x02,0x40]      
vfnmsub231pd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmsub231pd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbe,0x4c,0x02,0x40]      
vfnmsub231pd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmsub231pd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbe,0x7a,0x40]      
vfnmsub231pd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231pd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbe,0x72,0x40]      
vfnmsub231pd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231pd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbe,0x7a,0x40]      
vfnmsub231pd 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub231pd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbe,0x4a,0x40]      
vfnmsub231pd 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub231pd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbe,0x3a]      
vfnmsub231pd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231pd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbe,0x32]      
vfnmsub231pd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231pd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbe,0x3a]      
vfnmsub231pd (%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub231pd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0xbe,0x0a]      
vfnmsub231pd (%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub231pd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xbe,0xff]      
vfnmsub231pd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub231pd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbe,0xf6]      
vfnmsub231pd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub231pd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0xbe,0xff]      
vfnmsub231pd %ymm7, %ymm7, %ymm7 

// CHECK: vfnmsub231pd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0xbe,0xc9]      
vfnmsub231pd %ymm9, %ymm9, %ymm9 

// CHECK: vfnmsub231ps 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbe,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub231ps 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbe,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub231ps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbe,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096, %ymm7, %ymm7 

// CHECK: vfnmsub231ps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbe,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ps 485498096, %ymm9, %ymm9 

// CHECK: vfnmsub231ps -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbe,0x7c,0x82,0xc0]      
vfnmsub231ps -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231ps 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbe,0x7c,0x82,0x40]      
vfnmsub231ps 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231ps -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbe,0x74,0x82,0xc0]      
vfnmsub231ps -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231ps 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbe,0x74,0x82,0x40]      
vfnmsub231ps 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231ps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbe,0x7c,0x82,0xc0]      
vfnmsub231ps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub231ps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbe,0x7c,0x82,0x40]      
vfnmsub231ps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vfnmsub231ps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbe,0x4c,0x82,0xc0]      
vfnmsub231ps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub231ps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbe,0x4c,0x82,0x40]      
vfnmsub231ps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vfnmsub231ps 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbe,0x7c,0x02,0x40]      
vfnmsub231ps 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub231ps 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbe,0x74,0x02,0x40]      
vfnmsub231ps 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub231ps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbe,0x7c,0x02,0x40]      
vfnmsub231ps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vfnmsub231ps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbe,0x4c,0x02,0x40]      
vfnmsub231ps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vfnmsub231ps 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbe,0x7a,0x40]      
vfnmsub231ps 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231ps 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbe,0x72,0x40]      
vfnmsub231ps 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231ps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbe,0x7a,0x40]      
vfnmsub231ps 64(%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub231ps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbe,0x4a,0x40]      
vfnmsub231ps 64(%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub231ps (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbe,0x3a]      
vfnmsub231ps (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231ps (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbe,0x32]      
vfnmsub231ps (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231ps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbe,0x3a]      
vfnmsub231ps (%rdx), %ymm7, %ymm7 

// CHECK: vfnmsub231ps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0xbe,0x0a]      
vfnmsub231ps (%rdx), %ymm9, %ymm9 

// CHECK: vfnmsub231ps %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xbe,0xff]      
vfnmsub231ps %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub231ps %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbe,0xf6]      
vfnmsub231ps %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub231ps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0xbe,0xff]      
vfnmsub231ps %ymm7, %ymm7, %ymm7 

// CHECK: vfnmsub231ps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0xbe,0xc9]      
vfnmsub231ps %ymm9, %ymm9, %ymm9 

// CHECK: vfnmsub231sd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231sd 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub231sd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbf,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231sd 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub231sd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbf,0x7c,0x82,0xc0]      
vfnmsub231sd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231sd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbf,0x7c,0x82,0x40]      
vfnmsub231sd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231sd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbf,0x74,0x82,0xc0]      
vfnmsub231sd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231sd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbf,0x74,0x82,0x40]      
vfnmsub231sd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231sd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbf,0x7c,0x02,0x40]      
vfnmsub231sd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub231sd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbf,0x74,0x02,0x40]      
vfnmsub231sd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub231sd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbf,0x7a,0x40]      
vfnmsub231sd 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231sd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbf,0x72,0x40]      
vfnmsub231sd 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231sd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0xbf,0x3a]      
vfnmsub231sd (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231sd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbf,0x32]      
vfnmsub231sd (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231sd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0xbf,0xff]      
vfnmsub231sd %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub231sd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0xbf,0xf6]      
vfnmsub231sd %xmm6, %xmm6, %xmm6 

// CHECK: vfnmsub231ss 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ss 485498096, %xmm15, %xmm15 

// CHECK: vfnmsub231ss 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbf,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vfnmsub231ss 485498096, %xmm6, %xmm6 

// CHECK: vfnmsub231ss -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x7c,0x82,0xc0]      
vfnmsub231ss -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231ss 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x7c,0x82,0x40]      
vfnmsub231ss 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vfnmsub231ss -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbf,0x74,0x82,0xc0]      
vfnmsub231ss -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231ss 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbf,0x74,0x82,0x40]      
vfnmsub231ss 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vfnmsub231ss 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x7c,0x02,0x40]      
vfnmsub231ss 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vfnmsub231ss 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbf,0x74,0x02,0x40]      
vfnmsub231ss 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vfnmsub231ss 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x7a,0x40]      
vfnmsub231ss 64(%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231ss 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbf,0x72,0x40]      
vfnmsub231ss 64(%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231ss (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x3a]      
vfnmsub231ss (%rdx), %xmm15, %xmm15 

// CHECK: vfnmsub231ss (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbf,0x32]      
vfnmsub231ss (%rdx), %xmm6, %xmm6 

// CHECK: vfnmsub231ss %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xbf,0xff]      
vfnmsub231ss %xmm15, %xmm15, %xmm15 

// CHECK: vfnmsub231ss %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xbf,0xf6]      
vfnmsub231ss %xmm6, %xmm6, %xmm6 

