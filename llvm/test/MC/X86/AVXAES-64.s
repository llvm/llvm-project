// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vaesdec 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xde,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesdec 485498096, %xmm15, %xmm15 

// CHECK: vaesdec 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xde,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesdec 485498096, %xmm6, %xmm6 

// CHECK: vaesdec 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xde,0x7c,0x82,0x40]      
vaesdec 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesdec -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xde,0x7c,0x82,0xc0]      
vaesdec -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesdec 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xde,0x74,0x82,0x40]      
vaesdec 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesdec -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xde,0x74,0x82,0xc0]      
vaesdec -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesdec 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xde,0x7c,0x02,0x40]      
vaesdec 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaesdec 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xde,0x74,0x02,0x40]      
vaesdec 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaesdec 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xde,0x7a,0x40]      
vaesdec 64(%rdx), %xmm15, %xmm15 

// CHECK: vaesdec 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xde,0x72,0x40]      
vaesdec 64(%rdx), %xmm6, %xmm6 

// CHECK: vaesdeclast 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesdeclast 485498096, %xmm15, %xmm15 

// CHECK: vaesdeclast 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdf,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesdeclast 485498096, %xmm6, %xmm6 

// CHECK: vaesdeclast 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdf,0x7c,0x82,0x40]      
vaesdeclast 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesdeclast -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdf,0x7c,0x82,0xc0]      
vaesdeclast -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesdeclast 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdf,0x74,0x82,0x40]      
vaesdeclast 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesdeclast -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdf,0x74,0x82,0xc0]      
vaesdeclast -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesdeclast 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdf,0x7c,0x02,0x40]      
vaesdeclast 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaesdeclast 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdf,0x74,0x02,0x40]      
vaesdeclast 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaesdeclast 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdf,0x7a,0x40]      
vaesdeclast 64(%rdx), %xmm15, %xmm15 

// CHECK: vaesdeclast 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdf,0x72,0x40]      
vaesdeclast 64(%rdx), %xmm6, %xmm6 

// CHECK: vaesdeclast (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdf,0x3a]      
vaesdeclast (%rdx), %xmm15, %xmm15 

// CHECK: vaesdeclast (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdf,0x32]      
vaesdeclast (%rdx), %xmm6, %xmm6 

// CHECK: vaesdeclast %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xdf,0xff]      
vaesdeclast %xmm15, %xmm15, %xmm15 

// CHECK: vaesdeclast %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdf,0xf6]      
vaesdeclast %xmm6, %xmm6, %xmm6 

// CHECK: vaesdec (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xde,0x3a]      
vaesdec (%rdx), %xmm15, %xmm15 

// CHECK: vaesdec (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xde,0x32]      
vaesdec (%rdx), %xmm6, %xmm6 

// CHECK: vaesdec %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xde,0xff]      
vaesdec %xmm15, %xmm15, %xmm15 

// CHECK: vaesdec %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xde,0xf6]      
vaesdec %xmm6, %xmm6, %xmm6 

// CHECK: vaesenc 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesenc 485498096, %xmm15, %xmm15 

// CHECK: vaesenc 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdc,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesenc 485498096, %xmm6, %xmm6 

// CHECK: vaesenc 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdc,0x7c,0x82,0x40]      
vaesenc 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesenc -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdc,0x7c,0x82,0xc0]      
vaesenc -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesenc 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdc,0x74,0x82,0x40]      
vaesenc 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesenc -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdc,0x74,0x82,0xc0]      
vaesenc -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesenc 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdc,0x7c,0x02,0x40]      
vaesenc 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaesenc 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdc,0x74,0x02,0x40]      
vaesenc 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaesenc 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdc,0x7a,0x40]      
vaesenc 64(%rdx), %xmm15, %xmm15 

// CHECK: vaesenc 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdc,0x72,0x40]      
vaesenc 64(%rdx), %xmm6, %xmm6 

// CHECK: vaesenclast 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesenclast 485498096, %xmm15, %xmm15 

// CHECK: vaesenclast 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdd,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vaesenclast 485498096, %xmm6, %xmm6 

// CHECK: vaesenclast 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdd,0x7c,0x82,0x40]      
vaesenclast 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesenclast -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdd,0x7c,0x82,0xc0]      
vaesenclast -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vaesenclast 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdd,0x74,0x82,0x40]      
vaesenclast 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesenclast -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdd,0x74,0x82,0xc0]      
vaesenclast -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vaesenclast 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdd,0x7c,0x02,0x40]      
vaesenclast 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vaesenclast 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdd,0x74,0x02,0x40]      
vaesenclast 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vaesenclast 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdd,0x7a,0x40]      
vaesenclast 64(%rdx), %xmm15, %xmm15 

// CHECK: vaesenclast 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdd,0x72,0x40]      
vaesenclast 64(%rdx), %xmm6, %xmm6 

// CHECK: vaesenclast (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdd,0x3a]      
vaesenclast (%rdx), %xmm15, %xmm15 

// CHECK: vaesenclast (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdd,0x32]      
vaesenclast (%rdx), %xmm6, %xmm6 

// CHECK: vaesenclast %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xdd,0xff]      
vaesenclast %xmm15, %xmm15, %xmm15 

// CHECK: vaesenclast %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdd,0xf6]      
vaesenclast %xmm6, %xmm6, %xmm6 

// CHECK: vaesenc (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0xdc,0x3a]      
vaesenc (%rdx), %xmm15, %xmm15 

// CHECK: vaesenc (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdc,0x32]      
vaesenc (%rdx), %xmm6, %xmm6 

// CHECK: vaesenc %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0xdc,0xff]      
vaesenc %xmm15, %xmm15, %xmm15 

// CHECK: vaesenc %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0xdc,0xf6]      
vaesenc %xmm6, %xmm6, %xmm6 

// CHECK: vaesimc 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0xdb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vaesimc 485498096, %xmm15 

// CHECK: vaesimc 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vaesimc 485498096, %xmm6 

// CHECK: vaesimc 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0xdb,0x7c,0x82,0x40]       
vaesimc 64(%rdx,%rax,4), %xmm15 

// CHECK: vaesimc -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0xdb,0x7c,0x82,0xc0]       
vaesimc -64(%rdx,%rax,4), %xmm15 

// CHECK: vaesimc 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0x74,0x82,0x40]       
vaesimc 64(%rdx,%rax,4), %xmm6 

// CHECK: vaesimc -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0x74,0x82,0xc0]       
vaesimc -64(%rdx,%rax,4), %xmm6 

// CHECK: vaesimc 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0xdb,0x7c,0x02,0x40]       
vaesimc 64(%rdx,%rax), %xmm15 

// CHECK: vaesimc 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0x74,0x02,0x40]       
vaesimc 64(%rdx,%rax), %xmm6 

// CHECK: vaesimc 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0xdb,0x7a,0x40]       
vaesimc 64(%rdx), %xmm15 

// CHECK: vaesimc 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0x72,0x40]       
vaesimc 64(%rdx), %xmm6 

// CHECK: vaesimc (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0xdb,0x3a]       
vaesimc (%rdx), %xmm15 

// CHECK: vaesimc (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0x32]       
vaesimc (%rdx), %xmm6 

// CHECK: vaesimc %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0xdb,0xff]       
vaesimc %xmm15, %xmm15 

// CHECK: vaesimc %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0xf6]       
vaesimc %xmm6, %xmm6 

// CHECK: vaeskeygenassist $0, 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0xdf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vaeskeygenassist $0, 485498096, %xmm15 

// CHECK: vaeskeygenassist $0, 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vaeskeygenassist $0, 485498096, %xmm6 

// CHECK: vaeskeygenassist $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0xdf,0x7c,0x82,0x40,0x00]      
vaeskeygenassist $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: vaeskeygenassist $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0xdf,0x7c,0x82,0xc0,0x00]      
vaeskeygenassist $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: vaeskeygenassist $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0x74,0x82,0x40,0x00]      
vaeskeygenassist $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: vaeskeygenassist $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0x74,0x82,0xc0,0x00]      
vaeskeygenassist $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: vaeskeygenassist $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0xdf,0x7c,0x02,0x40,0x00]      
vaeskeygenassist $0, 64(%rdx,%rax), %xmm15 

// CHECK: vaeskeygenassist $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0x74,0x02,0x40,0x00]      
vaeskeygenassist $0, 64(%rdx,%rax), %xmm6 

// CHECK: vaeskeygenassist $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0xdf,0x7a,0x40,0x00]      
vaeskeygenassist $0, 64(%rdx), %xmm15 

// CHECK: vaeskeygenassist $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0x72,0x40,0x00]      
vaeskeygenassist $0, 64(%rdx), %xmm6 

// CHECK: vaeskeygenassist $0, (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x63,0x79,0xdf,0x3a,0x00]      
vaeskeygenassist $0, (%rdx), %xmm15 

// CHECK: vaeskeygenassist $0, (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0x32,0x00]      
vaeskeygenassist $0, (%rdx), %xmm6 

// CHECK: vaeskeygenassist $0, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x79,0xdf,0xff,0x00]      
vaeskeygenassist $0, %xmm15, %xmm15 

// CHECK: vaeskeygenassist $0, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0xf6,0x00]      
vaeskeygenassist $0, %xmm6, %xmm6 

