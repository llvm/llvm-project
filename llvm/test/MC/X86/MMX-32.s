// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: emms 
// CHECK: encoding: [0x0f,0x77]          
emms 

// CHECK: maskmovq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf7,0xe4]        
maskmovq %mm4, %mm4 

// CHECK: movd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
movd -485498096(%edx,%eax,4), %mm4 

// CHECK: movd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
movd 485498096(%edx,%eax,4), %mm4 

// CHECK: movd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x6e,0xa2,0xf0,0x1c,0xf0,0x1c]        
movd 485498096(%edx), %mm4 

// CHECK: movd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6e,0x25,0xf0,0x1c,0xf0,0x1c]        
movd 485498096, %mm4 

// CHECK: movd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x6e,0x64,0x02,0x40]        
movd 64(%edx,%eax), %mm4 

// CHECK: movd (%edx), %mm4 
// CHECK: encoding: [0x0f,0x6e,0x22]        
movd (%edx), %mm4 

// CHECK: movd %mm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x7e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
movd %mm4, -485498096(%edx,%eax,4) 

// CHECK: movd %mm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x7e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
movd %mm4, 485498096(%edx,%eax,4) 

// CHECK: movd %mm4, 485498096(%edx) 
// CHECK: encoding: [0x0f,0x7e,0xa2,0xf0,0x1c,0xf0,0x1c]        
movd %mm4, 485498096(%edx) 

// CHECK: movd %mm4, 485498096 
// CHECK: encoding: [0x0f,0x7e,0x25,0xf0,0x1c,0xf0,0x1c]        
movd %mm4, 485498096 

// CHECK: movd %mm4, 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x7e,0x64,0x02,0x40]        
movd %mm4, 64(%edx,%eax) 

// CHECK: movd %mm4, (%edx) 
// CHECK: encoding: [0x0f,0x7e,0x22]        
movd %mm4, (%edx) 

// CHECK: movntq %mm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xe7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
movntq %mm4, -485498096(%edx,%eax,4) 

// CHECK: movntq %mm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xe7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
movntq %mm4, 485498096(%edx,%eax,4) 

// CHECK: movntq %mm4, 485498096(%edx) 
// CHECK: encoding: [0x0f,0xe7,0xa2,0xf0,0x1c,0xf0,0x1c]        
movntq %mm4, 485498096(%edx) 

// CHECK: movntq %mm4, 485498096 
// CHECK: encoding: [0x0f,0xe7,0x25,0xf0,0x1c,0xf0,0x1c]        
movntq %mm4, 485498096 

// CHECK: movntq %mm4, 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xe7,0x64,0x02,0x40]        
movntq %mm4, 64(%edx,%eax) 

// CHECK: movntq %mm4, (%edx) 
// CHECK: encoding: [0x0f,0xe7,0x22]        
movntq %mm4, (%edx) 

// CHECK: movq -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
movq -485498096(%edx,%eax,4), %mm4 

// CHECK: movq 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
movq 485498096(%edx,%eax,4), %mm4 

// CHECK: movq 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x6f,0xa2,0xf0,0x1c,0xf0,0x1c]        
movq 485498096(%edx), %mm4 

// CHECK: movq 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6f,0x25,0xf0,0x1c,0xf0,0x1c]        
movq 485498096, %mm4 

// CHECK: movq 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x6f,0x64,0x02,0x40]        
movq 64(%edx,%eax), %mm4 

// CHECK: movq (%edx), %mm4 
// CHECK: encoding: [0x0f,0x6f,0x22]        
movq (%edx), %mm4 

// CHECK: movq %mm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x7f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
movq %mm4, -485498096(%edx,%eax,4) 

// CHECK: movq %mm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x7f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
movq %mm4, 485498096(%edx,%eax,4) 

// CHECK: movq %mm4, 485498096(%edx) 
// CHECK: encoding: [0x0f,0x7f,0xa2,0xf0,0x1c,0xf0,0x1c]        
movq %mm4, 485498096(%edx) 

// CHECK: movq %mm4, 485498096 
// CHECK: encoding: [0x0f,0x7f,0x25,0xf0,0x1c,0xf0,0x1c]        
movq %mm4, 485498096 

// CHECK: movq %mm4, 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x7f,0x64,0x02,0x40]        
movq %mm4, 64(%edx,%eax) 

// CHECK: movq %mm4, (%edx) 
// CHECK: encoding: [0x0f,0x7f,0x22]        
movq %mm4, (%edx) 

// CHECK: movq %mm4, %mm4 
// CHECK: encoding: [0x0f,0x6f,0xe4]        
movq %mm4, %mm4 

// CHECK: packssdw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
packssdw -485498096(%edx,%eax,4), %mm4 

// CHECK: packssdw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
packssdw 485498096(%edx,%eax,4), %mm4 

// CHECK: packssdw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x6b,0xa2,0xf0,0x1c,0xf0,0x1c]        
packssdw 485498096(%edx), %mm4 

// CHECK: packssdw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6b,0x25,0xf0,0x1c,0xf0,0x1c]        
packssdw 485498096, %mm4 

// CHECK: packssdw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x6b,0x64,0x02,0x40]        
packssdw 64(%edx,%eax), %mm4 

// CHECK: packssdw (%edx), %mm4 
// CHECK: encoding: [0x0f,0x6b,0x22]        
packssdw (%edx), %mm4 

// CHECK: packssdw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x6b,0xe4]        
packssdw %mm4, %mm4 

// CHECK: packsswb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x63,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
packsswb -485498096(%edx,%eax,4), %mm4 

// CHECK: packsswb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x63,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
packsswb 485498096(%edx,%eax,4), %mm4 

// CHECK: packsswb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x63,0xa2,0xf0,0x1c,0xf0,0x1c]        
packsswb 485498096(%edx), %mm4 

// CHECK: packsswb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x63,0x25,0xf0,0x1c,0xf0,0x1c]        
packsswb 485498096, %mm4 

// CHECK: packsswb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x63,0x64,0x02,0x40]        
packsswb 64(%edx,%eax), %mm4 

// CHECK: packsswb (%edx), %mm4 
// CHECK: encoding: [0x0f,0x63,0x22]        
packsswb (%edx), %mm4 

// CHECK: packsswb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x63,0xe4]        
packsswb %mm4, %mm4 

// CHECK: packuswb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x67,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
packuswb -485498096(%edx,%eax,4), %mm4 

// CHECK: packuswb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x67,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
packuswb 485498096(%edx,%eax,4), %mm4 

// CHECK: packuswb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x67,0xa2,0xf0,0x1c,0xf0,0x1c]        
packuswb 485498096(%edx), %mm4 

// CHECK: packuswb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x67,0x25,0xf0,0x1c,0xf0,0x1c]        
packuswb 485498096, %mm4 

// CHECK: packuswb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x67,0x64,0x02,0x40]        
packuswb 64(%edx,%eax), %mm4 

// CHECK: packuswb (%edx), %mm4 
// CHECK: encoding: [0x0f,0x67,0x22]        
packuswb (%edx), %mm4 

// CHECK: packuswb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x67,0xe4]        
packuswb %mm4, %mm4 

// CHECK: paddb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfc,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddb -485498096(%edx,%eax,4), %mm4 

// CHECK: paddb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfc,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddb 485498096(%edx,%eax,4), %mm4 

// CHECK: paddb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xfc,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddb 485498096(%edx), %mm4 

// CHECK: paddb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfc,0x25,0xf0,0x1c,0xf0,0x1c]        
paddb 485498096, %mm4 

// CHECK: paddb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xfc,0x64,0x02,0x40]        
paddb 64(%edx,%eax), %mm4 

// CHECK: paddb (%edx), %mm4 
// CHECK: encoding: [0x0f,0xfc,0x22]        
paddb (%edx), %mm4 

// CHECK: paddb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfc,0xe4]        
paddb %mm4, %mm4 

// CHECK: paddd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfe,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddd -485498096(%edx,%eax,4), %mm4 

// CHECK: paddd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfe,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddd 485498096(%edx,%eax,4), %mm4 

// CHECK: paddd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xfe,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddd 485498096(%edx), %mm4 

// CHECK: paddd 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfe,0x25,0xf0,0x1c,0xf0,0x1c]        
paddd 485498096, %mm4 

// CHECK: paddd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xfe,0x64,0x02,0x40]        
paddd 64(%edx,%eax), %mm4 

// CHECK: paddd (%edx), %mm4 
// CHECK: encoding: [0x0f,0xfe,0x22]        
paddd (%edx), %mm4 

// CHECK: paddd %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfe,0xe4]        
paddd %mm4, %mm4 

// CHECK: paddq -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd4,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddq -485498096(%edx,%eax,4), %mm4 

// CHECK: paddq 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd4,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddq 485498096(%edx,%eax,4), %mm4 

// CHECK: paddq 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xd4,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddq 485498096(%edx), %mm4 

// CHECK: paddq 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd4,0x25,0xf0,0x1c,0xf0,0x1c]        
paddq 485498096, %mm4 

// CHECK: paddq 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xd4,0x64,0x02,0x40]        
paddq 64(%edx,%eax), %mm4 

// CHECK: paddq (%edx), %mm4 
// CHECK: encoding: [0x0f,0xd4,0x22]        
paddq (%edx), %mm4 

// CHECK: paddq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd4,0xe4]        
paddq %mm4, %mm4 

// CHECK: paddsb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xec,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddsb -485498096(%edx,%eax,4), %mm4 

// CHECK: paddsb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xec,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddsb 485498096(%edx,%eax,4), %mm4 

// CHECK: paddsb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xec,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddsb 485498096(%edx), %mm4 

// CHECK: paddsb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xec,0x25,0xf0,0x1c,0xf0,0x1c]        
paddsb 485498096, %mm4 

// CHECK: paddsb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xec,0x64,0x02,0x40]        
paddsb 64(%edx,%eax), %mm4 

// CHECK: paddsb (%edx), %mm4 
// CHECK: encoding: [0x0f,0xec,0x22]        
paddsb (%edx), %mm4 

// CHECK: paddsb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xec,0xe4]        
paddsb %mm4, %mm4 

// CHECK: paddsw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xed,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddsw -485498096(%edx,%eax,4), %mm4 

// CHECK: paddsw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xed,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddsw 485498096(%edx,%eax,4), %mm4 

// CHECK: paddsw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xed,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddsw 485498096(%edx), %mm4 

// CHECK: paddsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xed,0x25,0xf0,0x1c,0xf0,0x1c]        
paddsw 485498096, %mm4 

// CHECK: paddsw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xed,0x64,0x02,0x40]        
paddsw 64(%edx,%eax), %mm4 

// CHECK: paddsw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xed,0x22]        
paddsw (%edx), %mm4 

// CHECK: paddsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xed,0xe4]        
paddsw %mm4, %mm4 

// CHECK: paddusb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdc,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddusb -485498096(%edx,%eax,4), %mm4 

// CHECK: paddusb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdc,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddusb 485498096(%edx,%eax,4), %mm4 

// CHECK: paddusb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xdc,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddusb 485498096(%edx), %mm4 

// CHECK: paddusb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdc,0x25,0xf0,0x1c,0xf0,0x1c]        
paddusb 485498096, %mm4 

// CHECK: paddusb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xdc,0x64,0x02,0x40]        
paddusb 64(%edx,%eax), %mm4 

// CHECK: paddusb (%edx), %mm4 
// CHECK: encoding: [0x0f,0xdc,0x22]        
paddusb (%edx), %mm4 

// CHECK: paddusb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdc,0xe4]        
paddusb %mm4, %mm4 

// CHECK: paddusw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdd,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddusw -485498096(%edx,%eax,4), %mm4 

// CHECK: paddusw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdd,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddusw 485498096(%edx,%eax,4), %mm4 

// CHECK: paddusw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xdd,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddusw 485498096(%edx), %mm4 

// CHECK: paddusw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdd,0x25,0xf0,0x1c,0xf0,0x1c]        
paddusw 485498096, %mm4 

// CHECK: paddusw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xdd,0x64,0x02,0x40]        
paddusw 64(%edx,%eax), %mm4 

// CHECK: paddusw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xdd,0x22]        
paddusw (%edx), %mm4 

// CHECK: paddusw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdd,0xe4]        
paddusw %mm4, %mm4 

// CHECK: paddw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfd,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
paddw -485498096(%edx,%eax,4), %mm4 

// CHECK: paddw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfd,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
paddw 485498096(%edx,%eax,4), %mm4 

// CHECK: paddw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xfd,0xa2,0xf0,0x1c,0xf0,0x1c]        
paddw 485498096(%edx), %mm4 

// CHECK: paddw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfd,0x25,0xf0,0x1c,0xf0,0x1c]        
paddw 485498096, %mm4 

// CHECK: paddw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xfd,0x64,0x02,0x40]        
paddw 64(%edx,%eax), %mm4 

// CHECK: paddw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xfd,0x22]        
paddw (%edx), %mm4 

// CHECK: paddw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfd,0xe4]        
paddw %mm4, %mm4 

// CHECK: pand -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdb,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pand -485498096(%edx,%eax,4), %mm4 

// CHECK: pand 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdb,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pand 485498096(%edx,%eax,4), %mm4 

// CHECK: pand 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xdb,0xa2,0xf0,0x1c,0xf0,0x1c]        
pand 485498096(%edx), %mm4 

// CHECK: pand 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdb,0x25,0xf0,0x1c,0xf0,0x1c]        
pand 485498096, %mm4 

// CHECK: pand 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xdb,0x64,0x02,0x40]        
pand 64(%edx,%eax), %mm4 

// CHECK: pand (%edx), %mm4 
// CHECK: encoding: [0x0f,0xdb,0x22]        
pand (%edx), %mm4 

// CHECK: pand %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdb,0xe4]        
pand %mm4, %mm4 

// CHECK: pandn -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdf,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pandn -485498096(%edx,%eax,4), %mm4 

// CHECK: pandn 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xdf,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pandn 485498096(%edx,%eax,4), %mm4 

// CHECK: pandn 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xdf,0xa2,0xf0,0x1c,0xf0,0x1c]        
pandn 485498096(%edx), %mm4 

// CHECK: pandn 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdf,0x25,0xf0,0x1c,0xf0,0x1c]        
pandn 485498096, %mm4 

// CHECK: pandn 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xdf,0x64,0x02,0x40]        
pandn 64(%edx,%eax), %mm4 

// CHECK: pandn (%edx), %mm4 
// CHECK: encoding: [0x0f,0xdf,0x22]        
pandn (%edx), %mm4 

// CHECK: pandn %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdf,0xe4]        
pandn %mm4, %mm4 

// CHECK: pavgb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe0,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pavgb -485498096(%edx,%eax,4), %mm4 

// CHECK: pavgb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe0,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pavgb 485498096(%edx,%eax,4), %mm4 

// CHECK: pavgb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe0,0xa2,0xf0,0x1c,0xf0,0x1c]        
pavgb 485498096(%edx), %mm4 

// CHECK: pavgb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe0,0x25,0xf0,0x1c,0xf0,0x1c]        
pavgb 485498096, %mm4 

// CHECK: pavgb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe0,0x64,0x02,0x40]        
pavgb 64(%edx,%eax), %mm4 

// CHECK: pavgb (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe0,0x22]        
pavgb (%edx), %mm4 

// CHECK: pavgb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe0,0xe4]        
pavgb %mm4, %mm4 

// CHECK: pavgw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pavgw -485498096(%edx,%eax,4), %mm4 

// CHECK: pavgw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pavgw 485498096(%edx,%eax,4), %mm4 

// CHECK: pavgw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe3,0xa2,0xf0,0x1c,0xf0,0x1c]        
pavgw 485498096(%edx), %mm4 

// CHECK: pavgw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe3,0x25,0xf0,0x1c,0xf0,0x1c]        
pavgw 485498096, %mm4 

// CHECK: pavgw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe3,0x64,0x02,0x40]        
pavgw 64(%edx,%eax), %mm4 

// CHECK: pavgw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe3,0x22]        
pavgw (%edx), %mm4 

// CHECK: pavgw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe3,0xe4]        
pavgw %mm4, %mm4 

// CHECK: pcmpeqb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x74,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pcmpeqb -485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpeqb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x74,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pcmpeqb 485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpeqb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x74,0xa2,0xf0,0x1c,0xf0,0x1c]        
pcmpeqb 485498096(%edx), %mm4 

// CHECK: pcmpeqb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x74,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpeqb 485498096, %mm4 

// CHECK: pcmpeqb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x74,0x64,0x02,0x40]        
pcmpeqb 64(%edx,%eax), %mm4 

// CHECK: pcmpeqb (%edx), %mm4 
// CHECK: encoding: [0x0f,0x74,0x22]        
pcmpeqb (%edx), %mm4 

// CHECK: pcmpeqb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x74,0xe4]        
pcmpeqb %mm4, %mm4 

// CHECK: pcmpeqd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x76,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pcmpeqd -485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpeqd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x76,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pcmpeqd 485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpeqd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x76,0xa2,0xf0,0x1c,0xf0,0x1c]        
pcmpeqd 485498096(%edx), %mm4 

// CHECK: pcmpeqd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x76,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpeqd 485498096, %mm4 

// CHECK: pcmpeqd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x76,0x64,0x02,0x40]        
pcmpeqd 64(%edx,%eax), %mm4 

// CHECK: pcmpeqd (%edx), %mm4 
// CHECK: encoding: [0x0f,0x76,0x22]        
pcmpeqd (%edx), %mm4 

// CHECK: pcmpeqd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x76,0xe4]        
pcmpeqd %mm4, %mm4 

// CHECK: pcmpeqw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x75,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pcmpeqw -485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpeqw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x75,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pcmpeqw 485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpeqw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x75,0xa2,0xf0,0x1c,0xf0,0x1c]        
pcmpeqw 485498096(%edx), %mm4 

// CHECK: pcmpeqw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x75,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpeqw 485498096, %mm4 

// CHECK: pcmpeqw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x75,0x64,0x02,0x40]        
pcmpeqw 64(%edx,%eax), %mm4 

// CHECK: pcmpeqw (%edx), %mm4 
// CHECK: encoding: [0x0f,0x75,0x22]        
pcmpeqw (%edx), %mm4 

// CHECK: pcmpeqw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x75,0xe4]        
pcmpeqw %mm4, %mm4 

// CHECK: pcmpgtb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x64,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pcmpgtb -485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpgtb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x64,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pcmpgtb 485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpgtb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x64,0xa2,0xf0,0x1c,0xf0,0x1c]        
pcmpgtb 485498096(%edx), %mm4 

// CHECK: pcmpgtb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x64,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpgtb 485498096, %mm4 

// CHECK: pcmpgtb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x64,0x64,0x02,0x40]        
pcmpgtb 64(%edx,%eax), %mm4 

// CHECK: pcmpgtb (%edx), %mm4 
// CHECK: encoding: [0x0f,0x64,0x22]        
pcmpgtb (%edx), %mm4 

// CHECK: pcmpgtb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x64,0xe4]        
pcmpgtb %mm4, %mm4 

// CHECK: pcmpgtd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x66,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pcmpgtd -485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpgtd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x66,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pcmpgtd 485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpgtd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x66,0xa2,0xf0,0x1c,0xf0,0x1c]        
pcmpgtd 485498096(%edx), %mm4 

// CHECK: pcmpgtd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x66,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpgtd 485498096, %mm4 

// CHECK: pcmpgtd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x66,0x64,0x02,0x40]        
pcmpgtd 64(%edx,%eax), %mm4 

// CHECK: pcmpgtd (%edx), %mm4 
// CHECK: encoding: [0x0f,0x66,0x22]        
pcmpgtd (%edx), %mm4 

// CHECK: pcmpgtd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x66,0xe4]        
pcmpgtd %mm4, %mm4 

// CHECK: pcmpgtw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x65,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pcmpgtw -485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpgtw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x65,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pcmpgtw 485498096(%edx,%eax,4), %mm4 

// CHECK: pcmpgtw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x65,0xa2,0xf0,0x1c,0xf0,0x1c]        
pcmpgtw 485498096(%edx), %mm4 

// CHECK: pcmpgtw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x65,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpgtw 485498096, %mm4 

// CHECK: pcmpgtw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x65,0x64,0x02,0x40]        
pcmpgtw 64(%edx,%eax), %mm4 

// CHECK: pcmpgtw (%edx), %mm4 
// CHECK: encoding: [0x0f,0x65,0x22]        
pcmpgtw (%edx), %mm4 

// CHECK: pcmpgtw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x65,0xe4]        
pcmpgtw %mm4, %mm4 

// CHECK: pinsrw $0, -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xc4,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]       
pinsrw $0, -485498096(%edx,%eax,4), %mm4 

// CHECK: pinsrw $0, 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xc4,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]       
pinsrw $0, 485498096(%edx,%eax,4), %mm4 

// CHECK: pinsrw $0, 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xc4,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]       
pinsrw $0, 485498096(%edx), %mm4 

// CHECK: pinsrw $0, 485498096, %mm4 
// CHECK: encoding: [0x0f,0xc4,0x25,0xf0,0x1c,0xf0,0x1c,0x00]       
pinsrw $0, 485498096, %mm4 

// CHECK: pinsrw $0, 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xc4,0x64,0x02,0x40,0x00]       
pinsrw $0, 64(%edx,%eax), %mm4 

// CHECK: pinsrw $0, (%edx), %mm4 
// CHECK: encoding: [0x0f,0xc4,0x22,0x00]       
pinsrw $0, (%edx), %mm4 

// CHECK: pmaddwd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf5,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pmaddwd -485498096(%edx,%eax,4), %mm4 

// CHECK: pmaddwd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf5,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pmaddwd 485498096(%edx,%eax,4), %mm4 

// CHECK: pmaddwd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xf5,0xa2,0xf0,0x1c,0xf0,0x1c]        
pmaddwd 485498096(%edx), %mm4 

// CHECK: pmaddwd 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf5,0x25,0xf0,0x1c,0xf0,0x1c]        
pmaddwd 485498096, %mm4 

// CHECK: pmaddwd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xf5,0x64,0x02,0x40]        
pmaddwd 64(%edx,%eax), %mm4 

// CHECK: pmaddwd (%edx), %mm4 
// CHECK: encoding: [0x0f,0xf5,0x22]        
pmaddwd (%edx), %mm4 

// CHECK: pmaddwd %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf5,0xe4]        
pmaddwd %mm4, %mm4 

// CHECK: pmaxsw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xee,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pmaxsw -485498096(%edx,%eax,4), %mm4 

// CHECK: pmaxsw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xee,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pmaxsw 485498096(%edx,%eax,4), %mm4 

// CHECK: pmaxsw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xee,0xa2,0xf0,0x1c,0xf0,0x1c]        
pmaxsw 485498096(%edx), %mm4 

// CHECK: pmaxsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xee,0x25,0xf0,0x1c,0xf0,0x1c]        
pmaxsw 485498096, %mm4 

// CHECK: pmaxsw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xee,0x64,0x02,0x40]        
pmaxsw 64(%edx,%eax), %mm4 

// CHECK: pmaxsw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xee,0x22]        
pmaxsw (%edx), %mm4 

// CHECK: pmaxsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xee,0xe4]        
pmaxsw %mm4, %mm4 

// CHECK: pmaxub -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xde,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pmaxub -485498096(%edx,%eax,4), %mm4 

// CHECK: pmaxub 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xde,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pmaxub 485498096(%edx,%eax,4), %mm4 

// CHECK: pmaxub 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xde,0xa2,0xf0,0x1c,0xf0,0x1c]        
pmaxub 485498096(%edx), %mm4 

// CHECK: pmaxub 485498096, %mm4 
// CHECK: encoding: [0x0f,0xde,0x25,0xf0,0x1c,0xf0,0x1c]        
pmaxub 485498096, %mm4 

// CHECK: pmaxub 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xde,0x64,0x02,0x40]        
pmaxub 64(%edx,%eax), %mm4 

// CHECK: pmaxub (%edx), %mm4 
// CHECK: encoding: [0x0f,0xde,0x22]        
pmaxub (%edx), %mm4 

// CHECK: pmaxub %mm4, %mm4 
// CHECK: encoding: [0x0f,0xde,0xe4]        
pmaxub %mm4, %mm4 

// CHECK: pminsw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xea,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pminsw -485498096(%edx,%eax,4), %mm4 

// CHECK: pminsw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xea,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pminsw 485498096(%edx,%eax,4), %mm4 

// CHECK: pminsw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xea,0xa2,0xf0,0x1c,0xf0,0x1c]        
pminsw 485498096(%edx), %mm4 

// CHECK: pminsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xea,0x25,0xf0,0x1c,0xf0,0x1c]        
pminsw 485498096, %mm4 

// CHECK: pminsw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xea,0x64,0x02,0x40]        
pminsw 64(%edx,%eax), %mm4 

// CHECK: pminsw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xea,0x22]        
pminsw (%edx), %mm4 

// CHECK: pminsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xea,0xe4]        
pminsw %mm4, %mm4 

// CHECK: pminub -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xda,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pminub -485498096(%edx,%eax,4), %mm4 

// CHECK: pminub 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xda,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pminub 485498096(%edx,%eax,4), %mm4 

// CHECK: pminub 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xda,0xa2,0xf0,0x1c,0xf0,0x1c]        
pminub 485498096(%edx), %mm4 

// CHECK: pminub 485498096, %mm4 
// CHECK: encoding: [0x0f,0xda,0x25,0xf0,0x1c,0xf0,0x1c]        
pminub 485498096, %mm4 

// CHECK: pminub 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xda,0x64,0x02,0x40]        
pminub 64(%edx,%eax), %mm4 

// CHECK: pminub (%edx), %mm4 
// CHECK: encoding: [0x0f,0xda,0x22]        
pminub (%edx), %mm4 

// CHECK: pminub %mm4, %mm4 
// CHECK: encoding: [0x0f,0xda,0xe4]        
pminub %mm4, %mm4 

// CHECK: pmulhuw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe4,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pmulhuw -485498096(%edx,%eax,4), %mm4 

// CHECK: pmulhuw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe4,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pmulhuw 485498096(%edx,%eax,4), %mm4 

// CHECK: pmulhuw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe4,0xa2,0xf0,0x1c,0xf0,0x1c]        
pmulhuw 485498096(%edx), %mm4 

// CHECK: pmulhuw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe4,0x25,0xf0,0x1c,0xf0,0x1c]        
pmulhuw 485498096, %mm4 

// CHECK: pmulhuw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe4,0x64,0x02,0x40]        
pmulhuw 64(%edx,%eax), %mm4 

// CHECK: pmulhuw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe4,0x22]        
pmulhuw (%edx), %mm4 

// CHECK: pmulhuw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe4,0xe4]        
pmulhuw %mm4, %mm4 

// CHECK: pmulhw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe5,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pmulhw -485498096(%edx,%eax,4), %mm4 

// CHECK: pmulhw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe5,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pmulhw 485498096(%edx,%eax,4), %mm4 

// CHECK: pmulhw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe5,0xa2,0xf0,0x1c,0xf0,0x1c]        
pmulhw 485498096(%edx), %mm4 

// CHECK: pmulhw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe5,0x25,0xf0,0x1c,0xf0,0x1c]        
pmulhw 485498096, %mm4 

// CHECK: pmulhw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe5,0x64,0x02,0x40]        
pmulhw 64(%edx,%eax), %mm4 

// CHECK: pmulhw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe5,0x22]        
pmulhw (%edx), %mm4 

// CHECK: pmulhw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe5,0xe4]        
pmulhw %mm4, %mm4 

// CHECK: pmullw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd5,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pmullw -485498096(%edx,%eax,4), %mm4 

// CHECK: pmullw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd5,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pmullw 485498096(%edx,%eax,4), %mm4 

// CHECK: pmullw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xd5,0xa2,0xf0,0x1c,0xf0,0x1c]        
pmullw 485498096(%edx), %mm4 

// CHECK: pmullw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd5,0x25,0xf0,0x1c,0xf0,0x1c]        
pmullw 485498096, %mm4 

// CHECK: pmullw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xd5,0x64,0x02,0x40]        
pmullw 64(%edx,%eax), %mm4 

// CHECK: pmullw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xd5,0x22]        
pmullw (%edx), %mm4 

// CHECK: pmullw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd5,0xe4]        
pmullw %mm4, %mm4 

// CHECK: por -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xeb,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
por -485498096(%edx,%eax,4), %mm4 

// CHECK: por 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xeb,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
por 485498096(%edx,%eax,4), %mm4 

// CHECK: por 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xeb,0xa2,0xf0,0x1c,0xf0,0x1c]        
por 485498096(%edx), %mm4 

// CHECK: por 485498096, %mm4 
// CHECK: encoding: [0x0f,0xeb,0x25,0xf0,0x1c,0xf0,0x1c]        
por 485498096, %mm4 

// CHECK: por 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xeb,0x64,0x02,0x40]        
por 64(%edx,%eax), %mm4 

// CHECK: por (%edx), %mm4 
// CHECK: encoding: [0x0f,0xeb,0x22]        
por (%edx), %mm4 

// CHECK: por %mm4, %mm4 
// CHECK: encoding: [0x0f,0xeb,0xe4]        
por %mm4, %mm4 

// CHECK: psadbw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psadbw -485498096(%edx,%eax,4), %mm4 

// CHECK: psadbw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psadbw 485498096(%edx,%eax,4), %mm4 

// CHECK: psadbw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xf6,0xa2,0xf0,0x1c,0xf0,0x1c]        
psadbw 485498096(%edx), %mm4 

// CHECK: psadbw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf6,0x25,0xf0,0x1c,0xf0,0x1c]        
psadbw 485498096, %mm4 

// CHECK: psadbw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xf6,0x64,0x02,0x40]        
psadbw 64(%edx,%eax), %mm4 

// CHECK: psadbw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xf6,0x22]        
psadbw (%edx), %mm4 

// CHECK: psadbw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf6,0xe4]        
psadbw %mm4, %mm4 

// CHECK: pshufw $0, -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x70,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]       
pshufw $0, -485498096(%edx,%eax,4), %mm4 

// CHECK: pshufw $0, 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x70,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]       
pshufw $0, 485498096(%edx,%eax,4), %mm4 

// CHECK: pshufw $0, 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x70,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]       
pshufw $0, 485498096(%edx), %mm4 

// CHECK: pshufw $0, 485498096, %mm4 
// CHECK: encoding: [0x0f,0x70,0x25,0xf0,0x1c,0xf0,0x1c,0x00]       
pshufw $0, 485498096, %mm4 

// CHECK: pshufw $0, 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x70,0x64,0x02,0x40,0x00]       
pshufw $0, 64(%edx,%eax), %mm4 

// CHECK: pshufw $0, (%edx), %mm4 
// CHECK: encoding: [0x0f,0x70,0x22,0x00]       
pshufw $0, (%edx), %mm4 

// CHECK: pshufw $0, %mm4, %mm4 
// CHECK: encoding: [0x0f,0x70,0xe4,0x00]       
pshufw $0, %mm4, %mm4 

// CHECK: pslld $0, %mm4 
// CHECK: encoding: [0x0f,0x72,0xf4,0x00]        
pslld $0, %mm4 

// CHECK: pslld -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf2,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pslld -485498096(%edx,%eax,4), %mm4 

// CHECK: pslld 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pslld 485498096(%edx,%eax,4), %mm4 

// CHECK: pslld 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xf2,0xa2,0xf0,0x1c,0xf0,0x1c]        
pslld 485498096(%edx), %mm4 

// CHECK: pslld 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf2,0x25,0xf0,0x1c,0xf0,0x1c]        
pslld 485498096, %mm4 

// CHECK: pslld 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xf2,0x64,0x02,0x40]        
pslld 64(%edx,%eax), %mm4 

// CHECK: pslld (%edx), %mm4 
// CHECK: encoding: [0x0f,0xf2,0x22]        
pslld (%edx), %mm4 

// CHECK: pslld %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf2,0xe4]        
pslld %mm4, %mm4 

// CHECK: psllq $0, %mm4 
// CHECK: encoding: [0x0f,0x73,0xf4,0x00]        
psllq $0, %mm4 

// CHECK: psllq -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psllq -485498096(%edx,%eax,4), %mm4 

// CHECK: psllq 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psllq 485498096(%edx,%eax,4), %mm4 

// CHECK: psllq 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xf3,0xa2,0xf0,0x1c,0xf0,0x1c]        
psllq 485498096(%edx), %mm4 

// CHECK: psllq 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf3,0x25,0xf0,0x1c,0xf0,0x1c]        
psllq 485498096, %mm4 

// CHECK: psllq 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xf3,0x64,0x02,0x40]        
psllq 64(%edx,%eax), %mm4 

// CHECK: psllq (%edx), %mm4 
// CHECK: encoding: [0x0f,0xf3,0x22]        
psllq (%edx), %mm4 

// CHECK: psllq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf3,0xe4]        
psllq %mm4, %mm4 

// CHECK: psllw $0, %mm4 
// CHECK: encoding: [0x0f,0x71,0xf4,0x00]        
psllw $0, %mm4 

// CHECK: psllw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psllw -485498096(%edx,%eax,4), %mm4 

// CHECK: psllw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psllw 485498096(%edx,%eax,4), %mm4 

// CHECK: psllw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xf1,0xa2,0xf0,0x1c,0xf0,0x1c]        
psllw 485498096(%edx), %mm4 

// CHECK: psllw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf1,0x25,0xf0,0x1c,0xf0,0x1c]        
psllw 485498096, %mm4 

// CHECK: psllw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xf1,0x64,0x02,0x40]        
psllw 64(%edx,%eax), %mm4 

// CHECK: psllw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xf1,0x22]        
psllw (%edx), %mm4 

// CHECK: psllw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf1,0xe4]        
psllw %mm4, %mm4 

// CHECK: psrad $0, %mm4 
// CHECK: encoding: [0x0f,0x72,0xe4,0x00]        
psrad $0, %mm4 

// CHECK: psrad -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe2,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psrad -485498096(%edx,%eax,4), %mm4 

// CHECK: psrad 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psrad 485498096(%edx,%eax,4), %mm4 

// CHECK: psrad 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe2,0xa2,0xf0,0x1c,0xf0,0x1c]        
psrad 485498096(%edx), %mm4 

// CHECK: psrad 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe2,0x25,0xf0,0x1c,0xf0,0x1c]        
psrad 485498096, %mm4 

// CHECK: psrad 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe2,0x64,0x02,0x40]        
psrad 64(%edx,%eax), %mm4 

// CHECK: psrad (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe2,0x22]        
psrad (%edx), %mm4 

// CHECK: psrad %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe2,0xe4]        
psrad %mm4, %mm4 

// CHECK: psraw $0, %mm4 
// CHECK: encoding: [0x0f,0x71,0xe4,0x00]        
psraw $0, %mm4 

// CHECK: psraw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psraw -485498096(%edx,%eax,4), %mm4 

// CHECK: psraw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psraw 485498096(%edx,%eax,4), %mm4 

// CHECK: psraw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe1,0xa2,0xf0,0x1c,0xf0,0x1c]        
psraw 485498096(%edx), %mm4 

// CHECK: psraw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe1,0x25,0xf0,0x1c,0xf0,0x1c]        
psraw 485498096, %mm4 

// CHECK: psraw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe1,0x64,0x02,0x40]        
psraw 64(%edx,%eax), %mm4 

// CHECK: psraw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe1,0x22]        
psraw (%edx), %mm4 

// CHECK: psraw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe1,0xe4]        
psraw %mm4, %mm4 

// CHECK: psrld $0, %mm4 
// CHECK: encoding: [0x0f,0x72,0xd4,0x00]        
psrld $0, %mm4 

// CHECK: psrld -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd2,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psrld -485498096(%edx,%eax,4), %mm4 

// CHECK: psrld 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psrld 485498096(%edx,%eax,4), %mm4 

// CHECK: psrld 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xd2,0xa2,0xf0,0x1c,0xf0,0x1c]        
psrld 485498096(%edx), %mm4 

// CHECK: psrld 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd2,0x25,0xf0,0x1c,0xf0,0x1c]        
psrld 485498096, %mm4 

// CHECK: psrld 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xd2,0x64,0x02,0x40]        
psrld 64(%edx,%eax), %mm4 

// CHECK: psrld (%edx), %mm4 
// CHECK: encoding: [0x0f,0xd2,0x22]        
psrld (%edx), %mm4 

// CHECK: psrld %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd2,0xe4]        
psrld %mm4, %mm4 

// CHECK: psrlq $0, %mm4 
// CHECK: encoding: [0x0f,0x73,0xd4,0x00]        
psrlq $0, %mm4 

// CHECK: psrlq -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psrlq -485498096(%edx,%eax,4), %mm4 

// CHECK: psrlq 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psrlq 485498096(%edx,%eax,4), %mm4 

// CHECK: psrlq 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xd3,0xa2,0xf0,0x1c,0xf0,0x1c]        
psrlq 485498096(%edx), %mm4 

// CHECK: psrlq 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd3,0x25,0xf0,0x1c,0xf0,0x1c]        
psrlq 485498096, %mm4 

// CHECK: psrlq 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xd3,0x64,0x02,0x40]        
psrlq 64(%edx,%eax), %mm4 

// CHECK: psrlq (%edx), %mm4 
// CHECK: encoding: [0x0f,0xd3,0x22]        
psrlq (%edx), %mm4 

// CHECK: psrlq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd3,0xe4]        
psrlq %mm4, %mm4 

// CHECK: psrlw $0, %mm4 
// CHECK: encoding: [0x0f,0x71,0xd4,0x00]        
psrlw $0, %mm4 

// CHECK: psrlw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psrlw -485498096(%edx,%eax,4), %mm4 

// CHECK: psrlw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psrlw 485498096(%edx,%eax,4), %mm4 

// CHECK: psrlw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xd1,0xa2,0xf0,0x1c,0xf0,0x1c]        
psrlw 485498096(%edx), %mm4 

// CHECK: psrlw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd1,0x25,0xf0,0x1c,0xf0,0x1c]        
psrlw 485498096, %mm4 

// CHECK: psrlw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xd1,0x64,0x02,0x40]        
psrlw 64(%edx,%eax), %mm4 

// CHECK: psrlw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xd1,0x22]        
psrlw (%edx), %mm4 

// CHECK: psrlw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd1,0xe4]        
psrlw %mm4, %mm4 

// CHECK: psubb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubb -485498096(%edx,%eax,4), %mm4 

// CHECK: psubb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubb 485498096(%edx,%eax,4), %mm4 

// CHECK: psubb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xf8,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubb 485498096(%edx), %mm4 

// CHECK: psubb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf8,0x25,0xf0,0x1c,0xf0,0x1c]        
psubb 485498096, %mm4 

// CHECK: psubb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xf8,0x64,0x02,0x40]        
psubb 64(%edx,%eax), %mm4 

// CHECK: psubb (%edx), %mm4 
// CHECK: encoding: [0x0f,0xf8,0x22]        
psubb (%edx), %mm4 

// CHECK: psubb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf8,0xe4]        
psubb %mm4, %mm4 

// CHECK: psubd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfa,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubd -485498096(%edx,%eax,4), %mm4 

// CHECK: psubd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfa,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubd 485498096(%edx,%eax,4), %mm4 

// CHECK: psubd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xfa,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubd 485498096(%edx), %mm4 

// CHECK: psubd 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfa,0x25,0xf0,0x1c,0xf0,0x1c]        
psubd 485498096, %mm4 

// CHECK: psubd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xfa,0x64,0x02,0x40]        
psubd 64(%edx,%eax), %mm4 

// CHECK: psubd (%edx), %mm4 
// CHECK: encoding: [0x0f,0xfa,0x22]        
psubd (%edx), %mm4 

// CHECK: psubd %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfa,0xe4]        
psubd %mm4, %mm4 

// CHECK: psubq -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfb,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubq -485498096(%edx,%eax,4), %mm4 

// CHECK: psubq 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xfb,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubq 485498096(%edx,%eax,4), %mm4 

// CHECK: psubq 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xfb,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubq 485498096(%edx), %mm4 

// CHECK: psubq 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfb,0x25,0xf0,0x1c,0xf0,0x1c]        
psubq 485498096, %mm4 

// CHECK: psubq 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xfb,0x64,0x02,0x40]        
psubq 64(%edx,%eax), %mm4 

// CHECK: psubq (%edx), %mm4 
// CHECK: encoding: [0x0f,0xfb,0x22]        
psubq (%edx), %mm4 

// CHECK: psubq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfb,0xe4]        
psubq %mm4, %mm4 

// CHECK: psubsb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubsb -485498096(%edx,%eax,4), %mm4 

// CHECK: psubsb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubsb 485498096(%edx,%eax,4), %mm4 

// CHECK: psubsb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe8,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubsb 485498096(%edx), %mm4 

// CHECK: psubsb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe8,0x25,0xf0,0x1c,0xf0,0x1c]        
psubsb 485498096, %mm4 

// CHECK: psubsb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe8,0x64,0x02,0x40]        
psubsb 64(%edx,%eax), %mm4 

// CHECK: psubsb (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe8,0x22]        
psubsb (%edx), %mm4 

// CHECK: psubsb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe8,0xe4]        
psubsb %mm4, %mm4 

// CHECK: psubsw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe9,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubsw -485498096(%edx,%eax,4), %mm4 

// CHECK: psubsw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xe9,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubsw 485498096(%edx,%eax,4), %mm4 

// CHECK: psubsw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xe9,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubsw 485498096(%edx), %mm4 

// CHECK: psubsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe9,0x25,0xf0,0x1c,0xf0,0x1c]        
psubsw 485498096, %mm4 

// CHECK: psubsw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xe9,0x64,0x02,0x40]        
psubsw 64(%edx,%eax), %mm4 

// CHECK: psubsw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xe9,0x22]        
psubsw (%edx), %mm4 

// CHECK: psubsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe9,0xe4]        
psubsw %mm4, %mm4 

// CHECK: psubusb -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubusb -485498096(%edx,%eax,4), %mm4 

// CHECK: psubusb 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubusb 485498096(%edx,%eax,4), %mm4 

// CHECK: psubusb 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xd8,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubusb 485498096(%edx), %mm4 

// CHECK: psubusb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd8,0x25,0xf0,0x1c,0xf0,0x1c]        
psubusb 485498096, %mm4 

// CHECK: psubusb 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xd8,0x64,0x02,0x40]        
psubusb 64(%edx,%eax), %mm4 

// CHECK: psubusb (%edx), %mm4 
// CHECK: encoding: [0x0f,0xd8,0x22]        
psubusb (%edx), %mm4 

// CHECK: psubusb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd8,0xe4]        
psubusb %mm4, %mm4 

// CHECK: psubusw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd9,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubusw -485498096(%edx,%eax,4), %mm4 

// CHECK: psubusw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xd9,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubusw 485498096(%edx,%eax,4), %mm4 

// CHECK: psubusw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xd9,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubusw 485498096(%edx), %mm4 

// CHECK: psubusw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd9,0x25,0xf0,0x1c,0xf0,0x1c]        
psubusw 485498096, %mm4 

// CHECK: psubusw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xd9,0x64,0x02,0x40]        
psubusw 64(%edx,%eax), %mm4 

// CHECK: psubusw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xd9,0x22]        
psubusw (%edx), %mm4 

// CHECK: psubusw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd9,0xe4]        
psubusw %mm4, %mm4 

// CHECK: psubw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf9,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
psubw -485498096(%edx,%eax,4), %mm4 

// CHECK: psubw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xf9,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
psubw 485498096(%edx,%eax,4), %mm4 

// CHECK: psubw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xf9,0xa2,0xf0,0x1c,0xf0,0x1c]        
psubw 485498096(%edx), %mm4 

// CHECK: psubw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf9,0x25,0xf0,0x1c,0xf0,0x1c]        
psubw 485498096, %mm4 

// CHECK: psubw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xf9,0x64,0x02,0x40]        
psubw 64(%edx,%eax), %mm4 

// CHECK: psubw (%edx), %mm4 
// CHECK: encoding: [0x0f,0xf9,0x22]        
psubw (%edx), %mm4 

// CHECK: psubw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf9,0xe4]        
psubw %mm4, %mm4 

// CHECK: punpckhbw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x68,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
punpckhbw -485498096(%edx,%eax,4), %mm4 

// CHECK: punpckhbw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x68,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
punpckhbw 485498096(%edx,%eax,4), %mm4 

// CHECK: punpckhbw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x68,0xa2,0xf0,0x1c,0xf0,0x1c]        
punpckhbw 485498096(%edx), %mm4 

// CHECK: punpckhbw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x68,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckhbw 485498096, %mm4 

// CHECK: punpckhbw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x68,0x64,0x02,0x40]        
punpckhbw 64(%edx,%eax), %mm4 

// CHECK: punpckhbw (%edx), %mm4 
// CHECK: encoding: [0x0f,0x68,0x22]        
punpckhbw (%edx), %mm4 

// CHECK: punpckhbw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x68,0xe4]        
punpckhbw %mm4, %mm4 

// CHECK: punpckhdq -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
punpckhdq -485498096(%edx,%eax,4), %mm4 

// CHECK: punpckhdq 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x6a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
punpckhdq 485498096(%edx,%eax,4), %mm4 

// CHECK: punpckhdq 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x6a,0xa2,0xf0,0x1c,0xf0,0x1c]        
punpckhdq 485498096(%edx), %mm4 

// CHECK: punpckhdq 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6a,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckhdq 485498096, %mm4 

// CHECK: punpckhdq 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x6a,0x64,0x02,0x40]        
punpckhdq 64(%edx,%eax), %mm4 

// CHECK: punpckhdq (%edx), %mm4 
// CHECK: encoding: [0x0f,0x6a,0x22]        
punpckhdq (%edx), %mm4 

// CHECK: punpckhdq %mm4, %mm4 
// CHECK: encoding: [0x0f,0x6a,0xe4]        
punpckhdq %mm4, %mm4 

// CHECK: punpckhwd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x69,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
punpckhwd -485498096(%edx,%eax,4), %mm4 

// CHECK: punpckhwd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x69,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
punpckhwd 485498096(%edx,%eax,4), %mm4 

// CHECK: punpckhwd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x69,0xa2,0xf0,0x1c,0xf0,0x1c]        
punpckhwd 485498096(%edx), %mm4 

// CHECK: punpckhwd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x69,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckhwd 485498096, %mm4 

// CHECK: punpckhwd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x69,0x64,0x02,0x40]        
punpckhwd 64(%edx,%eax), %mm4 

// CHECK: punpckhwd (%edx), %mm4 
// CHECK: encoding: [0x0f,0x69,0x22]        
punpckhwd (%edx), %mm4 

// CHECK: punpckhwd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x69,0xe4]        
punpckhwd %mm4, %mm4 

// CHECK: punpcklbw -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x60,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
punpcklbw -485498096(%edx,%eax,4), %mm4 

// CHECK: punpcklbw 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x60,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
punpcklbw 485498096(%edx,%eax,4), %mm4 

// CHECK: punpcklbw 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x60,0xa2,0xf0,0x1c,0xf0,0x1c]        
punpcklbw 485498096(%edx), %mm4 

// CHECK: punpcklbw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x60,0x25,0xf0,0x1c,0xf0,0x1c]        
punpcklbw 485498096, %mm4 

// CHECK: punpcklbw 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x60,0x64,0x02,0x40]        
punpcklbw 64(%edx,%eax), %mm4 

// CHECK: punpcklbw (%edx), %mm4 
// CHECK: encoding: [0x0f,0x60,0x22]        
punpcklbw (%edx), %mm4 

// CHECK: punpcklbw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x60,0xe4]        
punpcklbw %mm4, %mm4 

// CHECK: punpckldq -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x62,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
punpckldq -485498096(%edx,%eax,4), %mm4 

// CHECK: punpckldq 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x62,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
punpckldq 485498096(%edx,%eax,4), %mm4 

// CHECK: punpckldq 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x62,0xa2,0xf0,0x1c,0xf0,0x1c]        
punpckldq 485498096(%edx), %mm4 

// CHECK: punpckldq 485498096, %mm4 
// CHECK: encoding: [0x0f,0x62,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckldq 485498096, %mm4 

// CHECK: punpckldq 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x62,0x64,0x02,0x40]        
punpckldq 64(%edx,%eax), %mm4 

// CHECK: punpckldq (%edx), %mm4 
// CHECK: encoding: [0x0f,0x62,0x22]        
punpckldq (%edx), %mm4 

// CHECK: punpckldq %mm4, %mm4 
// CHECK: encoding: [0x0f,0x62,0xe4]        
punpckldq %mm4, %mm4 

// CHECK: punpcklwd -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x61,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
punpcklwd -485498096(%edx,%eax,4), %mm4 

// CHECK: punpcklwd 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0x61,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
punpcklwd 485498096(%edx,%eax,4), %mm4 

// CHECK: punpcklwd 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0x61,0xa2,0xf0,0x1c,0xf0,0x1c]        
punpcklwd 485498096(%edx), %mm4 

// CHECK: punpcklwd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x61,0x25,0xf0,0x1c,0xf0,0x1c]        
punpcklwd 485498096, %mm4 

// CHECK: punpcklwd 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0x61,0x64,0x02,0x40]        
punpcklwd 64(%edx,%eax), %mm4 

// CHECK: punpcklwd (%edx), %mm4 
// CHECK: encoding: [0x0f,0x61,0x22]        
punpcklwd (%edx), %mm4 

// CHECK: punpcklwd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x61,0xe4]        
punpcklwd %mm4, %mm4 

// CHECK: pxor -485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xef,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
pxor -485498096(%edx,%eax,4), %mm4 

// CHECK: pxor 485498096(%edx,%eax,4), %mm4 
// CHECK: encoding: [0x0f,0xef,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
pxor 485498096(%edx,%eax,4), %mm4 

// CHECK: pxor 485498096(%edx), %mm4 
// CHECK: encoding: [0x0f,0xef,0xa2,0xf0,0x1c,0xf0,0x1c]        
pxor 485498096(%edx), %mm4 

// CHECK: pxor 485498096, %mm4 
// CHECK: encoding: [0x0f,0xef,0x25,0xf0,0x1c,0xf0,0x1c]        
pxor 485498096, %mm4 

// CHECK: pxor 64(%edx,%eax), %mm4 
// CHECK: encoding: [0x0f,0xef,0x64,0x02,0x40]        
pxor 64(%edx,%eax), %mm4 

// CHECK: pxor (%edx), %mm4 
// CHECK: encoding: [0x0f,0xef,0x22]        
pxor (%edx), %mm4 

// CHECK: pxor %mm4, %mm4 
// CHECK: encoding: [0x0f,0xef,0xe4]        
pxor %mm4, %mm4 

