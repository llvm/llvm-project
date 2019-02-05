// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: emms 
// CHECK: encoding: [0x0f,0x77]          
emms 

// CHECK: maskmovq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf7,0xe4]        
maskmovq %mm4, %mm4 

// CHECK: movd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6e,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
movd 485498096, %mm4 

// CHECK: movd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6e,0x62,0x40]        
movd 64(%rdx), %mm4 

// CHECK: movd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6e,0x64,0x82,0x40]        
movd 64(%rdx,%rax,4), %mm4 

// CHECK: movd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6e,0x64,0x82,0xc0]        
movd -64(%rdx,%rax,4), %mm4 

// CHECK: movd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x6e,0x64,0x02,0x40]        
movd 64(%rdx,%rax), %mm4 

// CHECK: movd %mm4, 485498096 
// CHECK: encoding: [0x0f,0x7e,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
movd %mm4, 485498096 

// CHECK: movd %mm4, 64(%rdx) 
// CHECK: encoding: [0x0f,0x7e,0x62,0x40]        
movd %mm4, 64(%rdx) 

// CHECK: movd %mm4, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x7e,0x64,0x82,0x40]        
movd %mm4, 64(%rdx,%rax,4) 

// CHECK: movd %mm4, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x7e,0x64,0x82,0xc0]        
movd %mm4, -64(%rdx,%rax,4) 

// CHECK: movd %mm4, 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x7e,0x64,0x02,0x40]        
movd %mm4, 64(%rdx,%rax) 

// CHECK: movd %mm4, %r13d 
// CHECK: encoding: [0x41,0x0f,0x7e,0xe5]        
movd %mm4, %r13d 

// CHECK: movd %mm4, %r15d 
// CHECK: encoding: [0x41,0x0f,0x7e,0xe7]        
movd %mm4, %r15d 

// CHECK: movd %mm4, (%rdx) 
// CHECK: encoding: [0x0f,0x7e,0x22]        
movd %mm4, (%rdx) 

// CHECK: movd %r13d, %mm4 
// CHECK: encoding: [0x41,0x0f,0x6e,0xe5]        
movd %r13d, %mm4 

// CHECK: movd %r15d, %mm4 
// CHECK: encoding: [0x41,0x0f,0x6e,0xe7]        
movd %r15d, %mm4 

// CHECK: movd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6e,0x22]        
movd (%rdx), %mm4 

// CHECK: movntq %mm4, 485498096 
// CHECK: encoding: [0x0f,0xe7,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
movntq %mm4, 485498096 

// CHECK: movntq %mm4, 64(%rdx) 
// CHECK: encoding: [0x0f,0xe7,0x62,0x40]        
movntq %mm4, 64(%rdx) 

// CHECK: movntq %mm4, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xe7,0x64,0x82,0x40]        
movntq %mm4, 64(%rdx,%rax,4) 

// CHECK: movntq %mm4, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xe7,0x64,0x82,0xc0]        
movntq %mm4, -64(%rdx,%rax,4) 

// CHECK: movntq %mm4, 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xe7,0x64,0x02,0x40]        
movntq %mm4, 64(%rdx,%rax) 

// CHECK: movntq %mm4, (%rdx) 
// CHECK: encoding: [0x0f,0xe7,0x22]        
movntq %mm4, (%rdx) 

// CHECK: movq 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6f,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
movq 485498096, %mm4 

// CHECK: movq 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6f,0x62,0x40]        
movq 64(%rdx), %mm4 

// CHECK: movq 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6f,0x64,0x82,0x40]        
movq 64(%rdx,%rax,4), %mm4 

// CHECK: movq -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6f,0x64,0x82,0xc0]        
movq -64(%rdx,%rax,4), %mm4 

// CHECK: movq 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x6f,0x64,0x02,0x40]        
movq 64(%rdx,%rax), %mm4 

// CHECK: movq %mm4, 485498096 
// CHECK: encoding: [0x0f,0x7f,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
movq %mm4, 485498096 

// CHECK: movq %mm4, 64(%rdx) 
// CHECK: encoding: [0x0f,0x7f,0x62,0x40]        
movq %mm4, 64(%rdx) 

// CHECK: movq %mm4, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x7f,0x64,0x82,0x40]        
movq %mm4, 64(%rdx,%rax,4) 

// CHECK: movq %mm4, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x7f,0x64,0x82,0xc0]        
movq %mm4, -64(%rdx,%rax,4) 

// CHECK: movq %mm4, 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x7f,0x64,0x02,0x40]        
movq %mm4, 64(%rdx,%rax) 

// CHECK: movq %mm4, %mm4 
// CHECK: encoding: [0x0f,0x6f,0xe4]        
movq %mm4, %mm4 

// CHECK: movq %mm4, (%rdx) 
// CHECK: encoding: [0x0f,0x7f,0x22]        
movq %mm4, (%rdx) 

// CHECK: movq (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6f,0x22]        
movq (%rdx), %mm4 

// CHECK: packssdw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6b,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
packssdw 485498096, %mm4 

// CHECK: packssdw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6b,0x62,0x40]        
packssdw 64(%rdx), %mm4 

// CHECK: packssdw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6b,0x64,0x82,0x40]        
packssdw 64(%rdx,%rax,4), %mm4 

// CHECK: packssdw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6b,0x64,0x82,0xc0]        
packssdw -64(%rdx,%rax,4), %mm4 

// CHECK: packssdw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x6b,0x64,0x02,0x40]        
packssdw 64(%rdx,%rax), %mm4 

// CHECK: packssdw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x6b,0xe4]        
packssdw %mm4, %mm4 

// CHECK: packssdw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6b,0x22]        
packssdw (%rdx), %mm4 

// CHECK: packsswb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x63,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
packsswb 485498096, %mm4 

// CHECK: packsswb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x63,0x62,0x40]        
packsswb 64(%rdx), %mm4 

// CHECK: packsswb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x63,0x64,0x82,0x40]        
packsswb 64(%rdx,%rax,4), %mm4 

// CHECK: packsswb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x63,0x64,0x82,0xc0]        
packsswb -64(%rdx,%rax,4), %mm4 

// CHECK: packsswb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x63,0x64,0x02,0x40]        
packsswb 64(%rdx,%rax), %mm4 

// CHECK: packsswb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x63,0xe4]        
packsswb %mm4, %mm4 

// CHECK: packsswb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x63,0x22]        
packsswb (%rdx), %mm4 

// CHECK: packuswb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x67,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
packuswb 485498096, %mm4 

// CHECK: packuswb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x67,0x62,0x40]        
packuswb 64(%rdx), %mm4 

// CHECK: packuswb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x67,0x64,0x82,0x40]        
packuswb 64(%rdx,%rax,4), %mm4 

// CHECK: packuswb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x67,0x64,0x82,0xc0]        
packuswb -64(%rdx,%rax,4), %mm4 

// CHECK: packuswb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x67,0x64,0x02,0x40]        
packuswb 64(%rdx,%rax), %mm4 

// CHECK: packuswb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x67,0xe4]        
packuswb %mm4, %mm4 

// CHECK: packuswb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x67,0x22]        
packuswb (%rdx), %mm4 

// CHECK: paddb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfc,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
paddb 485498096, %mm4 

// CHECK: paddb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfc,0x62,0x40]        
paddb 64(%rdx), %mm4 

// CHECK: paddb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfc,0x64,0x82,0x40]        
paddb 64(%rdx,%rax,4), %mm4 

// CHECK: paddb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfc,0x64,0x82,0xc0]        
paddb -64(%rdx,%rax,4), %mm4 

// CHECK: paddb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xfc,0x64,0x02,0x40]        
paddb 64(%rdx,%rax), %mm4 

// CHECK: paddb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfc,0xe4]        
paddb %mm4, %mm4 

// CHECK: paddb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfc,0x22]        
paddb (%rdx), %mm4 

// CHECK: paddd 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfe,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
paddd 485498096, %mm4 

// CHECK: paddd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfe,0x62,0x40]        
paddd 64(%rdx), %mm4 

// CHECK: paddd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfe,0x64,0x82,0x40]        
paddd 64(%rdx,%rax,4), %mm4 

// CHECK: paddd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfe,0x64,0x82,0xc0]        
paddd -64(%rdx,%rax,4), %mm4 

// CHECK: paddd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xfe,0x64,0x02,0x40]        
paddd 64(%rdx,%rax), %mm4 

// CHECK: paddd %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfe,0xe4]        
paddd %mm4, %mm4 

// CHECK: paddd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfe,0x22]        
paddd (%rdx), %mm4 

// CHECK: paddsb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xec,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
paddsb 485498096, %mm4 

// CHECK: paddsb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xec,0x62,0x40]        
paddsb 64(%rdx), %mm4 

// CHECK: paddsb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xec,0x64,0x82,0x40]        
paddsb 64(%rdx,%rax,4), %mm4 

// CHECK: paddsb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xec,0x64,0x82,0xc0]        
paddsb -64(%rdx,%rax,4), %mm4 

// CHECK: paddsb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xec,0x64,0x02,0x40]        
paddsb 64(%rdx,%rax), %mm4 

// CHECK: paddsb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xec,0xe4]        
paddsb %mm4, %mm4 

// CHECK: paddsb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xec,0x22]        
paddsb (%rdx), %mm4 

// CHECK: paddsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xed,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
paddsw 485498096, %mm4 

// CHECK: paddsw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xed,0x62,0x40]        
paddsw 64(%rdx), %mm4 

// CHECK: paddsw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xed,0x64,0x82,0x40]        
paddsw 64(%rdx,%rax,4), %mm4 

// CHECK: paddsw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xed,0x64,0x82,0xc0]        
paddsw -64(%rdx,%rax,4), %mm4 

// CHECK: paddsw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xed,0x64,0x02,0x40]        
paddsw 64(%rdx,%rax), %mm4 

// CHECK: paddsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xed,0xe4]        
paddsw %mm4, %mm4 

// CHECK: paddsw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xed,0x22]        
paddsw (%rdx), %mm4 

// CHECK: paddusb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdc,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
paddusb 485498096, %mm4 

// CHECK: paddusb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdc,0x62,0x40]        
paddusb 64(%rdx), %mm4 

// CHECK: paddusb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdc,0x64,0x82,0x40]        
paddusb 64(%rdx,%rax,4), %mm4 

// CHECK: paddusb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdc,0x64,0x82,0xc0]        
paddusb -64(%rdx,%rax,4), %mm4 

// CHECK: paddusb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xdc,0x64,0x02,0x40]        
paddusb 64(%rdx,%rax), %mm4 

// CHECK: paddusb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdc,0xe4]        
paddusb %mm4, %mm4 

// CHECK: paddusb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdc,0x22]        
paddusb (%rdx), %mm4 

// CHECK: paddusw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdd,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
paddusw 485498096, %mm4 

// CHECK: paddusw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdd,0x62,0x40]        
paddusw 64(%rdx), %mm4 

// CHECK: paddusw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdd,0x64,0x82,0x40]        
paddusw 64(%rdx,%rax,4), %mm4 

// CHECK: paddusw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdd,0x64,0x82,0xc0]        
paddusw -64(%rdx,%rax,4), %mm4 

// CHECK: paddusw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xdd,0x64,0x02,0x40]        
paddusw 64(%rdx,%rax), %mm4 

// CHECK: paddusw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdd,0xe4]        
paddusw %mm4, %mm4 

// CHECK: paddusw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdd,0x22]        
paddusw (%rdx), %mm4 

// CHECK: paddw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfd,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
paddw 485498096, %mm4 

// CHECK: paddw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfd,0x62,0x40]        
paddw 64(%rdx), %mm4 

// CHECK: paddw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfd,0x64,0x82,0x40]        
paddw 64(%rdx,%rax,4), %mm4 

// CHECK: paddw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfd,0x64,0x82,0xc0]        
paddw -64(%rdx,%rax,4), %mm4 

// CHECK: paddw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xfd,0x64,0x02,0x40]        
paddw 64(%rdx,%rax), %mm4 

// CHECK: paddw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfd,0xe4]        
paddw %mm4, %mm4 

// CHECK: paddw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfd,0x22]        
paddw (%rdx), %mm4 

// CHECK: pand 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdb,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pand 485498096, %mm4 

// CHECK: pand 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdb,0x62,0x40]        
pand 64(%rdx), %mm4 

// CHECK: pand 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdb,0x64,0x82,0x40]        
pand 64(%rdx,%rax,4), %mm4 

// CHECK: pand -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdb,0x64,0x82,0xc0]        
pand -64(%rdx,%rax,4), %mm4 

// CHECK: pand 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xdb,0x64,0x02,0x40]        
pand 64(%rdx,%rax), %mm4 

// CHECK: pand %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdb,0xe4]        
pand %mm4, %mm4 

// CHECK: pandn 485498096, %mm4 
// CHECK: encoding: [0x0f,0xdf,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pandn 485498096, %mm4 

// CHECK: pandn 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdf,0x62,0x40]        
pandn 64(%rdx), %mm4 

// CHECK: pandn 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdf,0x64,0x82,0x40]        
pandn 64(%rdx,%rax,4), %mm4 

// CHECK: pandn -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xdf,0x64,0x82,0xc0]        
pandn -64(%rdx,%rax,4), %mm4 

// CHECK: pandn 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xdf,0x64,0x02,0x40]        
pandn 64(%rdx,%rax), %mm4 

// CHECK: pandn %mm4, %mm4 
// CHECK: encoding: [0x0f,0xdf,0xe4]        
pandn %mm4, %mm4 

// CHECK: pandn (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdf,0x22]        
pandn (%rdx), %mm4 

// CHECK: pand (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xdb,0x22]        
pand (%rdx), %mm4 

// CHECK: pavgb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe0,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pavgb 485498096, %mm4 

// CHECK: pavgb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe0,0x62,0x40]        
pavgb 64(%rdx), %mm4 

// CHECK: pavgb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe0,0x64,0x82,0x40]        
pavgb 64(%rdx,%rax,4), %mm4 

// CHECK: pavgb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe0,0x64,0x82,0xc0]        
pavgb -64(%rdx,%rax,4), %mm4 

// CHECK: pavgb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe0,0x64,0x02,0x40]        
pavgb 64(%rdx,%rax), %mm4 

// CHECK: pavgb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe0,0xe4]        
pavgb %mm4, %mm4 

// CHECK: pavgb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe0,0x22]        
pavgb (%rdx), %mm4 

// CHECK: pavgw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe3,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pavgw 485498096, %mm4 

// CHECK: pavgw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe3,0x62,0x40]        
pavgw 64(%rdx), %mm4 

// CHECK: pavgw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe3,0x64,0x82,0x40]        
pavgw 64(%rdx,%rax,4), %mm4 

// CHECK: pavgw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe3,0x64,0x82,0xc0]        
pavgw -64(%rdx,%rax,4), %mm4 

// CHECK: pavgw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe3,0x64,0x02,0x40]        
pavgw 64(%rdx,%rax), %mm4 

// CHECK: pavgw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe3,0xe4]        
pavgw %mm4, %mm4 

// CHECK: pavgw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe3,0x22]        
pavgw (%rdx), %mm4 

// CHECK: pcmpeqb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x74,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpeqb 485498096, %mm4 

// CHECK: pcmpeqb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x74,0x62,0x40]        
pcmpeqb 64(%rdx), %mm4 

// CHECK: pcmpeqb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x74,0x64,0x82,0x40]        
pcmpeqb 64(%rdx,%rax,4), %mm4 

// CHECK: pcmpeqb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x74,0x64,0x82,0xc0]        
pcmpeqb -64(%rdx,%rax,4), %mm4 

// CHECK: pcmpeqb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x74,0x64,0x02,0x40]        
pcmpeqb 64(%rdx,%rax), %mm4 

// CHECK: pcmpeqb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x74,0xe4]        
pcmpeqb %mm4, %mm4 

// CHECK: pcmpeqb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x74,0x22]        
pcmpeqb (%rdx), %mm4 

// CHECK: pcmpeqd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x76,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpeqd 485498096, %mm4 

// CHECK: pcmpeqd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x76,0x62,0x40]        
pcmpeqd 64(%rdx), %mm4 

// CHECK: pcmpeqd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x76,0x64,0x82,0x40]        
pcmpeqd 64(%rdx,%rax,4), %mm4 

// CHECK: pcmpeqd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x76,0x64,0x82,0xc0]        
pcmpeqd -64(%rdx,%rax,4), %mm4 

// CHECK: pcmpeqd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x76,0x64,0x02,0x40]        
pcmpeqd 64(%rdx,%rax), %mm4 

// CHECK: pcmpeqd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x76,0xe4]        
pcmpeqd %mm4, %mm4 

// CHECK: pcmpeqd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x76,0x22]        
pcmpeqd (%rdx), %mm4 

// CHECK: pcmpeqw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x75,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpeqw 485498096, %mm4 

// CHECK: pcmpeqw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x75,0x62,0x40]        
pcmpeqw 64(%rdx), %mm4 

// CHECK: pcmpeqw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x75,0x64,0x82,0x40]        
pcmpeqw 64(%rdx,%rax,4), %mm4 

// CHECK: pcmpeqw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x75,0x64,0x82,0xc0]        
pcmpeqw -64(%rdx,%rax,4), %mm4 

// CHECK: pcmpeqw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x75,0x64,0x02,0x40]        
pcmpeqw 64(%rdx,%rax), %mm4 

// CHECK: pcmpeqw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x75,0xe4]        
pcmpeqw %mm4, %mm4 

// CHECK: pcmpeqw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x75,0x22]        
pcmpeqw (%rdx), %mm4 

// CHECK: pcmpgtb 485498096, %mm4 
// CHECK: encoding: [0x0f,0x64,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpgtb 485498096, %mm4 

// CHECK: pcmpgtb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x64,0x62,0x40]        
pcmpgtb 64(%rdx), %mm4 

// CHECK: pcmpgtb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x64,0x64,0x82,0x40]        
pcmpgtb 64(%rdx,%rax,4), %mm4 

// CHECK: pcmpgtb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x64,0x64,0x82,0xc0]        
pcmpgtb -64(%rdx,%rax,4), %mm4 

// CHECK: pcmpgtb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x64,0x64,0x02,0x40]        
pcmpgtb 64(%rdx,%rax), %mm4 

// CHECK: pcmpgtb %mm4, %mm4 
// CHECK: encoding: [0x0f,0x64,0xe4]        
pcmpgtb %mm4, %mm4 

// CHECK: pcmpgtb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x64,0x22]        
pcmpgtb (%rdx), %mm4 

// CHECK: pcmpgtd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x66,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpgtd 485498096, %mm4 

// CHECK: pcmpgtd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x66,0x62,0x40]        
pcmpgtd 64(%rdx), %mm4 

// CHECK: pcmpgtd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x66,0x64,0x82,0x40]        
pcmpgtd 64(%rdx,%rax,4), %mm4 

// CHECK: pcmpgtd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x66,0x64,0x82,0xc0]        
pcmpgtd -64(%rdx,%rax,4), %mm4 

// CHECK: pcmpgtd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x66,0x64,0x02,0x40]        
pcmpgtd 64(%rdx,%rax), %mm4 

// CHECK: pcmpgtd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x66,0xe4]        
pcmpgtd %mm4, %mm4 

// CHECK: pcmpgtd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x66,0x22]        
pcmpgtd (%rdx), %mm4 

// CHECK: pcmpgtw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x65,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pcmpgtw 485498096, %mm4 

// CHECK: pcmpgtw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x65,0x62,0x40]        
pcmpgtw 64(%rdx), %mm4 

// CHECK: pcmpgtw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x65,0x64,0x82,0x40]        
pcmpgtw 64(%rdx,%rax,4), %mm4 

// CHECK: pcmpgtw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x65,0x64,0x82,0xc0]        
pcmpgtw -64(%rdx,%rax,4), %mm4 

// CHECK: pcmpgtw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x65,0x64,0x02,0x40]        
pcmpgtw 64(%rdx,%rax), %mm4 

// CHECK: pcmpgtw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x65,0xe4]        
pcmpgtw %mm4, %mm4 

// CHECK: pcmpgtw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x65,0x22]        
pcmpgtw (%rdx), %mm4 

// CHECK: pmaddwd 485498096, %mm4
// CHECK: encoding: [0x0f,0xf5,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pmaddwd 485498096, %mm4 

// CHECK: pmaddwd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf5,0x62,0x40]        
pmaddwd 64(%rdx), %mm4 

// CHECK: pmaddwd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf5,0x64,0x82,0x40]        
pmaddwd 64(%rdx,%rax,4), %mm4 

// CHECK: pmaddwd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf5,0x64,0x82,0xc0]        
pmaddwd -64(%rdx,%rax,4), %mm4 

// CHECK: pmaddwd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xf5,0x64,0x02,0x40]        
pmaddwd 64(%rdx,%rax), %mm4 

// CHECK: pmaddwd %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf5,0xe4]        
pmaddwd %mm4, %mm4 

// CHECK: pmaddwd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf5,0x22]        
pmaddwd (%rdx), %mm4 

// CHECK: pmaxsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xee,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pmaxsw 485498096, %mm4 

// CHECK: pmaxsw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xee,0x62,0x40]        
pmaxsw 64(%rdx), %mm4 

// CHECK: pmaxsw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xee,0x64,0x82,0x40]        
pmaxsw 64(%rdx,%rax,4), %mm4 

// CHECK: pmaxsw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xee,0x64,0x82,0xc0]        
pmaxsw -64(%rdx,%rax,4), %mm4 

// CHECK: pmaxsw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xee,0x64,0x02,0x40]        
pmaxsw 64(%rdx,%rax), %mm4 

// CHECK: pmaxsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xee,0xe4]        
pmaxsw %mm4, %mm4 

// CHECK: pmaxsw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xee,0x22]        
pmaxsw (%rdx), %mm4 

// CHECK: pmaxub 485498096, %mm4 
// CHECK: encoding: [0x0f,0xde,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pmaxub 485498096, %mm4 

// CHECK: pmaxub 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xde,0x62,0x40]        
pmaxub 64(%rdx), %mm4 

// CHECK: pmaxub 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xde,0x64,0x82,0x40]        
pmaxub 64(%rdx,%rax,4), %mm4 

// CHECK: pmaxub -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xde,0x64,0x82,0xc0]        
pmaxub -64(%rdx,%rax,4), %mm4 

// CHECK: pmaxub 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xde,0x64,0x02,0x40]        
pmaxub 64(%rdx,%rax), %mm4 

// CHECK: pmaxub %mm4, %mm4 
// CHECK: encoding: [0x0f,0xde,0xe4]        
pmaxub %mm4, %mm4 

// CHECK: pmaxub (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xde,0x22]        
pmaxub (%rdx), %mm4 

// CHECK: pminsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xea,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pminsw 485498096, %mm4 

// CHECK: pminsw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xea,0x62,0x40]        
pminsw 64(%rdx), %mm4 

// CHECK: pminsw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xea,0x64,0x82,0x40]        
pminsw 64(%rdx,%rax,4), %mm4 

// CHECK: pminsw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xea,0x64,0x82,0xc0]        
pminsw -64(%rdx,%rax,4), %mm4 

// CHECK: pminsw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xea,0x64,0x02,0x40]        
pminsw 64(%rdx,%rax), %mm4 

// CHECK: pminsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xea,0xe4]        
pminsw %mm4, %mm4 

// CHECK: pminsw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xea,0x22]        
pminsw (%rdx), %mm4 

// CHECK: pminub 485498096, %mm4 
// CHECK: encoding: [0x0f,0xda,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pminub 485498096, %mm4 

// CHECK: pminub 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xda,0x62,0x40]        
pminub 64(%rdx), %mm4 

// CHECK: pminub 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xda,0x64,0x82,0x40]        
pminub 64(%rdx,%rax,4), %mm4 

// CHECK: pminub -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xda,0x64,0x82,0xc0]        
pminub -64(%rdx,%rax,4), %mm4 

// CHECK: pminub 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xda,0x64,0x02,0x40]        
pminub 64(%rdx,%rax), %mm4 

// CHECK: pminub %mm4, %mm4 
// CHECK: encoding: [0x0f,0xda,0xe4]        
pminub %mm4, %mm4 

// CHECK: pminub (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xda,0x22]        
pminub (%rdx), %mm4 

// CHECK: pmulhuw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe4,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pmulhuw 485498096, %mm4 

// CHECK: pmulhuw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe4,0x62,0x40]        
pmulhuw 64(%rdx), %mm4 

// CHECK: pmulhuw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe4,0x64,0x82,0x40]        
pmulhuw 64(%rdx,%rax,4), %mm4 

// CHECK: pmulhuw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe4,0x64,0x82,0xc0]        
pmulhuw -64(%rdx,%rax,4), %mm4 

// CHECK: pmulhuw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe4,0x64,0x02,0x40]        
pmulhuw 64(%rdx,%rax), %mm4 

// CHECK: pmulhuw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe4,0xe4]        
pmulhuw %mm4, %mm4 

// CHECK: pmulhuw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe4,0x22]        
pmulhuw (%rdx), %mm4 

// CHECK: pmulhw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe5,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pmulhw 485498096, %mm4 

// CHECK: pmulhw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe5,0x62,0x40]        
pmulhw 64(%rdx), %mm4 

// CHECK: pmulhw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe5,0x64,0x82,0x40]        
pmulhw 64(%rdx,%rax,4), %mm4 

// CHECK: pmulhw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe5,0x64,0x82,0xc0]        
pmulhw -64(%rdx,%rax,4), %mm4 

// CHECK: pmulhw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe5,0x64,0x02,0x40]        
pmulhw 64(%rdx,%rax), %mm4 

// CHECK: pmulhw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe5,0xe4]        
pmulhw %mm4, %mm4 

// CHECK: pmulhw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe5,0x22]        
pmulhw (%rdx), %mm4 

// CHECK: pmullw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd5,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pmullw 485498096, %mm4 

// CHECK: pmullw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd5,0x62,0x40]        
pmullw 64(%rdx), %mm4 

// CHECK: pmullw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd5,0x64,0x82,0x40]        
pmullw 64(%rdx,%rax,4), %mm4 

// CHECK: pmullw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd5,0x64,0x82,0xc0]        
pmullw -64(%rdx,%rax,4), %mm4 

// CHECK: pmullw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xd5,0x64,0x02,0x40]        
pmullw 64(%rdx,%rax), %mm4 

// CHECK: pmullw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd5,0xe4]        
pmullw %mm4, %mm4 

// CHECK: pmullw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd5,0x22]        
pmullw (%rdx), %mm4 

// CHECK: por 485498096, %mm4 
// CHECK: encoding: [0x0f,0xeb,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
por 485498096, %mm4 

// CHECK: por 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xeb,0x62,0x40]        
por 64(%rdx), %mm4 

// CHECK: por 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xeb,0x64,0x82,0x40]        
por 64(%rdx,%rax,4), %mm4 

// CHECK: por -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xeb,0x64,0x82,0xc0]        
por -64(%rdx,%rax,4), %mm4 

// CHECK: por 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xeb,0x64,0x02,0x40]        
por 64(%rdx,%rax), %mm4 

// CHECK: por %mm4, %mm4 
// CHECK: encoding: [0x0f,0xeb,0xe4]        
por %mm4, %mm4 

// CHECK: por (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xeb,0x22]        
por (%rdx), %mm4 

// CHECK: psadbw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf6,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psadbw 485498096, %mm4 

// CHECK: psadbw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf6,0x62,0x40]        
psadbw 64(%rdx), %mm4 

// CHECK: psadbw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf6,0x64,0x82,0x40]        
psadbw 64(%rdx,%rax,4), %mm4 

// CHECK: psadbw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf6,0x64,0x82,0xc0]        
psadbw -64(%rdx,%rax,4), %mm4 

// CHECK: psadbw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xf6,0x64,0x02,0x40]        
psadbw 64(%rdx,%rax), %mm4 

// CHECK: psadbw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf6,0xe4]        
psadbw %mm4, %mm4 

// CHECK: psadbw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf6,0x22]        
psadbw (%rdx), %mm4 

// CHECK: pshufw $0, 485498096, %mm4 
// CHECK: encoding: [0x0f,0x70,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]       
pshufw $0, 485498096, %mm4 

// CHECK: pshufw $0, 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x70,0x62,0x40,0x00]       
pshufw $0, 64(%rdx), %mm4 

// CHECK: pshufw $0, 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x70,0x64,0x82,0x40,0x00]       
pshufw $0, 64(%rdx,%rax,4), %mm4 

// CHECK: pshufw $0, -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x70,0x64,0x82,0xc0,0x00]       
pshufw $0, -64(%rdx,%rax,4), %mm4 

// CHECK: pshufw $0, 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x70,0x64,0x02,0x40,0x00]       
pshufw $0, 64(%rdx,%rax), %mm4 

// CHECK: pshufw $0, %mm4, %mm4 
// CHECK: encoding: [0x0f,0x70,0xe4,0x00]       
pshufw $0, %mm4, %mm4 

// CHECK: pshufw $0, (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x70,0x22,0x00]       
pshufw $0, (%rdx), %mm4 

// CHECK: pslld $0, %mm4 
// CHECK: encoding: [0x0f,0x72,0xf4,0x00]        
pslld $0, %mm4 

// CHECK: pslld 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf2,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pslld 485498096, %mm4 

// CHECK: pslld 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf2,0x62,0x40]        
pslld 64(%rdx), %mm4 

// CHECK: pslld 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf2,0x64,0x82,0x40]        
pslld 64(%rdx,%rax,4), %mm4 

// CHECK: pslld -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf2,0x64,0x82,0xc0]        
pslld -64(%rdx,%rax,4), %mm4 

// CHECK: pslld 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xf2,0x64,0x02,0x40]        
pslld 64(%rdx,%rax), %mm4 

// CHECK: pslld %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf2,0xe4]        
pslld %mm4, %mm4 

// CHECK: pslld (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf2,0x22]        
pslld (%rdx), %mm4 

// CHECK: psllq $0, %mm4 
// CHECK: encoding: [0x0f,0x73,0xf4,0x00]        
psllq $0, %mm4 

// CHECK: psllq 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf3,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psllq 485498096, %mm4 

// CHECK: psllq 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf3,0x62,0x40]        
psllq 64(%rdx), %mm4 

// CHECK: psllq 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf3,0x64,0x82,0x40]        
psllq 64(%rdx,%rax,4), %mm4 

// CHECK: psllq -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf3,0x64,0x82,0xc0]        
psllq -64(%rdx,%rax,4), %mm4 

// CHECK: psllq 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xf3,0x64,0x02,0x40]        
psllq 64(%rdx,%rax), %mm4 

// CHECK: psllq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf3,0xe4]        
psllq %mm4, %mm4 

// CHECK: psllq (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf3,0x22]        
psllq (%rdx), %mm4 

// CHECK: psllw $0, %mm4 
// CHECK: encoding: [0x0f,0x71,0xf4,0x00]        
psllw $0, %mm4 

// CHECK: psllw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf1,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psllw 485498096, %mm4 

// CHECK: psllw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf1,0x62,0x40]        
psllw 64(%rdx), %mm4 

// CHECK: psllw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf1,0x64,0x82,0x40]        
psllw 64(%rdx,%rax,4), %mm4 

// CHECK: psllw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf1,0x64,0x82,0xc0]        
psllw -64(%rdx,%rax,4), %mm4 

// CHECK: psllw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xf1,0x64,0x02,0x40]        
psllw 64(%rdx,%rax), %mm4 

// CHECK: psllw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf1,0xe4]        
psllw %mm4, %mm4 

// CHECK: psllw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf1,0x22]        
psllw (%rdx), %mm4 

// CHECK: psrad $0, %mm4 
// CHECK: encoding: [0x0f,0x72,0xe4,0x00]        
psrad $0, %mm4 

// CHECK: psrad 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe2,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psrad 485498096, %mm4 

// CHECK: psrad 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe2,0x62,0x40]        
psrad 64(%rdx), %mm4 

// CHECK: psrad 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe2,0x64,0x82,0x40]        
psrad 64(%rdx,%rax,4), %mm4 

// CHECK: psrad -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe2,0x64,0x82,0xc0]        
psrad -64(%rdx,%rax,4), %mm4 

// CHECK: psrad 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe2,0x64,0x02,0x40]        
psrad 64(%rdx,%rax), %mm4 

// CHECK: psrad %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe2,0xe4]        
psrad %mm4, %mm4 

// CHECK: psrad (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe2,0x22]        
psrad (%rdx), %mm4 

// CHECK: psraw $0, %mm4 
// CHECK: encoding: [0x0f,0x71,0xe4,0x00]        
psraw $0, %mm4 

// CHECK: psraw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe1,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psraw 485498096, %mm4 

// CHECK: psraw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe1,0x62,0x40]        
psraw 64(%rdx), %mm4 

// CHECK: psraw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe1,0x64,0x82,0x40]        
psraw 64(%rdx,%rax,4), %mm4 

// CHECK: psraw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe1,0x64,0x82,0xc0]        
psraw -64(%rdx,%rax,4), %mm4 

// CHECK: psraw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe1,0x64,0x02,0x40]        
psraw 64(%rdx,%rax), %mm4 

// CHECK: psraw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe1,0xe4]        
psraw %mm4, %mm4 

// CHECK: psraw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe1,0x22]        
psraw (%rdx), %mm4 

// CHECK: psrld $0, %mm4 
// CHECK: encoding: [0x0f,0x72,0xd4,0x00]        
psrld $0, %mm4 

// CHECK: psrld 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd2,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psrld 485498096, %mm4 

// CHECK: psrld 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd2,0x62,0x40]        
psrld 64(%rdx), %mm4 

// CHECK: psrld 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd2,0x64,0x82,0x40]        
psrld 64(%rdx,%rax,4), %mm4 

// CHECK: psrld -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd2,0x64,0x82,0xc0]        
psrld -64(%rdx,%rax,4), %mm4 

// CHECK: psrld 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xd2,0x64,0x02,0x40]        
psrld 64(%rdx,%rax), %mm4 

// CHECK: psrld %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd2,0xe4]        
psrld %mm4, %mm4 

// CHECK: psrld (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd2,0x22]        
psrld (%rdx), %mm4 

// CHECK: psrlq $0, %mm4 
// CHECK: encoding: [0x0f,0x73,0xd4,0x00]        
psrlq $0, %mm4 

// CHECK: psrlq 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd3,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psrlq 485498096, %mm4 

// CHECK: psrlq 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd3,0x62,0x40]        
psrlq 64(%rdx), %mm4 

// CHECK: psrlq 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd3,0x64,0x82,0x40]        
psrlq 64(%rdx,%rax,4), %mm4 

// CHECK: psrlq -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd3,0x64,0x82,0xc0]        
psrlq -64(%rdx,%rax,4), %mm4 

// CHECK: psrlq 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xd3,0x64,0x02,0x40]        
psrlq 64(%rdx,%rax), %mm4 

// CHECK: psrlq %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd3,0xe4]        
psrlq %mm4, %mm4 

// CHECK: psrlq (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd3,0x22]        
psrlq (%rdx), %mm4 

// CHECK: psrlw $0, %mm4 
// CHECK: encoding: [0x0f,0x71,0xd4,0x00]        
psrlw $0, %mm4 

// CHECK: psrlw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd1,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psrlw 485498096, %mm4 

// CHECK: psrlw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd1,0x62,0x40]        
psrlw 64(%rdx), %mm4 

// CHECK: psrlw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd1,0x64,0x82,0x40]        
psrlw 64(%rdx,%rax,4), %mm4 

// CHECK: psrlw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd1,0x64,0x82,0xc0]        
psrlw -64(%rdx,%rax,4), %mm4 

// CHECK: psrlw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xd1,0x64,0x02,0x40]        
psrlw 64(%rdx,%rax), %mm4 

// CHECK: psrlw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd1,0xe4]        
psrlw %mm4, %mm4 

// CHECK: psrlw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd1,0x22]        
psrlw (%rdx), %mm4 

// CHECK: psubb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf8,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psubb 485498096, %mm4 

// CHECK: psubb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf8,0x62,0x40]        
psubb 64(%rdx), %mm4 

// CHECK: psubb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf8,0x64,0x82,0x40]        
psubb 64(%rdx,%rax,4), %mm4 

// CHECK: psubb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf8,0x64,0x82,0xc0]        
psubb -64(%rdx,%rax,4), %mm4 

// CHECK: psubb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xf8,0x64,0x02,0x40]        
psubb 64(%rdx,%rax), %mm4 

// CHECK: psubb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf8,0xe4]        
psubb %mm4, %mm4 

// CHECK: psubb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf8,0x22]        
psubb (%rdx), %mm4 

// CHECK: psubd 485498096, %mm4 
// CHECK: encoding: [0x0f,0xfa,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psubd 485498096, %mm4 

// CHECK: psubd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfa,0x62,0x40]        
psubd 64(%rdx), %mm4 

// CHECK: psubd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfa,0x64,0x82,0x40]        
psubd 64(%rdx,%rax,4), %mm4 

// CHECK: psubd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xfa,0x64,0x82,0xc0]        
psubd -64(%rdx,%rax,4), %mm4 

// CHECK: psubd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xfa,0x64,0x02,0x40]        
psubd 64(%rdx,%rax), %mm4 

// CHECK: psubd %mm4, %mm4 
// CHECK: encoding: [0x0f,0xfa,0xe4]        
psubd %mm4, %mm4 

// CHECK: psubd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xfa,0x22]        
psubd (%rdx), %mm4 

// CHECK: psubsb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe8,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psubsb 485498096, %mm4 

// CHECK: psubsb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe8,0x62,0x40]        
psubsb 64(%rdx), %mm4 

// CHECK: psubsb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe8,0x64,0x82,0x40]        
psubsb 64(%rdx,%rax,4), %mm4 

// CHECK: psubsb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe8,0x64,0x82,0xc0]        
psubsb -64(%rdx,%rax,4), %mm4 

// CHECK: psubsb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe8,0x64,0x02,0x40]        
psubsb 64(%rdx,%rax), %mm4 

// CHECK: psubsb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe8,0xe4]        
psubsb %mm4, %mm4 

// CHECK: psubsb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe8,0x22]        
psubsb (%rdx), %mm4 

// CHECK: psubsw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xe9,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psubsw 485498096, %mm4 

// CHECK: psubsw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe9,0x62,0x40]        
psubsw 64(%rdx), %mm4 

// CHECK: psubsw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe9,0x64,0x82,0x40]        
psubsw 64(%rdx,%rax,4), %mm4 

// CHECK: psubsw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xe9,0x64,0x82,0xc0]        
psubsw -64(%rdx,%rax,4), %mm4 

// CHECK: psubsw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xe9,0x64,0x02,0x40]        
psubsw 64(%rdx,%rax), %mm4 

// CHECK: psubsw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xe9,0xe4]        
psubsw %mm4, %mm4 

// CHECK: psubsw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xe9,0x22]        
psubsw (%rdx), %mm4 

// CHECK: psubusb 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd8,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psubusb 485498096, %mm4 

// CHECK: psubusb 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd8,0x62,0x40]        
psubusb 64(%rdx), %mm4 

// CHECK: psubusb 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd8,0x64,0x82,0x40]        
psubusb 64(%rdx,%rax,4), %mm4 

// CHECK: psubusb -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd8,0x64,0x82,0xc0]        
psubusb -64(%rdx,%rax,4), %mm4 

// CHECK: psubusb 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xd8,0x64,0x02,0x40]        
psubusb 64(%rdx,%rax), %mm4 

// CHECK: psubusb %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd8,0xe4]        
psubusb %mm4, %mm4 

// CHECK: psubusb (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd8,0x22]        
psubusb (%rdx), %mm4 

// CHECK: psubusw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xd9,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psubusw 485498096, %mm4 

// CHECK: psubusw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd9,0x62,0x40]        
psubusw 64(%rdx), %mm4 

// CHECK: psubusw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd9,0x64,0x82,0x40]        
psubusw 64(%rdx,%rax,4), %mm4 

// CHECK: psubusw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xd9,0x64,0x82,0xc0]        
psubusw -64(%rdx,%rax,4), %mm4 

// CHECK: psubusw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xd9,0x64,0x02,0x40]        
psubusw 64(%rdx,%rax), %mm4 

// CHECK: psubusw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xd9,0xe4]        
psubusw %mm4, %mm4 

// CHECK: psubusw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xd9,0x22]        
psubusw (%rdx), %mm4 

// CHECK: psubw 485498096, %mm4 
// CHECK: encoding: [0x0f,0xf9,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
psubw 485498096, %mm4 

// CHECK: psubw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf9,0x62,0x40]        
psubw 64(%rdx), %mm4 

// CHECK: psubw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf9,0x64,0x82,0x40]        
psubw 64(%rdx,%rax,4), %mm4 

// CHECK: psubw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xf9,0x64,0x82,0xc0]        
psubw -64(%rdx,%rax,4), %mm4 

// CHECK: psubw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xf9,0x64,0x02,0x40]        
psubw 64(%rdx,%rax), %mm4 

// CHECK: psubw %mm4, %mm4 
// CHECK: encoding: [0x0f,0xf9,0xe4]        
psubw %mm4, %mm4 

// CHECK: psubw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xf9,0x22]        
psubw (%rdx), %mm4 

// CHECK: punpckhbw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x68,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckhbw 485498096, %mm4 

// CHECK: punpckhbw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x68,0x62,0x40]        
punpckhbw 64(%rdx), %mm4 

// CHECK: punpckhbw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x68,0x64,0x82,0x40]        
punpckhbw 64(%rdx,%rax,4), %mm4 

// CHECK: punpckhbw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x68,0x64,0x82,0xc0]        
punpckhbw -64(%rdx,%rax,4), %mm4 

// CHECK: punpckhbw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x68,0x64,0x02,0x40]        
punpckhbw 64(%rdx,%rax), %mm4 

// CHECK: punpckhbw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x68,0xe4]        
punpckhbw %mm4, %mm4 

// CHECK: punpckhbw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x68,0x22]        
punpckhbw (%rdx), %mm4 

// CHECK: punpckhdq 485498096, %mm4 
// CHECK: encoding: [0x0f,0x6a,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckhdq 485498096, %mm4 

// CHECK: punpckhdq 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6a,0x62,0x40]        
punpckhdq 64(%rdx), %mm4 

// CHECK: punpckhdq 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6a,0x64,0x82,0x40]        
punpckhdq 64(%rdx,%rax,4), %mm4 

// CHECK: punpckhdq -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x6a,0x64,0x82,0xc0]        
punpckhdq -64(%rdx,%rax,4), %mm4 

// CHECK: punpckhdq 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x6a,0x64,0x02,0x40]        
punpckhdq 64(%rdx,%rax), %mm4 

// CHECK: punpckhdq %mm4, %mm4 
// CHECK: encoding: [0x0f,0x6a,0xe4]        
punpckhdq %mm4, %mm4 

// CHECK: punpckhdq (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x6a,0x22]        
punpckhdq (%rdx), %mm4 

// CHECK: punpckhwd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x69,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckhwd 485498096, %mm4 

// CHECK: punpckhwd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x69,0x62,0x40]        
punpckhwd 64(%rdx), %mm4 

// CHECK: punpckhwd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x69,0x64,0x82,0x40]        
punpckhwd 64(%rdx,%rax,4), %mm4 

// CHECK: punpckhwd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x69,0x64,0x82,0xc0]        
punpckhwd -64(%rdx,%rax,4), %mm4 

// CHECK: punpckhwd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x69,0x64,0x02,0x40]        
punpckhwd 64(%rdx,%rax), %mm4 

// CHECK: punpckhwd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x69,0xe4]        
punpckhwd %mm4, %mm4 

// CHECK: punpckhwd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x69,0x22]        
punpckhwd (%rdx), %mm4 

// CHECK: punpcklbw 485498096, %mm4 
// CHECK: encoding: [0x0f,0x60,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
punpcklbw 485498096, %mm4 

// CHECK: punpcklbw 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x60,0x62,0x40]        
punpcklbw 64(%rdx), %mm4 

// CHECK: punpcklbw 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x60,0x64,0x82,0x40]        
punpcklbw 64(%rdx,%rax,4), %mm4 

// CHECK: punpcklbw -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x60,0x64,0x82,0xc0]        
punpcklbw -64(%rdx,%rax,4), %mm4 

// CHECK: punpcklbw 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x60,0x64,0x02,0x40]        
punpcklbw 64(%rdx,%rax), %mm4 

// CHECK: punpcklbw %mm4, %mm4 
// CHECK: encoding: [0x0f,0x60,0xe4]        
punpcklbw %mm4, %mm4 

// CHECK: punpcklbw (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x60,0x22]        
punpcklbw (%rdx), %mm4 

// CHECK: punpckldq 485498096, %mm4 
// CHECK: encoding: [0x0f,0x62,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
punpckldq 485498096, %mm4 

// CHECK: punpckldq 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x62,0x62,0x40]        
punpckldq 64(%rdx), %mm4 

// CHECK: punpckldq 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x62,0x64,0x82,0x40]        
punpckldq 64(%rdx,%rax,4), %mm4 

// CHECK: punpckldq -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x62,0x64,0x82,0xc0]        
punpckldq -64(%rdx,%rax,4), %mm4 

// CHECK: punpckldq 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x62,0x64,0x02,0x40]        
punpckldq 64(%rdx,%rax), %mm4 

// CHECK: punpckldq %mm4, %mm4 
// CHECK: encoding: [0x0f,0x62,0xe4]        
punpckldq %mm4, %mm4 

// CHECK: punpckldq (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x62,0x22]        
punpckldq (%rdx), %mm4 

// CHECK: punpcklwd 485498096, %mm4 
// CHECK: encoding: [0x0f,0x61,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
punpcklwd 485498096, %mm4 

// CHECK: punpcklwd 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0x61,0x62,0x40]        
punpcklwd 64(%rdx), %mm4 

// CHECK: punpcklwd 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x61,0x64,0x82,0x40]        
punpcklwd 64(%rdx,%rax,4), %mm4 

// CHECK: punpcklwd -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0x61,0x64,0x82,0xc0]        
punpcklwd -64(%rdx,%rax,4), %mm4 

// CHECK: punpcklwd 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0x61,0x64,0x02,0x40]        
punpcklwd 64(%rdx,%rax), %mm4 

// CHECK: punpcklwd %mm4, %mm4 
// CHECK: encoding: [0x0f,0x61,0xe4]        
punpcklwd %mm4, %mm4 

// CHECK: punpcklwd (%rdx), %mm4 
// CHECK: encoding: [0x0f,0x61,0x22]        
punpcklwd (%rdx), %mm4 

// CHECK: pxor 485498096, %mm4 
// CHECK: encoding: [0x0f,0xef,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
pxor 485498096, %mm4 

// CHECK: pxor 64(%rdx), %mm4 
// CHECK: encoding: [0x0f,0xef,0x62,0x40]        
pxor 64(%rdx), %mm4 

// CHECK: pxor 64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xef,0x64,0x82,0x40]        
pxor 64(%rdx,%rax,4), %mm4 

// CHECK: pxor -64(%rdx,%rax,4), %mm4 
// CHECK: encoding: [0x0f,0xef,0x64,0x82,0xc0]        
pxor -64(%rdx,%rax,4), %mm4 

// CHECK: pxor 64(%rdx,%rax), %mm4 
// CHECK: encoding: [0x0f,0xef,0x64,0x02,0x40]        
pxor 64(%rdx,%rax), %mm4 

// CHECK: pxor %mm4, %mm4 
// CHECK: encoding: [0x0f,0xef,0xe4]        
pxor %mm4, %mm4 

// CHECK: pxor (%rdx), %mm4 
// CHECK: encoding: [0x0f,0xef,0x22]        
pxor (%rdx), %mm4 

