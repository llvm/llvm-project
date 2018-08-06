// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clts 
// CHECK: encoding: [0x0f,0x06]          
clts 

// CHECK: larl 485498096, %r13d 
// CHECK: encoding: [0x44,0x0f,0x02,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
larl 485498096, %r13d 

// CHECK: larl 64(%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0x02,0x6a,0x40]        
larl 64(%rdx), %r13d 

// CHECK: larl 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0x02,0x6c,0x82,0x40]        
larl 64(%rdx,%rax,4), %r13d 

// CHECK: larl -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0x02,0x6c,0x82,0xc0]        
larl -64(%rdx,%rax,4), %r13d 

// CHECK: larl 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0x44,0x0f,0x02,0x6c,0x02,0x40]        
larl 64(%rdx,%rax), %r13d 

// CHECK: larl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x02,0xed]        
larl %r13d, %r13d 

// CHECK: larl (%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0x02,0x2a]        
larl (%rdx), %r13d 

// CHECK: lgdtq 485498096 
// CHECK: encoding: [0x0f,0x01,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
lgdtq 485498096 

// CHECK: lgdtq 64(%rdx) 
// CHECK: encoding: [0x0f,0x01,0x52,0x40]         
lgdtq 64(%rdx) 

// CHECK: lgdtq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x54,0x82,0x40]         
lgdtq 64(%rdx,%rax,4) 

// CHECK: lgdtq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x54,0x82,0xc0]         
lgdtq -64(%rdx,%rax,4) 

// CHECK: lgdtq 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x01,0x54,0x02,0x40]         
lgdtq 64(%rdx,%rax) 

// CHECK: lgdtq (%rdx) 
// CHECK: encoding: [0x0f,0x01,0x12]         
lgdtq (%rdx) 

// CHECK: lidtq 485498096 
// CHECK: encoding: [0x0f,0x01,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
lidtq 485498096 

// CHECK: lidtq 64(%rdx) 
// CHECK: encoding: [0x0f,0x01,0x5a,0x40]         
lidtq 64(%rdx) 

// CHECK: lidtq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x5c,0x82,0x40]         
lidtq 64(%rdx,%rax,4) 

// CHECK: lidtq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x5c,0x82,0xc0]         
lidtq -64(%rdx,%rax,4) 

// CHECK: lidtq 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x01,0x5c,0x02,0x40]         
lidtq 64(%rdx,%rax) 

// CHECK: lidtq (%rdx) 
// CHECK: encoding: [0x0f,0x01,0x1a]         
lidtq (%rdx) 

// CHECK: lldtw 485498096 
// CHECK: encoding: [0x0f,0x00,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
lldtw 485498096 

// CHECK: lldtw 64(%rdx) 
// CHECK: encoding: [0x0f,0x00,0x52,0x40]         
lldtw 64(%rdx) 

// CHECK: lldtw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x54,0x82,0x40]         
lldtw 64(%rdx,%rax,4) 

// CHECK: lldtw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x54,0x82,0xc0]         
lldtw -64(%rdx,%rax,4) 

// CHECK: lldtw 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x00,0x54,0x02,0x40]         
lldtw 64(%rdx,%rax) 

// CHECK: lldtw %r11w 
// CHECK: encoding: [0x41,0x0f,0x00,0xd3]         
lldtw %r11w 

// CHECK: lldtw (%rdx) 
// CHECK: encoding: [0x0f,0x00,0x12]         
lldtw (%rdx) 

// CHECK: lmsww 485498096 
// CHECK: encoding: [0x0f,0x01,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
lmsww 485498096 

// CHECK: lmsww 64(%rdx) 
// CHECK: encoding: [0x0f,0x01,0x72,0x40]         
lmsww 64(%rdx) 

// CHECK: lmsww 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x74,0x82,0x40]         
lmsww 64(%rdx,%rax,4) 

// CHECK: lmsww -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x74,0x82,0xc0]         
lmsww -64(%rdx,%rax,4) 

// CHECK: lmsww 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x01,0x74,0x02,0x40]         
lmsww 64(%rdx,%rax) 

// CHECK: lmsww %r11w 
// CHECK: encoding: [0x41,0x0f,0x01,0xf3]         
lmsww %r11w 

// CHECK: lmsww (%rdx) 
// CHECK: encoding: [0x0f,0x01,0x32]         
lmsww (%rdx) 

// CHECK: lsll 485498096, %r13d 
// CHECK: encoding: [0x44,0x0f,0x03,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
lsll 485498096, %r13d 

// CHECK: lsll 64(%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0x03,0x6a,0x40]        
lsll 64(%rdx), %r13d 

// CHECK: lsll 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0x03,0x6c,0x82,0x40]        
lsll 64(%rdx,%rax,4), %r13d 

// CHECK: lsll -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0x03,0x6c,0x82,0xc0]        
lsll -64(%rdx,%rax,4), %r13d 

// CHECK: lsll 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0x44,0x0f,0x03,0x6c,0x02,0x40]        
lsll 64(%rdx,%rax), %r13d 

// CHECK: lsll %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x03,0xed]        
lsll %r13d, %r13d 

// CHECK: lsll (%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0x03,0x2a]        
lsll (%rdx), %r13d 

// CHECK: ltrw 485498096 
// CHECK: encoding: [0x0f,0x00,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
ltrw 485498096 

// CHECK: ltrw 64(%rdx) 
// CHECK: encoding: [0x0f,0x00,0x5a,0x40]         
ltrw 64(%rdx) 

// CHECK: ltrw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x5c,0x82,0x40]         
ltrw 64(%rdx,%rax,4) 

// CHECK: ltrw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x5c,0x82,0xc0]         
ltrw -64(%rdx,%rax,4) 

// CHECK: ltrw 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x00,0x5c,0x02,0x40]         
ltrw 64(%rdx,%rax) 

// CHECK: ltrw %r11w 
// CHECK: encoding: [0x41,0x0f,0x00,0xdb]         
ltrw %r11w 

// CHECK: ltrw (%rdx) 
// CHECK: encoding: [0x0f,0x00,0x1a]         
ltrw (%rdx) 

// CHECK: sgdtq 485498096 
// CHECK: encoding: [0x0f,0x01,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
sgdtq 485498096 

// CHECK: sgdtq 64(%rdx) 
// CHECK: encoding: [0x0f,0x01,0x42,0x40]         
sgdtq 64(%rdx) 

// CHECK: sgdtq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x44,0x82,0x40]         
sgdtq 64(%rdx,%rax,4) 

// CHECK: sgdtq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x44,0x82,0xc0]         
sgdtq -64(%rdx,%rax,4) 

// CHECK: sgdtq 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x01,0x44,0x02,0x40]         
sgdtq 64(%rdx,%rax) 

// CHECK: sgdtq (%rdx) 
// CHECK: encoding: [0x0f,0x01,0x02]         
sgdtq (%rdx) 

// CHECK: sidtq 485498096 
// CHECK: encoding: [0x0f,0x01,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
sidtq 485498096 

// CHECK: sidtq 64(%rdx) 
// CHECK: encoding: [0x0f,0x01,0x4a,0x40]         
sidtq 64(%rdx) 

// CHECK: sidtq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x4c,0x82,0x40]         
sidtq 64(%rdx,%rax,4) 

// CHECK: sidtq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x4c,0x82,0xc0]         
sidtq -64(%rdx,%rax,4) 

// CHECK: sidtq 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x01,0x4c,0x02,0x40]         
sidtq 64(%rdx,%rax) 

// CHECK: sidtq (%rdx) 
// CHECK: encoding: [0x0f,0x01,0x0a]         
sidtq (%rdx) 

// CHECK: sldtw %r13w
// CHECK: encoding: [0x66,0x41,0x0f,0x00,0xc5]
sldtw %r13w

// CHECK: sldtl %r13d 
// CHECK: encoding: [0x41,0x0f,0x00,0xc5]         
sldtl %r13d 

// CHECK: sldtq %r13
// CHECK: encoding: [0x49,0x0f,0x00,0xc5]
sldtq %r13

// CHECK: sldtw 485498096 
// CHECK: encoding: [0x0f,0x00,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
sldtw 485498096 

// CHECK: sldtw 64(%rdx) 
// CHECK: encoding: [0x0f,0x00,0x42,0x40]         
sldtw 64(%rdx) 

// CHECK: sldtw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x44,0x82,0x40]         
sldtw 64(%rdx,%rax,4) 

// CHECK: sldtw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x44,0x82,0xc0]         
sldtw -64(%rdx,%rax,4) 

// CHECK: sldtw 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x00,0x44,0x02,0x40]         
sldtw 64(%rdx,%rax) 

// CHECK: sldtw (%rdx) 
// CHECK: encoding: [0x0f,0x00,0x02]         
sldtw (%rdx) 

// CHECK: smswl %r13d 
// CHECK: encoding: [0x41,0x0f,0x01,0xe5]         
smswl %r13d 

// CHECK: smsww 485498096 
// CHECK: encoding: [0x0f,0x01,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
smsww 485498096 

// CHECK: smsww 64(%rdx) 
// CHECK: encoding: [0x0f,0x01,0x62,0x40]         
smsww 64(%rdx) 

// CHECK: smsww 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x64,0x82,0x40]         
smsww 64(%rdx,%rax,4) 

// CHECK: smsww -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x64,0x82,0xc0]         
smsww -64(%rdx,%rax,4) 

// CHECK: smsww 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x01,0x64,0x02,0x40]         
smsww 64(%rdx,%rax) 

// CHECK: smsww (%rdx) 
// CHECK: encoding: [0x0f,0x01,0x22]         
smsww (%rdx) 

// CHECK: strl %r13d 
// CHECK: encoding: [0x41,0x0f,0x00,0xcd]         
strl %r13d 

// CHECK: strw 485498096 
// CHECK: encoding: [0x0f,0x00,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
strw 485498096 

// CHECK: strw 64(%rdx) 
// CHECK: encoding: [0x0f,0x00,0x4a,0x40]         
strw 64(%rdx) 

// CHECK: strw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x4c,0x82,0x40]         
strw 64(%rdx,%rax,4) 

// CHECK: strw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x4c,0x82,0xc0]         
strw -64(%rdx,%rax,4) 

// CHECK: strw 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x00,0x4c,0x02,0x40]         
strw 64(%rdx,%rax) 

// CHECK: strw (%rdx) 
// CHECK: encoding: [0x0f,0x00,0x0a]         
strw (%rdx) 

// CHECK: verr 485498096 
// CHECK: encoding: [0x0f,0x00,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
verr 485498096 

// CHECK: verr 64(%rdx) 
// CHECK: encoding: [0x0f,0x00,0x62,0x40]         
verr 64(%rdx) 

// CHECK: verr 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x64,0x82,0x40]         
verr 64(%rdx,%rax,4) 

// CHECK: verr -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x64,0x82,0xc0]         
verr -64(%rdx,%rax,4) 

// CHECK: verr 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x00,0x64,0x02,0x40]         
verr 64(%rdx,%rax) 

// CHECK: verr %r11w 
// CHECK: encoding: [0x41,0x0f,0x00,0xe3]         
verr %r11w 

// CHECK: verr (%rdx) 
// CHECK: encoding: [0x0f,0x00,0x22]         
verr (%rdx) 

// CHECK: verw 485498096 
// CHECK: encoding: [0x0f,0x00,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
verw 485498096 

// CHECK: verw 64(%rdx) 
// CHECK: encoding: [0x0f,0x00,0x6a,0x40]         
verw 64(%rdx) 

// CHECK: verw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x6c,0x82,0x40]         
verw 64(%rdx,%rax,4) 

// CHECK: verw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x00,0x6c,0x82,0xc0]         
verw -64(%rdx,%rax,4) 

// CHECK: verw 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x00,0x6c,0x02,0x40]         
verw 64(%rdx,%rax) 

// CHECK: verw %r11w 
// CHECK: encoding: [0x41,0x0f,0x00,0xeb]         
verw %r11w 

// CHECK: verw (%rdx) 
// CHECK: encoding: [0x0f,0x00,0x2a]         
verw (%rdx) 

