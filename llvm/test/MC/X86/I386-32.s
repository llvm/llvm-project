// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: bsfw %ax, %ax 
// CHECK: encoding: [0x66,0x0f,0xbc,0xc0]        
bsfw %ax, %ax 

// CHECK: bsrw %ax, %ax 
// CHECK: encoding: [0x66,0x0f,0xbd,0xc0]        
bsrw %ax, %ax 

// CHECK: bsfl %eax, %eax 
// CHECK: encoding: [0x0f,0xbc,0xc0]        
bsfl %eax, %eax 

// CHECK: bsrl %eax, %eax 
// CHECK: encoding: [0x0f,0xbd,0xc0]        
bsrl %eax, %eax 

// CHECK: btcw $0, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xf8,0x00]        
btcw $0, %ax 

// CHECK: btcw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xf8,0xff]        
btcw $-1, %ax 

// CHECK: btcw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xf8,0xff]        
btcw $255, %ax 

// CHECK: btcw %ax, %ax 
// CHECK: encoding: [0x66,0x0f,0xbb,0xc0]        
btcw %ax, %ax 

// CHECK: btw $0, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xe0,0x00]        
btw $0, %ax 

// CHECK: btw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xe0,0xff]        
btw $-1, %ax 

// CHECK: btw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xe0,0xff]        
btw $255, %ax 

// CHECK: btw %ax, %ax 
// CHECK: encoding: [0x66,0x0f,0xa3,0xc0]        
btw %ax, %ax 

// CHECK: btrw $0, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xf0,0x00]        
btrw $0, %ax 

// CHECK: btrw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xf0,0xff]        
btrw $-1, %ax 

// CHECK: btrw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xf0,0xff]        
btrw $255, %ax 

// CHECK: btrw %ax, %ax 
// CHECK: encoding: [0x66,0x0f,0xb3,0xc0]        
btrw %ax, %ax 

// CHECK: btsw $0, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xe8,0x00]        
btsw $0, %ax 

// CHECK: btsw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xe8,0xff]        
btsw $-1, %ax 

// CHECK: btsw $255, %ax 
// CHECK: encoding: [0x66,0x0f,0xba,0xe8,0xff]        
btsw $255, %ax 

// CHECK: btsw %ax, %ax 
// CHECK: encoding: [0x66,0x0f,0xab,0xc0]        
btsw %ax, %ax 

// CHECK: btcl $0, %eax 
// CHECK: encoding: [0x0f,0xba,0xf8,0x00]        
btcl $0, %eax 

// CHECK: btcl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xf8,0xff]        
btcl $-1, %eax 

// CHECK: btcl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xf8,0xff]        
btcl $255, %eax 

// CHECK: btcl %eax, %eax 
// CHECK: encoding: [0x0f,0xbb,0xc0]        
btcl %eax, %eax 

// CHECK: btl $0, %eax 
// CHECK: encoding: [0x0f,0xba,0xe0,0x00]        
btl $0, %eax 

// CHECK: btl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xe0,0xff]        
btl $-1, %eax 

// CHECK: btl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xe0,0xff]        
btl $255, %eax 

// CHECK: btl %eax, %eax 
// CHECK: encoding: [0x0f,0xa3,0xc0]        
btl %eax, %eax 

// CHECK: btrl $0, %eax 
// CHECK: encoding: [0x0f,0xba,0xf0,0x00]        
btrl $0, %eax 

// CHECK: btrl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xf0,0xff]        
btrl $-1, %eax 

// CHECK: btrl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xf0,0xff]        
btrl $255, %eax 

// CHECK: btrl %eax, %eax 
// CHECK: encoding: [0x0f,0xb3,0xc0]        
btrl %eax, %eax 

// CHECK: btsl $0, %eax 
// CHECK: encoding: [0x0f,0xba,0xe8,0x00]        
btsl $0, %eax 

// CHECK: btsl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xe8,0xff]        
btsl $-1, %eax 

// CHECK: btsl $255, %eax 
// CHECK: encoding: [0x0f,0xba,0xe8,0xff]        
btsl $255, %eax 

// CHECK: btsl %eax, %eax 
// CHECK: encoding: [0x0f,0xab,0xc0]        
btsl %eax, %eax 

// CHECK: cmpsb %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0x26,0xa6]        
cmpsb %es:(%edi), %es:(%esi) 

// CHECK: cmpsl %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0x26,0xa7]        
cmpsl %es:(%edi), %es:(%esi) 

// CHECK: cmpsw %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0x66,0x26,0xa7]        
cmpsw %es:(%edi), %es:(%esi) 

// CHECK: insb %dx, %es:(%edi) 
// CHECK: encoding: [0x6c]        
insb %dx, %es:(%edi) 

// CHECK: insl %dx, %es:(%edi) 
// CHECK: encoding: [0x6d]        
insl %dx, %es:(%edi) 

// CHECK: insw %dx, %es:(%edi) 
// CHECK: encoding: [0x66,0x6d]        
insw %dx, %es:(%edi) 

// CHECK: iretl 
// CHECK: encoding: [0xcf]          
iretl 

// CHECK: iretw 
// CHECK: encoding: [0x66,0xcf]          
iretw 

// CHECK: jecxz 64 
// CHECK: encoding: [0xe3,A]         
jecxz 64 

// CHECK: lodsl %es:(%esi), %eax 
// CHECK: encoding: [0x26,0xad]        
lodsl %es:(%esi), %eax 

// CHECK: movsb %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0x26,0xa4]        
movsb %es:(%esi), %es:(%edi) 

// CHECK: movsl %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0x26,0xa5]        
movsl %es:(%esi), %es:(%edi) 

// CHECK: movsw %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0x66,0x26,0xa5]        
movsw %es:(%esi), %es:(%edi) 

// CHECK: outsb %es:(%esi), %dx 
// CHECK: encoding: [0x26,0x6e]        
outsb %es:(%esi), %dx 

// CHECK: outsl %es:(%esi), %dx 
// CHECK: encoding: [0x26,0x6f]        
outsl %es:(%esi), %dx 

// CHECK: outsw %es:(%esi), %dx 
// CHECK: encoding: [0x66,0x26,0x6f]        
outsw %es:(%esi), %dx 

// CHECK: popal 
// CHECK: encoding: [0x61]          
popal 

// CHECK: popaw 
// CHECK: encoding: [0x66,0x61]          
popaw 

// CHECK: popfl 
// CHECK: encoding: [0x9d]          
popfl 

// CHECK: popfw 
// CHECK: encoding: [0x66,0x9d]          
popfw 

// CHECK: pushal 
// CHECK: encoding: [0x60]          
pushal 

// CHECK: pushaw 
// CHECK: encoding: [0x66,0x60]          
pushaw 

// CHECK: pushfl 
// CHECK: encoding: [0x9c]          
pushfl 

// CHECK: pushfw 
// CHECK: encoding: [0x66,0x9c]          
pushfw 

// CHECK: rep cmpsb %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf3,0x26,0xa6]       
rep cmpsb %es:(%edi), %es:(%esi) 

// CHECK: rep cmpsl %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf3,0x26,0xa7]       
rep cmpsl %es:(%edi), %es:(%esi) 

// CHECK: rep cmpsw %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf3,0x66,0x26,0xa7]       
rep cmpsw %es:(%edi), %es:(%esi) 

// CHECK: rep insb %dx, %es:(%edi) 
// CHECK: encoding: [0xf3,0x6c]       
rep insb %dx, %es:(%edi) 

// CHECK: rep insl %dx, %es:(%edi) 
// CHECK: encoding: [0xf3,0x6d]       
rep insl %dx, %es:(%edi) 

// CHECK: rep insw %dx, %es:(%edi) 
// CHECK: encoding: [0xf3,0x66,0x6d]       
rep insw %dx, %es:(%edi) 

// CHECK: rep lodsl %es:(%esi), %eax 
// CHECK: encoding: [0xf3,0x26,0xad]       
rep lodsl %es:(%esi), %eax 

// CHECK: rep movsb %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf3,0x26,0xa4]       
rep movsb %es:(%esi), %es:(%edi) 

// CHECK: rep movsl %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf3,0x26,0xa5]       
rep movsl %es:(%esi), %es:(%edi) 

// CHECK: rep movsw %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf3,0x66,0x26,0xa5]       
rep movsw %es:(%esi), %es:(%edi) 

// CHECK: repne cmpsb %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf2,0x26,0xa6]       
repne cmpsb %es:(%edi), %es:(%esi) 

// CHECK: repne cmpsl %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf2,0x26,0xa7]       
repne cmpsl %es:(%edi), %es:(%esi) 

// CHECK: repne cmpsw %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf2,0x66,0x26,0xa7]       
repne cmpsw %es:(%edi), %es:(%esi) 

// CHECK: repne insb %dx, %es:(%edi) 
// CHECK: encoding: [0xf2,0x6c]       
repne insb %dx, %es:(%edi) 

// CHECK: repne insl %dx, %es:(%edi) 
// CHECK: encoding: [0xf2,0x6d]       
repne insl %dx, %es:(%edi) 

// CHECK: repne insw %dx, %es:(%edi) 
// CHECK: encoding: [0xf2,0x66,0x6d]       
repne insw %dx, %es:(%edi) 

// CHECK: repne lodsl %es:(%esi), %eax 
// CHECK: encoding: [0xf2,0x26,0xad]       
repne lodsl %es:(%esi), %eax 

// CHECK: repne movsb %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf2,0x26,0xa4]       
repne movsb %es:(%esi), %es:(%edi) 

// CHECK: repne movsl %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf2,0x26,0xa5]       
repne movsl %es:(%esi), %es:(%edi) 

// CHECK: repne movsw %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf2,0x66,0x26,0xa5]       
repne movsw %es:(%esi), %es:(%edi) 

// CHECK: repne outsb %es:(%esi), %dx 
// CHECK: encoding: [0xf2,0x26,0x6e]       
repne outsb %es:(%esi), %dx 

// CHECK: repne outsl %es:(%esi), %dx 
// CHECK: encoding: [0xf2,0x26,0x6f]       
repne outsl %es:(%esi), %dx 

// CHECK: repne outsw %es:(%esi), %dx 
// CHECK: encoding: [0xf2,0x66,0x26,0x6f]       
repne outsw %es:(%esi), %dx 

// CHECK: repne scasl %es:(%edi), %eax 
// CHECK: encoding: [0xf2,0xaf]       
repne scasl %es:(%edi), %eax 

// CHECK: repne stosl %eax, %es:(%edi) 
// CHECK: encoding: [0xf2,0xab]       
repne stosl %eax, %es:(%edi) 

// CHECK: rep outsb %es:(%esi), %dx 
// CHECK: encoding: [0xf3,0x26,0x6e]       
rep outsb %es:(%esi), %dx 

// CHECK: rep outsl %es:(%esi), %dx 
// CHECK: encoding: [0xf3,0x26,0x6f]       
rep outsl %es:(%esi), %dx 

// CHECK: rep outsw %es:(%esi), %dx 
// CHECK: encoding: [0xf3,0x66,0x26,0x6f]       
rep outsw %es:(%esi), %dx 

// CHECK: rep scasl %es:(%edi), %eax 
// CHECK: encoding: [0xf3,0xaf]       
rep scasl %es:(%edi), %eax 

// CHECK: rep stosl %eax, %es:(%edi) 
// CHECK: encoding: [0xf3,0xab]       
rep stosl %eax, %es:(%edi) 

// CHECK: scasl %es:(%edi), %eax 
// CHECK: encoding: [0xaf]        
scasl %es:(%edi), %eax 

// CHECK: seta -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x97,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
seta -485498096(%edx,%eax,4) 

// CHECK: seta 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x97,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
seta 485498096(%edx,%eax,4) 

// CHECK: seta 485498096(%edx) 
// CHECK: encoding: [0x0f,0x97,0x82,0xf0,0x1c,0xf0,0x1c]         
seta 485498096(%edx) 

// CHECK: seta 485498096 
// CHECK: encoding: [0x0f,0x97,0x05,0xf0,0x1c,0xf0,0x1c]         
seta 485498096 

// CHECK: seta 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x97,0x44,0x02,0x40]         
seta 64(%edx,%eax) 

// CHECK: setae -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x93,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setae -485498096(%edx,%eax,4) 

// CHECK: setae 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x93,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setae 485498096(%edx,%eax,4) 

// CHECK: setae 485498096(%edx) 
// CHECK: encoding: [0x0f,0x93,0x82,0xf0,0x1c,0xf0,0x1c]         
setae 485498096(%edx) 

// CHECK: setae 485498096 
// CHECK: encoding: [0x0f,0x93,0x05,0xf0,0x1c,0xf0,0x1c]         
setae 485498096 

// CHECK: setae 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x93,0x44,0x02,0x40]         
setae 64(%edx,%eax) 

// CHECK: seta (%edx) 
// CHECK: encoding: [0x0f,0x97,0x02]         
seta (%edx) 

// CHECK: setae (%edx) 
// CHECK: encoding: [0x0f,0x93,0x02]         
setae (%edx) 

// CHECK: setb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x92,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setb -485498096(%edx,%eax,4) 

// CHECK: setb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x92,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setb 485498096(%edx,%eax,4) 

// CHECK: setb 485498096(%edx) 
// CHECK: encoding: [0x0f,0x92,0x82,0xf0,0x1c,0xf0,0x1c]         
setb 485498096(%edx) 

// CHECK: setb 485498096 
// CHECK: encoding: [0x0f,0x92,0x05,0xf0,0x1c,0xf0,0x1c]         
setb 485498096 

// CHECK: setb 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x92,0x44,0x02,0x40]         
setb 64(%edx,%eax) 

// CHECK: setbe -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x96,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setbe -485498096(%edx,%eax,4) 

// CHECK: setbe 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x96,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setbe 485498096(%edx,%eax,4) 

// CHECK: setbe 485498096(%edx) 
// CHECK: encoding: [0x0f,0x96,0x82,0xf0,0x1c,0xf0,0x1c]         
setbe 485498096(%edx) 

// CHECK: setbe 485498096 
// CHECK: encoding: [0x0f,0x96,0x05,0xf0,0x1c,0xf0,0x1c]         
setbe 485498096 

// CHECK: setbe 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x96,0x44,0x02,0x40]         
setbe 64(%edx,%eax) 

// CHECK: setb (%edx) 
// CHECK: encoding: [0x0f,0x92,0x02]         
setb (%edx) 

// CHECK: setbe (%edx) 
// CHECK: encoding: [0x0f,0x96,0x02]         
setbe (%edx) 

// CHECK: sete -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x94,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
sete -485498096(%edx,%eax,4) 

// CHECK: sete 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x94,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
sete 485498096(%edx,%eax,4) 

// CHECK: sete 485498096(%edx) 
// CHECK: encoding: [0x0f,0x94,0x82,0xf0,0x1c,0xf0,0x1c]         
sete 485498096(%edx) 

// CHECK: sete 485498096 
// CHECK: encoding: [0x0f,0x94,0x05,0xf0,0x1c,0xf0,0x1c]         
sete 485498096 

// CHECK: sete 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x94,0x44,0x02,0x40]         
sete 64(%edx,%eax) 

// CHECK: sete (%edx) 
// CHECK: encoding: [0x0f,0x94,0x02]         
sete (%edx) 

// CHECK: setg -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9f,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setg -485498096(%edx,%eax,4) 

// CHECK: setg 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9f,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setg 485498096(%edx,%eax,4) 

// CHECK: setg 485498096(%edx) 
// CHECK: encoding: [0x0f,0x9f,0x82,0xf0,0x1c,0xf0,0x1c]         
setg 485498096(%edx) 

// CHECK: setg 485498096 
// CHECK: encoding: [0x         
setg 485498096 

// CHECK: setg 485498096 
// CHECK: encoding: [0x0f,0x9f,0x05,0xf0,0x1c,0xf0,0x1c]         
setg 485498096 

// CHECK: setg 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x9f,0x44,0x02,0x40]         
setg 64(%edx,%eax) 

// CHECK: setge -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9d,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setge -485498096(%edx,%eax,4) 

// CHECK: setge 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9d,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setge 485498096(%edx,%eax,4) 

// CHECK: setge 485498096(%edx) 
// CHECK: encoding: [0x0f,0x9d,0x82,0xf0,0x1c,0xf0,0x1c]         
setge 485498096(%edx) 

// CHECK: setge 485498096 
// CHECK: encoding: [0x0f,0x9d,0x05,0xf0,0x1c,0xf0,0x1c]         
setge 485498096 

// CHECK: setge 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x9d,0x44,0x02,0x40]         
setge 64(%edx,%eax) 

// CHECK: setg (%edx) 
// CHECK: encoding: [0x0f,0x9f,0x02]         
setg (%edx) 

// CHECK: setge (%edx) 
// CHECK: encoding: [0x0f,0x9d,0x02]         
setge (%edx) 

// CHECK: setl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9c,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setl -485498096(%edx,%eax,4) 

// CHECK: setl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9c,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setl 485498096(%edx,%eax,4) 

// CHECK: setl 485498096(%edx) 
// CHECK: encoding: [0x0f,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
setl 485498096(%edx) 

// CHECK: setl 485498096 
// CHECK: encoding: [0x0f,0x9c,0x05,0xf0,0x1c,0xf0,0x1c]         
setl 485498096 

// CHECK: setl 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x9c,0x44,0x02,0x40]         
setl 64(%edx,%eax) 

// CHECK: setle -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9e,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setle -485498096(%edx,%eax,4) 

// CHECK: setle 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9e,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setle 485498096(%edx,%eax,4) 

// CHECK: setle 485498096(%edx) 
// CHECK: encoding: [0x0f,0x9e,0x82,0xf0,0x1c,0xf0,0x1c]         
setle 485498096(%edx) 

// CHECK: setle 485498096 
// CHECK: encoding: [0x0f,0x9e,0x05,0xf0,0x1c,0xf0,0x1c]         
setle 485498096 

// CHECK: setle 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x9e,0x44,0x02,0x40]         
setle 64(%edx,%eax) 

// CHECK: setl (%edx) 
// CHECK: encoding: [0x0f,0x9c,0x02]         
setl (%edx) 

// CHECK: setle (%edx) 
// CHECK: encoding: [0x0f,0x9e,0x02]         
setle (%edx) 

// CHECK: setne -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x95,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setne -485498096(%edx,%eax,4) 

// CHECK: setne 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x95,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setne 485498096(%edx,%eax,4) 

// CHECK: setne 485498096(%edx) 
// CHECK: encoding: [0x0f,0x95,0x82,0xf0,0x1c,0xf0,0x1c]         
setne 485498096(%edx) 

// CHECK: setne 485498096 
// CHECK: encoding: [0x0f,0x95,0x05,0xf0,0x1c,0xf0,0x1c]         
setne 485498096 

// CHECK: setne 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x95,0x44,0x02,0x40]         
setne 64(%edx,%eax) 

// CHECK: setne (%edx) 
// CHECK: encoding: [0x0f,0x95,0x02]         
setne (%edx) 

// CHECK: setno -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x91,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setno -485498096(%edx,%eax,4) 

// CHECK: setno 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x91,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setno 485498096(%edx,%eax,4) 

// CHECK: setno 485498096(%edx) 
// CHECK: encoding: [0x0f,0x91,0x82,0xf0,0x1c,0xf0,0x1c]         
setno 485498096(%edx) 

// CHECK: setno 485498096 
// CHECK: encoding: [0x0f,0x91,0x05,0xf0,0x1c,0xf0,0x1c]         
setno 485498096 

// CHECK: setno 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x91,0x44,0x02,0x40]         
setno 64(%edx,%eax) 

// CHECK: setno (%edx) 
// CHECK: encoding: [0x0f,0x91,0x02]         
setno (%edx) 

// CHECK: setnp -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9b,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setnp -485498096(%edx,%eax,4) 

// CHECK: setnp 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9b,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setnp 485498096(%edx,%eax,4) 

// CHECK: setnp 485498096(%edx) 
// CHECK: encoding: [0x0f,0x9b,0x82,0xf0,0x1c,0xf0,0x1c]         
setnp 485498096(%edx) 

// CHECK: setnp 485498096 
// CHECK: encoding: [0x0f,0x9b,0x05,0xf0,0x1c,0xf0,0x1c]         
setnp 485498096 

// CHECK: setnp 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x9b,0x44,0x02,0x40]         
setnp 64(%edx,%eax) 

// CHECK: setnp (%edx) 
// CHECK: encoding: [0x0f,0x9b,0x02]         
setnp (%edx) 

// CHECK: setns -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x99,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setns -485498096(%edx,%eax,4) 

// CHECK: setns 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x99,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setns 485498096(%edx,%eax,4) 

// CHECK: setns 485498096(%edx) 
// CHECK: encoding: [0x0f,0x99,0x82,0xf0,0x1c,0xf0,0x1c]         
setns 485498096(%edx) 

// CHECK: setns 485498096 
// CHECK: encoding: [0x0f,0x99,0x05,0xf0,0x1c,0xf0,0x1c]         
setns 485498096 

// CHECK: setns 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x99,0x44,0x02,0x40]         
setns 64(%edx,%eax) 

// CHECK: setns (%edx) 
// CHECK: encoding: [0x0f,0x99,0x02]         
setns (%edx) 

// CHECK: seto -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x90,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
seto -485498096(%edx,%eax,4) 

// CHECK: seto 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x90,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
seto 485498096(%edx,%eax,4) 

// CHECK: seto 485498096(%edx) 
// CHECK: encoding: [0x0f,0x90,0x82,0xf0,0x1c,0xf0,0x1c]         
seto 485498096(%edx) 

// CHECK: seto 485498096 
// CHECK: encoding: [0x0f,0x90,0x05,0xf0,0x1c,0xf0,0x1c]         
seto 485498096 

// CHECK: seto 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x90,0x44,0x02,0x40]         
seto 64(%edx,%eax) 

// CHECK: seto (%edx) 
// CHECK: encoding: [0x0f,0x90,0x02]         
seto (%edx) 

// CHECK: setp -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9a,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
setp -485498096(%edx,%eax,4) 

// CHECK: setp 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x9a,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
setp 485498096(%edx,%eax,4) 

// CHECK: setp 485498096(%edx) 
// CHECK: encoding: [0x0f,0x9a,0x82,0xf0,0x1c,0xf0,0x1c]         
setp 485498096(%edx) 

// CHECK: setp 485498096 
// CHECK: encoding: [0x0f,0x9a,0x05,0xf0,0x1c,0xf0,0x1c]         
setp 485498096 

// CHECK: setp 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x9a,0x44,0x02,0x40]         
setp 64(%edx,%eax) 

// CHECK: setp (%edx) 
// CHECK: encoding: [0x0f,0x9a,0x02]         
setp (%edx) 

// CHECK: sets -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x98,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
sets -485498096(%edx,%eax,4) 

// CHECK: sets 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x98,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
sets 485498096(%edx,%eax,4) 

// CHECK: sets 485498096(%edx) 
// CHECK: encoding: [0x0f,0x98,0x82,0xf0,0x1c,0xf0,0x1c]         
sets 485498096(%edx) 

// CHECK: sets 485498096 
// CHECK: encoding: [0x0f,0x98,0x05,0xf0,0x1c,0xf0,0x1c]         
sets 485498096 

// CHECK: sets 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x98,0x44,0x02,0x40]         
sets 64(%edx,%eax) 

// CHECK: sets (%edx) 
// CHECK: encoding: [0x0f,0x98,0x02]         
sets (%edx) 

// CHECK: shldl $0, %eax, %eax 
// CHECK: encoding: [0x0f,0xa4,0xc0,0x00]       
shldl $0, %eax, %eax 

// CHECK: shldl %cl, %eax, %eax 
// CHECK: encoding: [0x0f,0xa5,0xc0]       
shldl %cl, %eax, %eax 

// CHECK: shrdl $0, %eax, %eax 
// CHECK: encoding: [0x0f,0xac,0xc0,0x00]       
shrdl $0, %eax, %eax 

// CHECK: shrdl %cl, %eax, %eax 
// CHECK: encoding: [0x0f,0xad,0xc0]       
shrdl %cl, %eax, %eax 

// CHECK: stosl %eax, %es:(%edi) 
// CHECK: encoding: [0xab]        
stosl %eax, %es:(%edi) 

