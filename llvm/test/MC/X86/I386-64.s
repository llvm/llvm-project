// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: bsfw %r13w, %r13w 
// CHECK: encoding: [0x66,0x45,0x0f,0xbc,0xed]        
bsfw %r13w, %r13w 

// CHECK: bsrw %r13w, %r13w 
// CHECK: encoding: [0x66,0x45,0x0f,0xbd,0xed]        
bsrw %r13w, %r13w 

// CHECK: bsfl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xbc,0xed]        
bsfl %r13d, %r13d 

// CHECK: bsrl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xbd,0xed]        
bsrl %r13d, %r13d 

// CHECK: bsfq %r13, %r13 
// CHECK: encoding: [0x4d,0x0f,0xbc,0xed]        
bsfq %r13, %r13 

// CHECK: bsrq %r13, %r13 
// CHECK: encoding: [0x4d,0x0f,0xbd,0xed]        
bsrq %r13, %r13 


// CHECK: btcw $0, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xfd,0x00]        
btcw $0, %r13w 

// CHECK: btcw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xfd,0xff]        
btcw $-1, %r13w 

// CHECK: btcw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xfd,0xff]        
btcw $255, %r13w 

// CHECK: btcw %r13w, %r13w 
// CHECK: encoding: [0x66,0x45,0x0f,0xbb,0xed]        
btcw %r13w, %r13w 

// CHECK: btw $0, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xe5,0x00]        
btw $0, %r13w 

// CHECK: btw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xe5,0xff]        
btw $-1, %r13w 

// CHECK: btw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xe5,0xff]        
btw $255, %r13w 

// CHECK: btw %r13w, %r13w 
// CHECK: encoding: [0x66,0x45,0x0f,0xa3,0xed]        
btw %r13w, %r13w 

// CHECK: btrw $0, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xf5,0x00]        
btrw $0, %r13w 

// CHECK: btrw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xf5,0xff]        
btrw $-1, %r13w 

// CHECK: btrw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xf5,0xff]        
btrw $255, %r13w 

// CHECK: btrw %r13w, %r13w 
// CHECK: encoding: [0x66,0x45,0x0f,0xb3,0xed]        
btrw %r13w, %r13w 

// CHECK: btsw $0, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xed,0x00]        
btsw $0, %r13w 

// CHECK: btsw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xed,0xff]        
btsw $-1, %r13w 

// CHECK: btsw $255, %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xba,0xed,0xff]        
btsw $255, %r13w 

// CHECK: btsw %r13w, %r13w 
// CHECK: encoding: [0x66,0x45,0x0f,0xab,0xed]        
btsw %r13w, %r13w 

// CHECK: btcl $0, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xfd,0x00]        
btcl $0, %r13d 

// CHECK: btcl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xfd,0xff]        
btcl $-1, %r13d 

// CHECK: btcl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xfd,0xff]        
btcl $255, %r13d 

// CHECK: btcl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xbb,0xed]        
btcl %r13d, %r13d 

// CHECK: btl $0, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xe5,0x00]        
btl $0, %r13d 

// CHECK: btl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xe5,0xff]        
btl $-1, %r13d 

// CHECK: btl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xe5,0xff]        
btl $255, %r13d 

// CHECK: btl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xa3,0xed]        
btl %r13d, %r13d 

// CHECK: btrl $0, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xf5,0x00]        
btrl $0, %r13d 

// CHECK: btrl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xf5,0xff]        
btrl $-1, %r13d 

// CHECK: btrl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xf5,0xff]        
btrl $255, %r13d 

// CHECK: btrl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xb3,0xed]        
btrl %r13d, %r13d 

// CHECK: btsl $0, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xed,0x00]        
btsl $0, %r13d 

// CHECK: btsl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xed,0xff]        
btsl $-1, %r13d 

// CHECK: btsl $255, %r13d 
// CHECK: encoding: [0x41,0x0f,0xba,0xed,0xff]        
btsl $255, %r13d 

// CHECK: btsl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xab,0xed]        
btsl %r13d, %r13d 

// CHECK: btcq $0, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xfd,0x00]        
btcq $0, %r13 

// CHECK: btcq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xfd,0xff]        
btcq $-1, %r13 

// CHECK: btcq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xfd,0xff]        
btcq $255, %r13 

// CHECK: btcq %r13, %r13 
// CHECK: encoding: [0x4d,0x0f,0xbb,0xed]        
btcq %r13, %r13 

// CHECK: btq $0, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xe5,0x00]        
btq $0, %r13 

// CHECK: btq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xe5,0xff]        
btq $-1, %r13 

// CHECK: btq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xe5,0xff]        
btq $255, %r13 

// CHECK: btq %r13, %r13 
// CHECK: encoding: [0x4d,0x0f,0xa3,0xed]        
btq %r13, %r13 

// CHECK: btrq $0, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xf5,0x00]        
btrq $0, %r13 

// CHECK: btrq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xf5,0xff]        
btrq $-1, %r13 

// CHECK: btrq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xf5,0xff]        
btrq $255, %r13 

// CHECK: btrq %r13, %r13 
// CHECK: encoding: [0x4d,0x0f,0xb3,0xed]        
btrq %r13, %r13 

// CHECK: btsq $0, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xed,0x00]        
btsq $0, %r13 

// CHECK: btsq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xed,0xff]        
btsq $-1, %r13 

// CHECK: btsq $255, %r13 
// CHECK: encoding: [0x49,0x0f,0xba,0xed,0xff]        
btsq $255, %r13 

// CHECK: btsq %r13, %r13 
// CHECK: encoding: [0x4d,0x0f,0xab,0xed]        
btsq %r13, %r13 

// CHECK: cmpsb %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x65,0xa6]        
cmpsb %es:(%rdi), %gs:(%rsi) 

// CHECK: cmpsl %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x65,0xa7]        
cmpsl %es:(%rdi), %gs:(%rsi) 

// CHECK: cmpsq %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x48,0x65,0xa7]        
cmpsq %es:(%rdi), %gs:(%rsi) 

// CHECK: cmpsw %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x66,0x65,0xa7]        
cmpsw %es:(%rdi), %gs:(%rsi) 

// CHECK: insb %dx, %es:(%rdi) 
// CHECK: encoding: [0x6c]        
insb %dx, %es:(%rdi) 

// CHECK: insl %dx, %es:(%rdi) 
// CHECK: encoding: [0x6d]        
insl %dx, %es:(%rdi) 

// CHECK: insw %dx, %es:(%rdi) 
// CHECK: encoding: [0x66,0x6d]        
insw %dx, %es:(%rdi) 

// CHECK: iretl 
// CHECK: encoding: [0xcf]          
iretl 

// CHECK: iretq 
// CHECK: encoding: [0x48,0xcf]          
iretq 

// CHECK: iretw 
// CHECK: encoding: [0x66,0xcf]          
iretw 

// CHECK: lodsl %gs:(%rsi), %eax 
// CHECK: encoding: [0x65,0xad]        
lodsl %gs:(%rsi), %eax 

// CHECK: lzcntl %r13d, %r13d 
// CHECK: encoding: [0xf3,0x45,0x0f,0xbd,0xed]        
lzcntl %r13d, %r13d 

// CHECK: movsb %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x65,0xa4]        
movsb %gs:(%rsi), %es:(%rdi) 

// CHECK: movsbl 485498096, %r13d 
// CHECK: encoding: [0x44,0x0f,0xbe,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
movsbl 485498096, %r13d 

// CHECK: movsbl 64(%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbe,0x6a,0x40]        
movsbl 64(%rdx), %r13d 

// CHECK: movsbl 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbe,0x6c,0x82,0x40]        
movsbl 64(%rdx,%rax,4), %r13d 

// CHECK: movsbl -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbe,0x6c,0x82,0xc0]        
movsbl -64(%rdx,%rax,4), %r13d 

// CHECK: movsbl 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbe,0x6c,0x02,0x40]        
movsbl 64(%rdx,%rax), %r13d 

// CHECK: movsbl %r11b, %r13d 
// CHECK: encoding: [0x45,0x0f,0xbe,0xeb]        
movsbl %r11b, %r13d 

// CHECK: movsbl %r14b, %r13d 
// CHECK: encoding: [0x45,0x0f,0xbe,0xee]        
movsbl %r14b, %r13d 

// CHECK: movsbl (%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbe,0x2a]        
movsbl (%rdx), %r13d 

// CHECK: movsl %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x65,0xa5]        
movsl %gs:(%rsi), %es:(%rdi) 

// CHECK: movsq %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x48,0x65,0xa5]        
movsq %gs:(%rsi), %es:(%rdi) 

// CHECK: movsw %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x66,0x65,0xa5]        
movsw %gs:(%rsi), %es:(%rdi) 

// CHECK: movswl 485498096, %r13d 
// CHECK: encoding: [0x44,0x0f,0xbf,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
movswl 485498096, %r13d 

// CHECK: movswl 64(%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbf,0x6a,0x40]        
movswl 64(%rdx), %r13d 

// CHECK: movswl 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbf,0x6c,0x82,0x40]        
movswl 64(%rdx,%rax,4), %r13d 

// CHECK: movswl -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbf,0x6c,0x82,0xc0]        
movswl -64(%rdx,%rax,4), %r13d 

// CHECK: movswl 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbf,0x6c,0x02,0x40]        
movswl 64(%rdx,%rax), %r13d 

// CHECK: movswl %r11w, %r13d 
// CHECK: encoding: [0x45,0x0f,0xbf,0xeb]        
movswl %r11w, %r13d 

// CHECK: movswl %r14w, %r13d 
// CHECK: encoding: [0x45,0x0f,0xbf,0xee]        
movswl %r14w, %r13d 

// CHECK: movswl (%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xbf,0x2a]        
movswl (%rdx), %r13d 

// CHECK: movzbl 485498096, %r13d 
// CHECK: encoding: [0x44,0x0f,0xb6,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
movzbl 485498096, %r13d 

// CHECK: movzbl 64(%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb6,0x6a,0x40]        
movzbl 64(%rdx), %r13d 

// CHECK: movzbl 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb6,0x6c,0x82,0x40]        
movzbl 64(%rdx,%rax,4), %r13d 

// CHECK: movzbl -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb6,0x6c,0x82,0xc0]        
movzbl -64(%rdx,%rax,4), %r13d 

// CHECK: movzbl 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb6,0x6c,0x02,0x40]        
movzbl 64(%rdx,%rax), %r13d 

// CHECK: movzbl %r11b, %r13d 
// CHECK: encoding: [0x45,0x0f,0xb6,0xeb]        
movzbl %r11b, %r13d 

// CHECK: movzbl %r14b, %r13d 
// CHECK: encoding: [0x45,0x0f,0xb6,0xee]        
movzbl %r14b, %r13d 

// CHECK: movzbl (%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb6,0x2a]        
movzbl (%rdx), %r13d 

// CHECK: movzwl 485498096, %r13d 
// CHECK: encoding: [0x44,0x0f,0xb7,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
movzwl 485498096, %r13d 

// CHECK: movzwl 64(%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb7,0x6a,0x40]        
movzwl 64(%rdx), %r13d 

// CHECK: movzwl 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb7,0x6c,0x82,0x40]        
movzwl 64(%rdx,%rax,4), %r13d 

// CHECK: movzwl -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb7,0x6c,0x82,0xc0]        
movzwl -64(%rdx,%rax,4), %r13d 

// CHECK: movzwl 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb7,0x6c,0x02,0x40]        
movzwl 64(%rdx,%rax), %r13d 

// CHECK: movzwl %r11w, %r13d 
// CHECK: encoding: [0x45,0x0f,0xb7,0xeb]        
movzwl %r11w, %r13d 

// CHECK: movzwl %r14w, %r13d 
// CHECK: encoding: [0x45,0x0f,0xb7,0xee]        
movzwl %r14w, %r13d 

// CHECK: movzwl (%rdx), %r13d 
// CHECK: encoding: [0x44,0x0f,0xb7,0x2a]        
movzwl (%rdx), %r13d 

// CHECK: outsb %gs:(%rsi), %dx 
// CHECK: encoding: [0x65,0x6e]        
outsb %gs:(%rsi), %dx 

// CHECK: outsl %gs:(%rsi), %dx 
// CHECK: encoding: [0x65,0x6f]        
outsl %gs:(%rsi), %dx 

// CHECK: outsw %gs:(%rsi), %dx 
// CHECK: encoding: [0x66,0x65,0x6f]        
outsw %gs:(%rsi), %dx 

// CHECK: rep cmpsb %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x65,0xa6]       
rep cmpsb %es:(%rdi), %gs:(%rsi) 

// CHECK: rep cmpsl %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x65,0xa7]       
rep cmpsl %es:(%rdi), %gs:(%rsi) 

// CHECK: rep cmpsq %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x48,0x65,0xa7]       
rep cmpsq %es:(%rdi), %gs:(%rsi) 

// CHECK: rep cmpsw %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x66,0x65,0xa7]       
rep cmpsw %es:(%rdi), %gs:(%rsi) 

// CHECK: rep insb %dx, %es:(%rdi) 
// CHECK: encoding: [0xf3,0x6c]       
rep insb %dx, %es:(%rdi) 

// CHECK: rep insl %dx, %es:(%rdi) 
// CHECK: encoding: [0xf3,0x6d]       
rep insl %dx, %es:(%rdi) 

// CHECK: rep insw %dx, %es:(%rdi) 
// CHECK: encoding: [0xf3,0x66,0x6d]       
rep insw %dx, %es:(%rdi) 

// CHECK: rep lodsl %gs:(%rsi), %eax 
// CHECK: encoding: [0xf3,0x65,0xad]       
rep lodsl %gs:(%rsi), %eax 

// CHECK: rep movsb %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x65,0xa4]       
rep movsb %gs:(%rsi), %es:(%rdi) 

// CHECK: rep movsl %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x65,0xa5]       
rep movsl %gs:(%rsi), %es:(%rdi) 

// CHECK: rep movsq %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x48,0x65,0xa5]       
rep movsq %gs:(%rsi), %es:(%rdi) 

// CHECK: rep movsw %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x66,0x65,0xa5]       
rep movsw %gs:(%rsi), %es:(%rdi) 

// CHECK: repne cmpsb %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x65,0xa6]       
repne cmpsb %es:(%rdi), %gs:(%rsi) 

// CHECK: repne cmpsl %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x65,0xa7]       
repne cmpsl %es:(%rdi), %gs:(%rsi) 

// CHECK: repne cmpsq %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x48,0x65,0xa7]       
repne cmpsq %es:(%rdi), %gs:(%rsi) 

// CHECK: repne cmpsw %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x66,0x65,0xa7]       
repne cmpsw %es:(%rdi), %gs:(%rsi) 

// CHECK: repne insb %dx, %es:(%rdi) 
// CHECK: encoding: [0xf2,0x6c]       
repne insb %dx, %es:(%rdi) 

// CHECK: repne insl %dx, %es:(%rdi) 
// CHECK: encoding: [0xf2,0x6d]       
repne insl %dx, %es:(%rdi) 

// CHECK: repne insw %dx, %es:(%rdi) 
// CHECK: encoding: [0xf2,0x66,0x6d]       
repne insw %dx, %es:(%rdi) 

// CHECK: repne lodsl %gs:(%rsi), %eax 
// CHECK: encoding: [0xf2,0x65,0xad]       
repne lodsl %gs:(%rsi), %eax 

// CHECK: repne movsb %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x65,0xa4]       
repne movsb %gs:(%rsi), %es:(%rdi) 

// CHECK: repne movsl %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x65,0xa5]       
repne movsl %gs:(%rsi), %es:(%rdi) 

// CHECK: repne movsq %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x48,0x65,0xa5]       
repne movsq %gs:(%rsi), %es:(%rdi) 

// CHECK: repne movsw %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x66,0x65,0xa5]       
repne movsw %gs:(%rsi), %es:(%rdi) 

// CHECK: repne outsb %gs:(%rsi), %dx 
// CHECK: encoding: [0xf2,0x65,0x6e]       
repne outsb %gs:(%rsi), %dx 

// CHECK: repne outsl %gs:(%rsi), %dx 
// CHECK: encoding: [0xf2,0x65,0x6f]       
repne outsl %gs:(%rsi), %dx 

// CHECK: repne outsw %gs:(%rsi), %dx 
// CHECK: encoding: [0xf2,0x66,0x65,0x6f]       
repne outsw %gs:(%rsi), %dx 

// CHECK: repne scasl %es:(%rdi), %eax 
// CHECK: encoding: [0xf2,0xaf]       
repne scasl %es:(%rdi), %eax 

// CHECK: repne stosl %eax, %es:(%rdi) 
// CHECK: encoding: [0xf2,0xab]       
repne stosl %eax, %es:(%rdi) 

// CHECK: rep outsb %gs:(%rsi), %dx 
// CHECK: encoding: [0xf3,0x65,0x6e]       
rep outsb %gs:(%rsi), %dx 

// CHECK: rep outsl %gs:(%rsi), %dx 
// CHECK: encoding: [0xf3,0x65,0x6f]       
rep outsl %gs:(%rsi), %dx 

// CHECK: rep outsw %gs:(%rsi), %dx 
// CHECK: encoding: [0xf3,0x66,0x65,0x6f]       
rep outsw %gs:(%rsi), %dx 

// CHECK: rep scasl %es:(%rdi), %eax 
// CHECK: encoding: [0xf3,0xaf]       
rep scasl %es:(%rdi), %eax 

// CHECK: rep stosl %eax, %es:(%rdi) 
// CHECK: encoding: [0xf3,0xab]       
rep stosl %eax, %es:(%rdi) 

// CHECK: scasl %es:(%rdi), %eax 
// CHECK: encoding: [0xaf]        
scasl %es:(%rdi), %eax 

// CHECK: seta 485498096 
// CHECK: encoding: [0x0f,0x97,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
seta 485498096 

// CHECK: seta 64(%rdx) 
// CHECK: encoding: [0x0f,0x97,0x42,0x40]         
seta 64(%rdx) 

// CHECK: seta 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x97,0x44,0x82,0x40]         
seta 64(%rdx,%rax,4) 

// CHECK: seta -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x97,0x44,0x82,0xc0]         
seta -64(%rdx,%rax,4) 

// CHECK: seta 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x97,0x44,0x02,0x40]         
seta 64(%rdx,%rax) 

// CHECK: setae 485498096 
// CHECK: encoding: [0x0f,0x93,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setae 485498096 

// CHECK: setae 64(%rdx) 
// CHECK: encoding: [0x0f,0x93,0x42,0x40]         
setae 64(%rdx) 

// CHECK: setae 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x93,0x44,0x82,0x40]         
setae 64(%rdx,%rax,4) 

// CHECK: setae -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x93,0x44,0x82,0xc0]         
setae -64(%rdx,%rax,4) 

// CHECK: setae 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x93,0x44,0x02,0x40]         
setae 64(%rdx,%rax) 

// CHECK: setae %r14b 
// CHECK: encoding: [0x41,0x0f,0x93,0xc6]         
setae %r14b 

// CHECK: setae (%rdx) 
// CHECK: encoding: [0x0f,0x93,0x02]         
setae (%rdx) 

// CHECK: seta %r14b 
// CHECK: encoding: [0x41,0x0f,0x97,0xc6]         
seta %r14b 

// CHECK: seta (%rdx) 
// CHECK: encoding: [0x0f,0x97,0x02]         
seta (%rdx) 

// CHECK: setb 485498096 
// CHECK: encoding: [0x0f,0x92,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setb 485498096 

// CHECK: setb 64(%rdx) 
// CHECK: encoding: [0x0f,0x92,0x42,0x40]         
setb 64(%rdx) 

// CHECK: setb 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x92,0x44,0x82,0x40]         
setb 64(%rdx,%rax,4) 

// CHECK: setb -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x92,0x44,0x82,0xc0]         
setb -64(%rdx,%rax,4) 

// CHECK: setb 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x92,0x44,0x02,0x40]         
setb 64(%rdx,%rax) 

// CHECK: setbe 485498096 
// CHECK: encoding: [0x0f,0x96,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setbe 485498096 

// CHECK: setbe 64(%rdx) 
// CHECK: encoding: [0x0f,0x96,0x42,0x40]         
setbe 64(%rdx) 

// CHECK: setbe 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x96,0x44,0x82,0x40]         
setbe 64(%rdx,%rax,4) 

// CHECK: setbe -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x96,0x44,0x82,0xc0]         
setbe -64(%rdx,%rax,4) 

// CHECK: setbe 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x96,0x44,0x02,0x40]         
setbe 64(%rdx,%rax) 

// CHECK: setbe %r14b 
// CHECK: encoding: [0x41,0x0f,0x96,0xc6]         
setbe %r14b 

// CHECK: setbe (%rdx) 
// CHECK: encoding: [0x0f,0x96,0x02]         
setbe (%rdx) 

// CHECK: setb %r14b 
// CHECK: encoding: [0x41,0x0f,0x92,0xc6]         
setb %r14b 

// CHECK: setb (%rdx) 
// CHECK: encoding: [0x0f,0x92,0x02]         
setb (%rdx) 

// CHECK: sete 485498096 
// CHECK: encoding: [0x0f,0x94,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
sete 485498096 

// CHECK: sete 64(%rdx) 
// CHECK: encoding: [0x0f,0x94,0x42,0x40]         
sete 64(%rdx) 

// CHECK: sete 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x94,0x44,0x82,0x40]         
sete 64(%rdx,%rax,4) 

// CHECK: sete -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x94,0x44,0x82,0xc0]         
sete -64(%rdx,%rax,4) 

// CHECK: sete 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x94,0x44,0x02,0x40]         
sete 64(%rdx,%rax) 

// CHECK: sete %r14b 
// CHECK: encoding: [0x41,0x0f,0x94,0xc6]         
sete %r14b 

// CHECK: sete (%rdx) 
// CHECK: encoding: [0x0f,0x94,0x02]         
sete (%rdx) 

// CHECK: setg 485498096 
// CHECK: encoding: [0x0f,0x9f,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setg 485498096 

// CHECK: setg 64(%rdx) 
// CHECK: encoding: [0x0f,0x9f,0x42,0x40]         
setg 64(%rdx) 

// CHECK: setg 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9f,0x44,0x82,0x40]         
setg 64(%rdx,%rax,4) 

// CHECK: setg -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9f,0x44,0x82,0xc0]         
setg -64(%rdx,%rax,4) 

// CHECK: setg 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x9f,0x44,0x02,0x40]         
setg 64(%rdx,%rax) 

// CHECK: setge 485498096 
// CHECK: encoding: [0x0f,0x9d,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setge 485498096 

// CHECK: setge 64(%rdx) 
// CHECK: encoding: [0x0f,0x9d,0x42,0x40]         
setge 64(%rdx) 

// CHECK: setge 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9d,0x44,0x82,0x40]         
setge 64(%rdx,%rax,4) 

// CHECK: setge -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9d,0x44,0x82,0xc0]         
setge -64(%rdx,%rax,4) 

// CHECK: setge 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x9d,0x44,0x02,0         
setge 64(%rdx,%rax) 

// CHECK: setge 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x9d,0x44,0x02,0x40]         
setge 64(%rdx,%rax) 

// CHECK: setge %r14b 
// CHECK: encoding: [0x41,0x0f,0x9d,0xc6]         
setge %r14b 

// CHECK: setge (%rdx) 
// CHECK: encoding: [0x0f,0x9d,0x02]         
setge (%rdx) 

// CHECK: setg %r14b 
// CHECK: encoding: [0x41,0x0f,0x9f,0xc6]         
setg %r14b 

// CHECK: setg (%rdx) 
// CHECK: encoding: [0x0f,0x9f,0x02]         
setg (%rdx) 

// CHECK: setl 485498096 
// CHECK: encoding: [0x0f,0x9c,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setl 485498096 

// CHECK: setl 64(%rdx) 
// CHECK: encoding: [0x0f,0x9c,0x42,0x40]         
setl 64(%rdx) 

// CHECK: setl 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9c,0x44,0x82,0x40]         
setl 64(%rdx,%rax,4) 

// CHECK: setl -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9c,0x44,0x82,0xc0]         
setl -64(%rdx,%rax,4) 

// CHECK: setl 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x9c,0x44,0x02,0x40]         
setl 64(%rdx,%rax) 

// CHECK: setle 485498096 
// CHECK: encoding: [0x0f,0x9e,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setle 485498096 

// CHECK: setle 64(%rdx) 
// CHECK: encoding: [0x0f,0x9e,0x42,0x40]         
setle 64(%rdx) 

// CHECK: setle 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9e,0x44,0x82,0x40]         
setle 64(%rdx,%rax,4) 

// CHECK: setle -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9e,0x44,0x82,0xc0]         
setle -64(%rdx,%rax,4) 

// CHECK: setle 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x9e,0x44,0x02,0x40]         
setle 64(%rdx,%rax) 

// CHECK: setle %r14b 
// CHECK: encoding: [0x41,0x0f,0x9e,0xc6]         
setle %r14b 

// CHECK: setle (%rdx) 
// CHECK: encoding: [0x0f,0x9e,0x02]         
setle (%rdx) 

// CHECK: setl %r14b 
// CHECK: encoding: [0x41,0x0f,0x9c,0xc6]         
setl %r14b 

// CHECK: setl (%rdx) 
// CHECK: encoding: [0x0f,0x9c,0x02]         
setl (%rdx) 

// CHECK: setne 485498096 
// CHECK: encoding: [0x0f,0x95,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setne 485498096 

// CHECK: setne 64(%rdx) 
// CHECK: encoding: [0x0f,0x95,0x42,0x40]         
setne 64(%rdx) 

// CHECK: setne 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x95,0x44,0x82,0x40]         
setne 64(%rdx,%rax,4) 

// CHECK: setne -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x95,0x44,0x82,0xc0]         
setne -64(%rdx,%rax,4) 

// CHECK: setne 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x95,0x44,0x02,0x40]         
setne 64(%rdx,%rax) 

// CHECK: setne %r14b 
// CHECK: encoding: [0x41,0x0f,0x95,0xc6]         
setne %r14b 

// CHECK: setne (%rdx) 
// CHECK: encoding: [0x0f,0x95,0x02]         
setne (%rdx) 

// CHECK: setno 485498096 
// CHECK: encoding: [0x0f,0x91,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setno 485498096 

// CHECK: setno 64(%rdx) 
// CHECK: encoding: [0x0f,0x91,0x42,0x40]         
setno 64(%rdx) 

// CHECK: setno 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x91,0x44,0x82,0x40]         
setno 64(%rdx,%rax,4) 

// CHECK: setno -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x91,0x44,0x82,0xc0]         
setno -64(%rdx,%rax,4) 

// CHECK: setno 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x91,0x44,0x02,0x40]         
setno 64(%rdx,%rax) 

// CHECK: setno %r14b 
// CHECK: encoding: [0x41,0x0f,0x91,0xc6]         
setno %r14b 

// CHECK: setno (%rdx) 
// CHECK: encoding: [0x0f,0x91,0x02]         
setno (%rdx) 

// CHECK: setnp 485498096 
// CHECK: encoding: [0x0f,0x9b,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setnp 485498096 

// CHECK: setnp 64(%rdx) 
// CHECK: encoding: [0x0f,0x9b,0x42,0x40]         
setnp 64(%rdx) 

// CHECK: setnp 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9b,0x44,0x82,0x40]         
setnp 64(%rdx,%rax,4) 

// CHECK: setnp -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9b,0x44,0x82,0xc0]         
setnp -64(%rdx,%rax,4) 

// CHECK: setnp 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x9b,0x44,0x02,0x40]         
setnp 64(%rdx,%rax) 

// CHECK: setnp %r14b 
// CHECK: encoding: [0x41,0x0f,0x9b,0xc6]         
setnp %r14b 

// CHECK: setnp (%rdx) 
// CHECK: encoding: [0x0f,0x9b,0x02]         
setnp (%rdx) 

// CHECK: setns 485498096 
// CHECK: encoding: [0x0f,0x99,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setns 485498096 

// CHECK: setns 64(%rdx) 
// CHECK: encoding: [0x0f,0x99,0x42,0x40]         
setns 64(%rdx) 

// CHECK: setns 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x99,0x44,0x82,0x40]         
setns 64(%rdx,%rax,4) 

// CHECK: setns -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x99,0x44,0x82,0xc0]         
setns -64(%rdx,%rax,4) 

// CHECK: setns 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x99,0x44,0x02,0x40]         
setns 64(%rdx,%rax) 

// CHECK: setns %r14b 
// CHECK: encoding: [0x41,0x0f,0x99,0xc6]         
setns %r14b 

// CHECK: setns (%rdx) 
// CHECK: encoding: [0x0f,0x99,0x02]         
setns (%rdx) 

// CHECK: seto 485498096 
// CHECK: encoding: [0x0f,0x90,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
seto 485498096 

// CHECK: seto 64(%rdx) 
// CHECK: encoding: [0x0f,0x90,0x42,0x40]         
seto 64(%rdx) 

// CHECK: seto 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x90,0x44,0x82,0x40]         
seto 64(%rdx,%rax,4) 

// CHECK: seto -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x90,0x44,0x82,0xc0]         
seto -64(%rdx,%rax,4) 

// CHECK: seto 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x90,0x44,0x02,0x40]         
seto 64(%rdx,%rax) 

// CHECK: seto %r14b 
// CHECK: encoding: [0x41,0x0f,0x90,0xc6]         
seto %r14b 

// CHECK: seto (%rdx) 
// CHECK: encoding: [0x0f,0x90,0x02]         
seto (%rdx) 

// CHECK: setp 485498096 
// CHECK: encoding: [0x0f,0x9a,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
setp 485498096 

// CHECK: setp 64(%rdx) 
// CHECK: encoding: [0x0f,0x9a,0x42,0x40]         
setp 64(%rdx) 

// CHECK: setp 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9a,0x44,0x82,0x40]         
setp 64(%rdx,%rax,4) 

// CHECK: setp -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x9a,0x44,0x82,0xc0]         
setp -64(%rdx,%rax,4) 

// CHECK: setp 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x9a,0x44,0x02,0x40]         
setp 64(%rdx,%rax) 

// CHECK: setp %r14b 
// CHECK: encoding: [0x41,0x0f,0x9a,0xc6]         
setp %r14b 

// CHECK: setp (%rdx) 
// CHECK: encoding: [0x0f,0x9a,0x02]         
setp (%rdx) 

// CHECK: sets 485498096 
// CHECK: encoding: [0x0f,0x98,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
sets 485498096 

// CHECK: sets 64(%rdx) 
// CHECK: encoding: [0x0f,0x98,0x42,0x40]         
sets 64(%rdx) 

// CHECK: sets 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x98,0x44,0x82,0x40]         
sets 64(%rdx,%rax,4) 

// CHECK: sets -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x98,0x44,0x82,0xc0]         
sets -64(%rdx,%rax,4) 

// CHECK: sets 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x98,0x44,0x02,0x40]         
sets 64(%rdx,%rax) 

// CHECK: sets %r14b 
// CHECK: encoding: [0x41,0x0f,0x98,0xc6]         
sets %r14b 

// CHECK: sets (%rdx) 
// CHECK: encoding: [0x0f,0x98,0x02]         
sets (%rdx) 

// CHECK: shldl $0, %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xa4,0xed,0x00]       
shldl $0, %r13d, %r13d 

// CHECK: shldl %cl, %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xa5,0xed]       
shldl %cl, %r13d, %r13d 

// CHECK: shrdl $0, %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xac,0xed,0x00]       
shrdl $0, %r13d, %r13d 

// CHECK: shrdl %cl, %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xad,0xed]       
shrdl %cl, %r13d, %r13d 

// CHECK: stosl %eax, %es:(%rdi) 
// CHECK: encoding: [0xab]        
stosl %eax, %es:(%rdi) 

// CHECK: tzcntl %r13d, %r13d 
// CHECK: encoding: [0xf3,0x45,0x0f,0xbc,0xed]        
tzcntl %r13d, %r13d 

