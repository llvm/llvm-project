// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: bswapl %r13d 
// CHECK: encoding: [0x41,0x0f,0xcd]         
bswapl %r13d 

// CHECK: cmpxchgb %r14b, 485498096 
// CHECK: encoding: [0x44,0x0f,0xb0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
cmpxchgb %r14b, 485498096 

// CHECK: cmpxchgb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x0f,0xb0,0x72,0x40]        
cmpxchgb %r14b, 64(%rdx) 

// CHECK: cmpxchgb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x0f,0xb0,0x74,0x82,0x40]        
cmpxchgb %r14b, 64(%rdx,%rax,4) 

// CHECK: cmpxchgb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x0f,0xb0,0x74,0x82,0xc0]        
cmpxchgb %r14b, -64(%rdx,%rax,4) 

// CHECK: cmpxchgb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x0f,0xb0,0x74,0x02,0x40]        
cmpxchgb %r14b, 64(%rdx,%rax) 

// CHECK: cmpxchgb %r14b, %r14b 
// CHECK: encoding: [0x45,0x0f,0xb0,0xf6]        
cmpxchgb %r14b, %r14b 

// CHECK: cmpxchgb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x0f,0xb0,0x32]        
cmpxchgb %r14b, (%rdx) 

// CHECK: cmpxchgl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xb1,0xed]        
cmpxchgl %r13d, %r13d 

// CHECK: cmpxchgw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x0f,0xb1,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
cmpxchgw %r14w, 485498096 

// CHECK: cmpxchgw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x0f,0xb1,0x72,0x40]        
cmpxchgw %r14w, 64(%rdx) 

// CHECK: cmpxchgw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x0f,0xb1,0x74,0x82,0x40]        
cmpxchgw %r14w, 64(%rdx,%rax,4) 

// CHECK: cmpxchgw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x0f,0xb1,0x74,0x82,0xc0]        
cmpxchgw %r14w, -64(%rdx,%rax,4) 

// CHECK: cmpxchgw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x0f,0xb1,0x74,0x02,0x40]        
cmpxchgw %r14w, 64(%rdx,%rax) 

// CHECK: cmpxchgw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x0f,0xb1,0xf6]        
cmpxchgw %r14w, %r14w 

// CHECK: cmpxchgw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x0f,0xb1,0x32]        
cmpxchgw %r14w, (%rdx) 

// CHECK: cpuid 
// CHECK: encoding: [0x0f,0xa2]          
cpuid 

// CHECK: invd 
// CHECK: encoding: [0x0f,0x08]          
invd 

// CHECK: invlpg 485498096 
// CHECK: encoding: [0x0f,0x01,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
invlpg 485498096 

// CHECK: invlpg 64(%rdx) 
// CHECK: encoding: [0x0f,0x01,0x7a,0x40]         
invlpg 64(%rdx) 

// CHECK: invlpg 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x7c,0x82,0x40]         
invlpg 64(%rdx,%rax,4) 

// CHECK: invlpg -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x01,0x7c,0x82,0xc0]         
invlpg -64(%rdx,%rax,4) 

// CHECK: invlpg 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x01,0x7c,0x02,0x40]         
invlpg 64(%rdx,%rax) 

// CHECK: invlpg (%rdx) 
// CHECK: encoding: [0x0f,0x01,0x3a]         
invlpg (%rdx) 

// CHECK: rsm 
// CHECK: encoding: [0x0f,0xaa]          
rsm 

// CHECK: wbinvd 
// CHECK: encoding: [0x0f,0x09]          
wbinvd 

// CHECK: xaddb %r14b, 485498096 
// CHECK: encoding: [0x44,0x0f,0xc0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xaddb %r14b, 485498096 

// CHECK: xaddb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x0f,0xc0,0x72,0x40]        
xaddb %r14b, 64(%rdx) 

// CHECK: xaddb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x0f,0xc0,0x74,0x82,0x40]        
xaddb %r14b, 64(%rdx,%rax,4) 

// CHECK: xaddb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x0f,0xc0,0x74,0x82,0xc0]        
xaddb %r14b, -64(%rdx,%rax,4) 

// CHECK: xaddb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x0f,0xc0,0x74,0x02,0x40]        
xaddb %r14b, 64(%rdx,%rax) 

// CHECK: xaddb %r14b, %r14b 
// CHECK: encoding: [0x45,0x0f,0xc0,0xf6]        
xaddb %r14b, %r14b 

// CHECK: xaddb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x0f,0xc0,0x32]        
xaddb %r14b, (%rdx) 

// CHECK: xaddl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xc1,0xed]        
xaddl %r13d, %r13d 

// CHECK: xaddw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x0f,0xc1,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xaddw %r14w, 485498096 

// CHECK: xaddw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x0f,0xc1,0x72,0x40]        
xaddw %r14w, 64(%rdx) 

// CHECK: xaddw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x0f,0xc1,0x74,0x82,0x40]        
xaddw %r14w, 64(%rdx,%rax,4) 

// CHECK: xaddw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x0f,0xc1,0x74,0x82,0xc0]        
xaddw %r14w, -64(%rdx,%rax,4) 

// CHECK: xaddw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x0f,0xc1,0x74,0x02,0x40]        
xaddw %r14w, 64(%rdx,%rax) 

// CHECK: xaddw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x0f,0xc1,0xf6]        
xaddw %r14w, %r14w 

// CHECK: xaddw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x0f,0xc1,0x32]        
xaddw %r14w, (%rdx) 

