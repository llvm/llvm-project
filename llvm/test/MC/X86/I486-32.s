// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: bswapl %eax 
// CHECK: encoding: [0x0f,0xc8]         
bswapl %eax 

// CHECK: cmpxchgl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xb1,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
cmpxchgl %eax, 3809469200(%edx,%eax,4) 

// CHECK: cmpxchgl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xb1,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
cmpxchgl %eax, 485498096(%edx,%eax,4) 

// CHECK: cmpxchgl %eax, 485498096(%edx) 
// CHECK: encoding: [0x0f,0xb1,0x82,0xf0,0x1c,0xf0,0x1c]        
cmpxchgl %eax, 485498096(%edx) 

// CHECK: cmpxchgl %eax, 485498096 
// CHECK: encoding: [0x0f,0xb1,0x05,0xf0,0x1c,0xf0,0x1c]        
cmpxchgl %eax, 485498096 

// CHECK: cmpxchgl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xb1,0x44,0x02,0x40]        
cmpxchgl %eax, 64(%edx,%eax) 

// CHECK: cmpxchgl %eax, %eax 
// CHECK: encoding: [0x0f,0xb1,0xc0]        
cmpxchgl %eax, %eax 

// CHECK: cmpxchgl %eax, (%edx) 
// CHECK: encoding: [0x0f,0xb1,0x02]        
cmpxchgl %eax, (%edx) 

// CHECK: cpuid 
// CHECK: encoding: [0x0f,0xa2]          
cpuid 

// CHECK: invd 
// CHECK: encoding: [0x0f,0x08]          
invd 

// CHECK: invlpg -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x01,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
invlpg -485498096(%edx,%eax,4) 

// CHECK: invlpg 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0x01,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
invlpg 485498096(%edx,%eax,4) 

// CHECK: invlpg 485498096(%edx) 
// CHECK: encoding: [0x0f,0x01,0xba,0xf0,0x1c,0xf0,0x1c]         
invlpg 485498096(%edx) 

// CHECK: invlpg 485498096 
// CHECK: encoding: [0x0f,0x01,0x3d,0xf0,0x1c,0xf0,0x1c]         
invlpg 485498096 

// CHECK: invlpg 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0x01,0x7c,0x02,0x40]         
invlpg 64(%edx,%eax) 

// CHECK: invlpg (%edx) 
// CHECK: encoding: [0x0f,0x01,0x3a]         
invlpg (%edx) 

// CHECK: rsm 
// CHECK: encoding: [0x0f,0xaa]          
rsm 

// CHECK: wbinvd 
// CHECK: encoding: [0x0f,0x09]          
wbinvd 

// CHECK: xaddl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc1,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
xaddl %eax, 3809469200(%edx,%eax,4) 

// CHECK: xaddl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc1,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
xaddl %eax, 485498096(%edx,%eax,4) 

// CHECK: xaddl %eax, 485498096(%edx) 
// CHECK: encoding: [0x0f,0xc1,0x82,0xf0,0x1c,0xf0,0x1c]        
xaddl %eax, 485498096(%edx) 

// CHECK: xaddl %eax, 485498096 
// CHECK: encoding: [0x0f,0xc1,0x05,0xf0,0x1c,0xf0,0x1c]        
xaddl %eax, 485498096 

// CHECK: xaddl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xc1,0x44,0x02,0x40]        
xaddl %eax, 64(%edx,%eax) 

// CHECK: xaddl %eax, %eax 
// CHECK: encoding: [0x0f,0xc1,0xc0]        
xaddl %eax, %eax 

// CHECK: xaddl %eax, (%edx) 
// CHECK: encoding: [0x0f,0xc1,0x02]        
xaddl %eax, (%edx) 

