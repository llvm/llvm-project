// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xgetbv 
// CHECK: encoding: [0x0f,0x01,0xd0]          
xgetbv 

// CHECK: xrstor -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
xrstor -485498096(%edx,%eax,4) 

// CHECK: xrstor 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
xrstor 485498096(%edx,%eax,4) 

// CHECK: xrstor 485498096(%edx) 
// CHECK: encoding: [0x0f,0xae,0xaa,0xf0,0x1c,0xf0,0x1c]         
xrstor 485498096(%edx) 

// CHECK: xrstor 485498096 
// CHECK: encoding: [0x0f,0xae,0x2d,0xf0,0x1c,0xf0,0x1c]         
xrstor 485498096 

// CHECK: xrstor 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xae,0x6c,0x02,0x40]         
xrstor 64(%edx,%eax) 

// CHECK: xrstor (%edx) 
// CHECK: encoding: [0x0f,0xae,0x2a]         
xrstor (%edx) 

// CHECK: xsave -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
xsave -485498096(%edx,%eax,4) 

// CHECK: xsave 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
xsave 485498096(%edx,%eax,4) 

// CHECK: xsave 485498096(%edx) 
// CHECK: encoding: [0x0f,0xae,0xa2,0xf0,0x1c,0xf0,0x1c]         
xsave 485498096(%edx) 

// CHECK: xsave 485498096 
// CHECK: encoding: [0x0f,0xae,0x25,0xf0,0x1c,0xf0,0x1c]         
xsave 485498096 

// CHECK: xsave 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xae,0x64,0x02,0x40]         
xsave 64(%edx,%eax) 

// CHECK: xsave (%edx) 
// CHECK: encoding: [0x0f,0xae,0x22]         
xsave (%edx) 

// CHECK: xsetbv 
// CHECK: encoding: [0x0f,0x01,0xd1]          
xsetbv 

