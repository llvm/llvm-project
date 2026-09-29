// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: fxrstor -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0x8c,0x82,0x10,0xe3,0x0f,0xe3]         
fxrstor -485498096(%edx,%eax,4) 

// CHECK: fxrstor 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]         
fxrstor 485498096(%edx,%eax,4) 

// CHECK: fxrstor 485498096(%edx) 
// CHECK: encoding: [0x0f,0xae,0x8a,0xf0,0x1c,0xf0,0x1c]         
fxrstor 485498096(%edx) 

// CHECK: fxrstor 485498096 
// CHECK: encoding: [0x0f,0xae,0x0d,0xf0,0x1c,0xf0,0x1c]         
fxrstor 485498096 

// CHECK: fxrstor 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xae,0x4c,0x02,0x40]         
fxrstor 64(%edx,%eax) 

// CHECK: fxrstor (%edx) 
// CHECK: encoding: [0x0f,0xae,0x0a]         
fxrstor (%edx) 

// CHECK: fxsave -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
fxsave -485498096(%edx,%eax,4) 

// CHECK: fxsave 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
fxsave 485498096(%edx,%eax,4) 

// CHECK: fxsave 485498096(%edx) 
// CHECK: encoding: [0x0f,0xae,0x82,0xf0,0x1c,0xf0,0x1c]         
fxsave 485498096(%edx) 

// CHECK: fxsave 485498096 
// CHECK: encoding: [0x0f,0xae,0x05,0xf0,0x1c,0xf0,0x1c]         
fxsave 485498096 

// CHECK: fxsave 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xae,0x44,0x02,0x40]         
fxsave 64(%edx,%eax) 

// CHECK: fxsave (%edx) 
// CHECK: encoding: [0x0f,0xae,0x02]         
fxsave (%edx) 

