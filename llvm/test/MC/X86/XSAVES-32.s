// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xrstors -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc7,0x9c,0x82,0x10,0xe3,0x0f,0xe3]         
xrstors -485498096(%edx,%eax,4) 

// CHECK: xrstors 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc7,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
xrstors 485498096(%edx,%eax,4) 

// CHECK: xrstors 485498096(%edx) 
// CHECK: encoding: [0x0f,0xc7,0x9a,0xf0,0x1c,0xf0,0x1c]         
xrstors 485498096(%edx) 

// CHECK: xrstors 485498096 
// CHECK: encoding: [0x0f,0xc7,0x1d,0xf0,0x1c,0xf0,0x1c]         
xrstors 485498096 

// CHECK: xrstors 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xc7,0x5c,0x02,0x40]         
xrstors 64(%edx,%eax) 

// CHECK: xrstors (%edx) 
// CHECK: encoding: [0x0f,0xc7,0x1a]         
xrstors (%edx) 

// CHECK: xsaves -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc7,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
xsaves -485498096(%edx,%eax,4) 

// CHECK: xsaves 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc7,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
xsaves 485498096(%edx,%eax,4) 

// CHECK: xsaves 485498096(%edx) 
// CHECK: encoding: [0x0f,0xc7,0xaa,0xf0,0x1c,0xf0,0x1c]         
xsaves 485498096(%edx) 

// CHECK: xsaves 485498096 
// CHECK: encoding: [0x0f,0xc7,0x2d,0xf0,0x1c,0xf0,0x1c]         
xsaves 485498096 

// CHECK: xsaves 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xc7,0x6c,0x02,0x40]         
xsaves 64(%edx,%eax) 

// CHECK: xsaves (%edx) 
// CHECK: encoding: [0x0f,0xc7,0x2a]         
xsaves (%edx) 

