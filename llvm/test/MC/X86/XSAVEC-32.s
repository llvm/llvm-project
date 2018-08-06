// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xsavec -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
xsavec -485498096(%edx,%eax,4) 

// CHECK: xsavec 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xc7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
xsavec 485498096(%edx,%eax,4) 

// CHECK: xsavec 485498096(%edx) 
// CHECK: encoding: [0x0f,0xc7,0xa2,0xf0,0x1c,0xf0,0x1c]         
xsavec 485498096(%edx) 

// CHECK: xsavec 485498096 
// CHECK: encoding: [0x0f,0xc7,0x25,0xf0,0x1c,0xf0,0x1c]         
xsavec 485498096 

// CHECK: xsavec 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xc7,0x64,0x02,0x40]         
xsavec 64(%edx,%eax) 

// CHECK: xsavec (%edx) 
// CHECK: encoding: [0x0f,0xc7,0x22]         
xsavec (%edx) 

