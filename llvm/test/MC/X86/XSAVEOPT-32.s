// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xsaveopt -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0xb4,0x82,0x10,0xe3,0x0f,0xe3]         
xsaveopt -485498096(%edx,%eax,4) 

// CHECK: xsaveopt 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x0f,0xae,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]         
xsaveopt 485498096(%edx,%eax,4) 

// CHECK: xsaveopt 485498096(%edx) 
// CHECK: encoding: [0x0f,0xae,0xb2,0xf0,0x1c,0xf0,0x1c]         
xsaveopt 485498096(%edx) 

// CHECK: xsaveopt 485498096 
// CHECK: encoding: [0x0f,0xae,0x35,0xf0,0x1c,0xf0,0x1c]         
xsaveopt 485498096 

// CHECK: xsaveopt 64(%edx,%eax) 
// CHECK: encoding: [0x0f,0xae,0x74,0x02,0x40]         
xsaveopt 64(%edx,%eax) 

// CHECK: xsaveopt (%edx) 
// CHECK: encoding: [0x0f,0xae,0x32]         
xsaveopt (%edx) 

