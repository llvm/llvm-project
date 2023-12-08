// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clflushopt -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x0f,0xae,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
clflushopt -485498096(%edx,%eax,4) 

// CHECK: clflushopt 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x0f,0xae,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
clflushopt 485498096(%edx,%eax,4) 

// CHECK: clflushopt 485498096(%edx) 
// CHECK: encoding: [0x66,0x0f,0xae,0xba,0xf0,0x1c,0xf0,0x1c]         
clflushopt 485498096(%edx) 

// CHECK: clflushopt 485498096 
// CHECK: encoding: [0x66,0x0f,0xae,0x3d,0xf0,0x1c,0xf0,0x1c]         
clflushopt 485498096 

// CHECK: clflushopt 64(%edx,%eax) 
// CHECK: encoding: [0x66,0x0f,0xae,0x7c,0x02,0x40]         
clflushopt 64(%edx,%eax) 

// CHECK: clflushopt (%edx) 
// CHECK: encoding: [0x66,0x0f,0xae,0x3a]         
clflushopt (%edx) 

