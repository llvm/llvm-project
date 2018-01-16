// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xsaveopt 485498096 
// CHECK: encoding: [0x0f,0xae,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
xsaveopt 485498096 

// CHECK: xsaveopt64 485498096 
// CHECK: encoding: [0x48,0x0f,0xae,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
xsaveopt64 485498096 

// CHECK: xsaveopt64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x72,0x40]         
xsaveopt64 64(%rdx) 

// CHECK: xsaveopt64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x74,0x82,0x40]         
xsaveopt64 64(%rdx,%rax,4) 

// CHECK: xsaveopt64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x74,0x82,0xc0]         
xsaveopt64 -64(%rdx,%rax,4) 

// CHECK: xsaveopt64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xae,0x74,0x02,0x40]         
xsaveopt64 64(%rdx,%rax) 

// CHECK: xsaveopt 64(%rdx) 
// CHECK: encoding: [0x0f,0xae,0x72,0x40]         
xsaveopt 64(%rdx) 

// CHECK: xsaveopt64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x32]         
xsaveopt64 (%rdx) 

// CHECK: xsaveopt 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x74,0x82,0x40]         
xsaveopt 64(%rdx,%rax,4) 

// CHECK: xsaveopt -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x74,0x82,0xc0]         
xsaveopt -64(%rdx,%rax,4) 

// CHECK: xsaveopt 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xae,0x74,0x02,0x40]         
xsaveopt 64(%rdx,%rax) 

// CHECK: xsaveopt (%rdx) 
// CHECK: encoding: [0x0f,0xae,0x32]         
xsaveopt (%rdx) 

