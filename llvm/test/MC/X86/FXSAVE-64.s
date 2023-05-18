// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: fxrstor 485498096 
// CHECK: encoding: [0x0f,0xae,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
fxrstor 485498096 

// CHECK: fxrstor 64(%rdx) 
// CHECK: encoding: [0x0f,0xae,0x4a,0x40]         
fxrstor 64(%rdx) 

// CHECK: fxrstor 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x4c,0x82,0x40]         
fxrstor 64(%rdx,%rax,4) 

// CHECK: fxrstor -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x4c,0x82,0xc0]         
fxrstor -64(%rdx,%rax,4) 

// CHECK: fxrstor 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xae,0x4c,0x02,0x40]         
fxrstor 64(%rdx,%rax) 

// CHECK: fxrstor (%rdx) 
// CHECK: encoding: [0x0f,0xae,0x0a]         
fxrstor (%rdx) 

// CHECK: fxsave 485498096 
// CHECK: encoding: [0x0f,0xae,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
fxsave 485498096 

// CHECK: fxsave 64(%rdx) 
// CHECK: encoding: [0x0f,0xae,0x42,0x40]         
fxsave 64(%rdx) 

// CHECK: fxsave 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x44,0x82,0x40]         
fxsave 64(%rdx,%rax,4) 

// CHECK: fxsave -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x44,0x82,0xc0]         
fxsave -64(%rdx,%rax,4) 

// CHECK: fxsave 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xae,0x44,0x02,0x40]         
fxsave 64(%rdx,%rax) 

// CHECK: fxsave (%rdx) 
// CHECK: encoding: [0x0f,0xae,0x02]         
fxsave (%rdx) 

