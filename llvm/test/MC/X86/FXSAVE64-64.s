// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: fxrstor64 485498096 
// CHECK: encoding: [0x48,0x0f,0xae,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
fxrstor64 485498096 

// CHECK: fxrstor64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x4a,0x40]         
fxrstor64 64(%rdx) 

// CHECK: fxrstor64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x4c,0x82,0x40]         
fxrstor64 64(%rdx,%rax,4) 

// CHECK: fxrstor64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x4c,0x82,0xc0]         
fxrstor64 -64(%rdx,%rax,4) 

// CHECK: fxrstor64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xae,0x4c,0x02,0x40]         
fxrstor64 64(%rdx,%rax) 

// CHECK: fxrstor64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x0a]         
fxrstor64 (%rdx) 

// CHECK: fxsave64 485498096 
// CHECK: encoding: [0x48,0x0f,0xae,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
fxsave64 485498096 

// CHECK: fxsave64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x42,0x40]         
fxsave64 64(%rdx) 

// CHECK: fxsave64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x44,0x82,0x40]         
fxsave64 64(%rdx,%rax,4) 

// CHECK: fxsave64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x44,0x82,0xc0]         
fxsave64 -64(%rdx,%rax,4) 

// CHECK: fxsave64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xae,0x44,0x02,0x40]         
fxsave64 64(%rdx,%rax) 

// CHECK: fxsave64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x02]         
fxsave64 (%rdx) 

