// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xgetbv 
// CHECK: encoding: [0x0f,0x01,0xd0]          
xgetbv 

// CHECK: xrstor 485498096 
// CHECK: encoding: [0x0f,0xae,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
xrstor 485498096 

// CHECK: xrstor64 485498096 
// CHECK: encoding: [0x48,0x0f,0xae,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
xrstor64 485498096 

// CHECK: xrstor64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x6a,0x40]         
xrstor64 64(%rdx) 

// CHECK: xrstor64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x6c,0x82,0x40]         
xrstor64 64(%rdx,%rax,4) 

// CHECK: xrstor64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x6c,0x82,0xc0]         
xrstor64 -64(%rdx,%rax,4) 

// CHECK: xrstor64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xae,0x6c,0x02,0x40]         
xrstor64 64(%rdx,%rax) 

// CHECK: xrstor 64(%rdx) 
// CHECK: encoding: [0x0f,0xae,0x6a,0x40]         
xrstor 64(%rdx) 

// CHECK: xrstor64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x2a]         
xrstor64 (%rdx) 

// CHECK: xrstor 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x6c,0x82,0x40]         
xrstor 64(%rdx,%rax,4) 

// CHECK: xrstor -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x6c,0x82,0xc0]         
xrstor -64(%rdx,%rax,4) 

// CHECK: xrstor 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xae,0x6c,0x02,0x40]         
xrstor 64(%rdx,%rax) 

// CHECK: xrstor (%rdx) 
// CHECK: encoding: [0x0f,0xae,0x2a]         
xrstor (%rdx) 

// CHECK: xsave 485498096 
// CHECK: encoding: [0x0f,0xae,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
xsave 485498096 

// CHECK: xsave64 485498096 
// CHECK: encoding: [0x48,0x0f,0xae,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
xsave64 485498096 

// CHECK: xsave64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x62,0x40]         
xsave64 64(%rdx) 

// CHECK: xsave64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x64,0x82,0x40]         
xsave64 64(%rdx,%rax,4) 

// CHECK: xsave64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xae,0x64,0x82,0xc0]         
xsave64 -64(%rdx,%rax,4) 

// CHECK: xsave64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xae,0x64,0x02,0x40]         
xsave64 64(%rdx,%rax) 

// CHECK: xsave 64(%rdx) 
// CHECK: encoding: [0x0f,0xae,0x62,0x40]         
xsave 64(%rdx) 

// CHECK: xsave64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xae,0x22]         
xsave64 (%rdx) 

// CHECK: xsave 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x64,0x82,0x40]         
xsave 64(%rdx,%rax,4) 

// CHECK: xsave -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xae,0x64,0x82,0xc0]         
xsave -64(%rdx,%rax,4) 

// CHECK: xsave 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xae,0x64,0x02,0x40]         
xsave 64(%rdx,%rax) 

// CHECK: xsave (%rdx) 
// CHECK: encoding: [0x0f,0xae,0x22]         
xsave (%rdx) 

// CHECK: xsetbv 
// CHECK: encoding: [0x0f,0x01,0xd1]          
xsetbv 

