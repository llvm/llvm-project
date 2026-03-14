// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xsavec 485498096 
// CHECK: encoding: [0x0f,0xc7,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
xsavec 485498096 

// CHECK: xsavec64 485498096 
// CHECK: encoding: [0x48,0x0f,0xc7,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
xsavec64 485498096 

// CHECK: xsavec64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x62,0x40]         
xsavec64 64(%rdx) 

// CHECK: xsavec64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x64,0x82,0x40]         
xsavec64 64(%rdx,%rax,4) 

// CHECK: xsavec64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x64,0x82,0xc0]         
xsavec64 -64(%rdx,%rax,4) 

// CHECK: xsavec64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x64,0x02,0x40]         
xsavec64 64(%rdx,%rax) 

// CHECK: xsavec 64(%rdx) 
// CHECK: encoding: [0x0f,0xc7,0x62,0x40]         
xsavec 64(%rdx) 

// CHECK: xsavec64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x22]         
xsavec64 (%rdx) 

// CHECK: xsavec 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xc7,0x64,0x82,0x40]         
xsavec 64(%rdx,%rax,4) 

// CHECK: xsavec -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xc7,0x64,0x82,0xc0]         
xsavec -64(%rdx,%rax,4) 

// CHECK: xsavec 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xc7,0x64,0x02,0x40]         
xsavec 64(%rdx,%rax) 

// CHECK: xsavec (%rdx) 
// CHECK: encoding: [0x0f,0xc7,0x22]         
xsavec (%rdx) 

