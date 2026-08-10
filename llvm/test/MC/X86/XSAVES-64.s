// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xrstors 485498096 
// CHECK: encoding: [0x0f,0xc7,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
xrstors 485498096 

// CHECK: xrstors64 485498096 
// CHECK: encoding: [0x48,0x0f,0xc7,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
xrstors64 485498096 

// CHECK: xrstors64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x5a,0x40]         
xrstors64 64(%rdx) 

// CHECK: xrstors64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x5c,0x82,0x40]         
xrstors64 64(%rdx,%rax,4) 

// CHECK: xrstors64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x5c,0x82,0xc0]         
xrstors64 -64(%rdx,%rax,4) 

// CHECK: xrstors64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x5c,0x02,0x40]         
xrstors64 64(%rdx,%rax) 

// CHECK: xrstors 64(%rdx) 
// CHECK: encoding: [0x0f,0xc7,0x5a,0x40]         
xrstors 64(%rdx) 

// CHECK: xrstors64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x1a]         
xrstors64 (%rdx) 

// CHECK: xrstors 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xc7,0x5c,0x82,0x40]         
xrstors 64(%rdx,%rax,4) 

// CHECK: xrstors -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xc7,0x5c,0x82,0xc0]         
xrstors -64(%rdx,%rax,4) 

// CHECK: xrstors 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xc7,0x5c,0x02,0x40]         
xrstors 64(%rdx,%rax) 

// CHECK: xrstors (%rdx) 
// CHECK: encoding: [0x0f,0xc7,0x1a]         
xrstors (%rdx) 

// CHECK: xsaves 485498096 
// CHECK: encoding: [0x0f,0xc7,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
xsaves 485498096 

// CHECK: xsaves64 485498096 
// CHECK: encoding: [0x48,0x0f,0xc7,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
xsaves64 485498096 

// CHECK: xsaves64 64(%rdx) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x6a,0x40]         
xsaves64 64(%rdx) 

// CHECK: xsaves64 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x6c,0x82,0x40]         
xsaves64 64(%rdx,%rax,4) 

// CHECK: xsaves64 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x6c,0x82,0xc0]         
xsaves64 -64(%rdx,%rax,4) 

// CHECK: xsaves64 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x6c,0x02,0x40]         
xsaves64 64(%rdx,%rax) 

// CHECK: xsaves 64(%rdx) 
// CHECK: encoding: [0x0f,0xc7,0x6a,0x40]         
xsaves 64(%rdx) 

// CHECK: xsaves64 (%rdx) 
// CHECK: encoding: [0x48,0x0f,0xc7,0x2a]         
xsaves64 (%rdx) 

// CHECK: xsaves 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xc7,0x6c,0x82,0x40]         
xsaves 64(%rdx,%rax,4) 

// CHECK: xsaves -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0xc7,0x6c,0x82,0xc0]         
xsaves -64(%rdx,%rax,4) 

// CHECK: xsaves 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0xc7,0x6c,0x02,0x40]         
xsaves 64(%rdx,%rax) 

// CHECK: xsaves (%rdx) 
// CHECK: encoding: [0x0f,0xc7,0x2a]         
xsaves (%rdx) 

