// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clflushopt 485498096 
// CHECK: encoding: [0x66,0x0f,0xae,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
clflushopt 485498096 

// CHECK: clflushopt 64(%rdx) 
// CHECK: encoding: [0x66,0x0f,0xae,0x7a,0x40]         
clflushopt 64(%rdx) 

// CHECK: clflushopt 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x0f,0xae,0x7c,0x82,0x40]         
clflushopt 64(%rdx,%rax,4) 

// CHECK: clflushopt -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x0f,0xae,0x7c,0x82,0xc0]         
clflushopt -64(%rdx,%rax,4) 

// CHECK: clflushopt 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x0f,0xae,0x7c,0x02,0x40]         
clflushopt 64(%rdx,%rax) 

// CHECK: clflushopt (%rdx) 
// CHECK: encoding: [0x66,0x0f,0xae,0x3a]         
clflushopt (%rdx) 

