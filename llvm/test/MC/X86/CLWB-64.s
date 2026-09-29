// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clwb 485498096 
// CHECK: encoding: [0x66,0x0f,0xae,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
clwb 485498096 

// CHECK: clwb 64(%rdx) 
// CHECK: encoding: [0x66,0x0f,0xae,0x72,0x40]         
clwb 64(%rdx) 

// CHECK: clwb 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x0f,0xae,0x74,0x82,0x40]         
clwb 64(%rdx,%rax,4) 

// CHECK: clwb -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x0f,0xae,0x74,0x82,0xc0]         
clwb -64(%rdx,%rax,4) 

// CHECK: clwb 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x0f,0xae,0x74,0x02,0x40]         
clwb 64(%rdx,%rax) 

// CHECK: clwb (%rdx) 
// CHECK: encoding: [0x66,0x0f,0xae,0x32]         
clwb (%rdx) 

