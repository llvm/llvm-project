// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdfsbasel %r13d 
// CHECK: encoding: [0xf3,0x41,0x0f,0xae,0xc5]         
rdfsbasel %r13d 

// CHECK: rdfsbaseq %r13 
// CHECK: encoding: [0xf3,0x49,0x0f,0xae,0xc5]         
rdfsbaseq %r13 

// CHECK: rdgsbasel %r13d 
// CHECK: encoding: [0xf3,0x41,0x0f,0xae,0xcd]         
rdgsbasel %r13d 

// CHECK: rdgsbaseq %r13 
// CHECK: encoding: [0xf3,0x49,0x0f,0xae,0xcd]         
rdgsbaseq %r13 

// CHECK: wrfsbasel %r13d 
// CHECK: encoding: [0xf3,0x41,0x0f,0xae,0xd5]         
wrfsbasel %r13d 

// CHECK: wrfsbaseq %r13 
// CHECK: encoding: [0xf3,0x49,0x0f,0xae,0xd5]         
wrfsbaseq %r13 

// CHECK: wrgsbasel %r13d 
// CHECK: encoding: [0xf3,0x41,0x0f,0xae,0xdd]         
wrgsbasel %r13d 

// CHECK: wrgsbaseq %r13 
// CHECK: encoding: [0xf3,0x49,0x0f,0xae,0xdd]         
wrgsbaseq %r13 

