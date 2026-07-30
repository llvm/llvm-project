// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdrandl %r13d 
// CHECK: encoding: [0x41,0x0f,0xc7,0xf5]         
rdrandl %r13d 

// CHECK: rdrandq %r13 
// CHECK: encoding: [0x49,0x0f,0xc7,0xf5]         
rdrandq %r13 

// CHECK: rdrandw %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xc7,0xf5]         
rdrandw %r13w 

