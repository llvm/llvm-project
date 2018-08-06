// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: xabort $0 
// CHECK: encoding: [0xc6,0xf8,0x00]         
xabort $0 

// CHECK: xbegin 64 
// CHECK: encoding: [0xc7,0xf8,A,A,A,A]         
xbegin 64 

// CHECK: xend 
// CHECK: encoding: [0x0f,0x01,0xd5]          
xend 

// CHECK: xtest 
// CHECK: encoding: [0x0f,0x01,0xd6]          
xtest 

