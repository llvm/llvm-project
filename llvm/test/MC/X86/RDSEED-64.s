// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdseedl %r13d 
// CHECK: encoding: [0x41,0x0f,0xc7,0xfd]         
rdseedl %r13d 

// CHECK: rdseedq %r13 
// CHECK: encoding: [0x49,0x0f,0xc7,0xfd]         
rdseedq %r13 

// CHECK: rdseedw %r13w 
// CHECK: encoding: [0x66,0x41,0x0f,0xc7,0xfd]         
rdseedw %r13w 

