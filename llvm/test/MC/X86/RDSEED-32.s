// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdseedl %eax 
// CHECK: encoding: [0x0f,0xc7,0xf8]         
rdseedl %eax 

