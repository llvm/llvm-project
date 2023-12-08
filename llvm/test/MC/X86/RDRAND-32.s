// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdrandl %eax 
// CHECK: encoding: [0x0f,0xc7,0xf0]         
rdrandl %eax 

