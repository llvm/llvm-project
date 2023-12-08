// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vmfunc 
// CHECK: encoding: [0x0f,0x01,0xd4]          
vmfunc 

