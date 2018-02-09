// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: encls 
// CHECK: encoding: [0x0f,0x01,0xcf]          
encls 

// CHECK: enclu 
// CHECK: encoding: [0x0f,0x01,0xd7]          
enclu 

