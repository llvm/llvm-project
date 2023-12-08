// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clzero 
// CHECK: encoding: [0x0f,0x01,0xfc]          
clzero 

