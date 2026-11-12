// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdpmc 
// CHECK: encoding: [0x0f,0x33]          
rdpmc 

