// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: rdpkru 
// CHECK: encoding: [0x0f,0x01,0xee]          
rdpkru 

// CHECK: wrpkru 
// CHECK: encoding: [0x0f,0x01,0xef]          
wrpkru 

