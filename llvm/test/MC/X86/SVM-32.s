// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clgi 
// CHECK: encoding: [0x0f,0x01,0xdd]          
clgi 

// CHECK: invlpga %eax, %ecx 
// CHECK: encoding: [0x0f,0x01,0xdf]        
invlpga %eax, %ecx 

// CHECK: skinit %eax 
// CHECK: encoding: [0x0f,0x01,0xde]         
skinit %eax 

// CHECK: stgi 
// CHECK: encoding: [0x0f,0x01,0xdc]          
stgi 

// CHECK: vmload %eax 
// CHECK: encoding: [0x0f,0x01,0xda]         
vmload %eax 

// CHECK: vmmcall 
// CHECK: encoding: [0x0f,0x01,0xd9]          
vmmcall 

// CHECK: vmrun %eax 
// CHECK: encoding: [0x0f,0x01,0xd8]         
vmrun %eax 

// CHECK: vmsave %eax 
// CHECK: encoding: [0x0f,0x01,0xdb]         
vmsave %eax 

