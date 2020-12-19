// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clgi 
// CHECK: encoding: [0x0f,0x01,0xdd]          
clgi 

// CHECK: invlpga
// CHECK: encoding: [0x0f,0x01,0xdf]        
invlpga %rax, %ecx 

// CHECK: invlpga
// CHECK: encoding: [0x0f,0x01,0xdf]        
invlpga

// CHECK: skinit
// CHECK: encoding: [0x0f,0x01,0xde]         
skinit %eax 

// CHECK: skinit
// CHECK: encoding: [0x0f,0x01,0xde]         
skinit

// CHECK: stgi 
// CHECK: encoding: [0x0f,0x01,0xdc]          
stgi 

// CHECK: vmload
// CHECK: encoding: [0x0f,0x01,0xda]         
vmload %rax 

// CHECK: vmload
// CHECK: encoding: [0x0f,0x01,0xda]         
vmload

// CHECK: vmmcall 
// CHECK: encoding: [0x0f,0x01,0xd9]          
vmmcall 

// CHECK: vmrun
// CHECK: encoding: [0x0f,0x01,0xd8]         
vmrun %rax 

// CHECK: vmrun
// CHECK: encoding: [0x0f,0x01,0xd8]         
vmrun

// CHECK: vmsave
// CHECK: encoding: [0x0f,0x01,0xdb]         
vmsave %rax 

// CHECK: vmsave
// CHECK: encoding: [0x0f,0x01,0xdb]         
vmsave

