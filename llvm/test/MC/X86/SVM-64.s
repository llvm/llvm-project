// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clgi 
// CHECK: encoding: [0x0f,0x01,0xdd]          
clgi 

// CHECK: invlpga %rax, %ecx 
// CHECK: encoding: [0x0f,0x01,0xdf]        
invlpga %rax, %ecx 

// CHECK: skinit %eax 
// CHECK: encoding: [0x0f,0x01,0xde]         
skinit %eax 

// CHECK: stgi 
// CHECK: encoding: [0x0f,0x01,0xdc]          
stgi 

// CHECK: vmload %rax 
// CHECK: encoding: [0x0f,0x01,0xda]         
vmload %rax 

// CHECK: vmmcall 
// CHECK: encoding: [0x0f,0x01,0xd9]          
vmmcall 

// CHECK: vmrun %rax 
// CHECK: encoding: [0x0f,0x01,0xd8]         
vmrun %rax 

// CHECK: vmsave %rax 
// CHECK: encoding: [0x0f,0x01,0xdb]         
vmsave %rax 

