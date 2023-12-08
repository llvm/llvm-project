// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: clgi 
// CHECK: encoding: [0x0f,0x01,0xdd]          
clgi 

// CHECK: invlpga
// CHECK: encoding: [0x0f,0x01,0xdf]        
invlpga %eax, %ecx 

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
vmload %eax 

// CHECK: vmload
// CHECK: encoding: [0x0f,0x01,0xda]         
vmload

// CHECK: vmmcall 
// CHECK: encoding: [0x0f,0x01,0xd9]          
vmmcall 

// CHECK: vmrun
// CHECK: encoding: [0x0f,0x01,0xd8]         
vmrun %eax 

// CHECK: vmrun
// CHECK: encoding: [0x0f,0x01,0xd8]         
vmrun

// CHECK: vmsave
// CHECK: encoding: [0x0f,0x01,0xdb]         
vmsave %eax 

// CHECK: vmsave
// CHECK: encoding: [0x0f,0x01,0xdb]         
vmsave

