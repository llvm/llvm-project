// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: bound %eax, 3809469200(%edx,%eax,4)
// CHECK: encoding: [0x62,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
bound %eax, 3809469200(%edx,%eax,4)

// CHECK: bound %eax, 485498096
// CHECK: encoding: [0x62,0x05,0xf0,0x1c,0xf0,0x1c]        
bound %eax, 485498096

// CHECK: bound %eax, 485498096(%edx,%eax,4)
// CHECK: encoding: [0x62,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
bound %eax, 485498096(%edx,%eax,4)

// CHECK: bound %eax, 485498096(%edx)
// CHECK: encoding: [0x62,0x82,0xf0,0x1c,0xf0,0x1c]        
bound %eax, 485498096(%edx)

// CHECK: bound %eax, 64(%edx,%eax)
// CHECK: encoding: [0x62,0x44,0x02,0x40]        
bound %eax, 64(%edx,%eax)

// CHECK: bound %eax, (%edx)
// CHECK: encoding: [0x62,0x02]        
bound %eax, (%edx)

// CHECK: enter $0, $0 
// CHECK: encoding: [0xc8,0x00,0x00,0x00]        
enter $0, $0 

// CHECK: imull $0, %eax, %eax 
// CHECK: encoding: [0x6b,0xc0,0x00]       
imull $0, %eax, %eax 

// CHECK: insb %dx, %es:(%edi) 
// CHECK: encoding: [0x6c]        
insb %dx, %es:(%edi) 

// CHECK: insl %dx, %es:(%edi) 
// CHECK: encoding: [0x6d]        
insl %dx, %es:(%edi) 

// CHECK: insw %dx, %es:(%edi) 
// CHECK: encoding: [0x66,0x6d]        
insw %dx, %es:(%edi) 

// CHECK: leave 
// CHECK: encoding: [0xc9]          
leave 

// CHECK: outsb %es:(%esi), %dx 
// CHECK: encoding: [0x26,0x6e]        
outsb %es:(%esi), %dx 

// CHECK: outsl %es:(%esi), %dx 
// CHECK: encoding: [0x26,0x6f]        
outsl %es:(%esi), %dx 

// CHECK: outsw %es:(%esi), %dx 
// CHECK: encoding: [0x26,0x66,0x6f]        
outsw %es:(%esi), %dx 

// CHECK: popal 
// CHECK: encoding: [0x61]          
popal 

// CHECK: popaw 
// CHECK: encoding: [0x66,0x61]          
popaw 

// CHECK: pushal 
// CHECK: encoding: [0x60]          
pushal 

// CHECK: pushaw 
// CHECK: encoding: [0x66,0x60]          
pushaw 

// CHECK: pushl $0 
// CHECK: encoding: [0x6a,0x00]         
pushl $0 

// CHECK: pushw $0 
// CHECK: encoding: [0x66,0x6a,0x00]         
pushw $0 

// CHECK: rclb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rclb $0, -485498096(%edx,%eax,4) 

// CHECK: rclb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rclb $0, 485498096(%edx,%eax,4) 

// CHECK: rclb $0, 485498096(%edx) 
// CHECK: encoding: [0xc0,0x92,0xf0,0x1c,0xf0,0x1c,0x00]        
rclb $0, 485498096(%edx) 

// CHECK: rclb $0, 485498096 
// CHECK: encoding: [0xc0,0x15,0xf0,0x1c,0xf0,0x1c,0x00]        
rclb $0, 485498096 

// CHECK: rclb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc0,0x54,0x02,0x40,0x00]        
rclb $0, 64(%edx,%eax) 

// CHECK: rclb $0, (%edx) 
// CHECK: encoding: [0xc0,0x12,0x00]        
rclb $0, (%edx) 

// CHECK: rcll $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rcll $0, -485498096(%edx,%eax,4) 

// CHECK: rcll $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rcll $0, 485498096(%edx,%eax,4) 

// CHECK: rcll $0, 485498096(%edx) 
// CHECK: encoding: [0xc1,0x92,0xf0,0x1c,0xf0,0x1c,0x00]        
rcll $0, 485498096(%edx) 

// CHECK: rcll $0, 485498096 
// CHECK: encoding: [0xc1,0x15,0xf0,0x1c,0xf0,0x1c,0x00]        
rcll $0, 485498096 

// CHECK: rcll $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc1,0x54,0x02,0x40,0x00]        
rcll $0, 64(%edx,%eax) 

// CHECK: rcll $0, %eax 
// CHECK: encoding: [0xc1,0xd0,0x00]        
rcll $0, %eax 

// CHECK: rcll $0, (%edx) 
// CHECK: encoding: [0xc1,0x12,0x00]        
rcll $0, (%edx) 

// CHECK: rclw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rclw $0, -485498096(%edx,%eax,4) 

// CHECK: rclw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rclw $0, 485498096(%edx,%eax,4) 

// CHECK: rclw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc1,0x92,0xf0,0x1c,0xf0,0x1c,0x00]        
rclw $0, 485498096(%edx) 

// CHECK: rclw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x15,0xf0,0x1c,0xf0,0x1c,0x00]        
rclw $0, 485498096 

// CHECK: rclw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc1,0x54,0x02,0x40,0x00]        
rclw $0, 64(%edx,%eax) 

// CHECK: rclw $0, (%edx) 
// CHECK: encoding: [0x66,0xc1,0x12,0x00]        
rclw $0, (%edx) 

// CHECK: rcrb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x9c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rcrb $0, -485498096(%edx,%eax,4) 

// CHECK: rcrb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x9c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrb $0, 485498096(%edx,%eax,4) 

// CHECK: rcrb $0, 485498096(%edx) 
// CHECK: encoding: [0xc0,0x9a,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrb $0, 485498096(%edx) 

// CHECK: rcrb $0, 485498096 
// CHECK: encoding: [0xc0,0x1d,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrb $0, 485498096 

// CHECK: rcrb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc0,0x5c,0x02,0x40,0x00]        
rcrb $0, 64(%edx,%eax) 

// CHECK: rcrb $0, (%edx) 
// CHECK: encoding: [0xc0,0x1a,0x00]        
rcrb $0, (%edx) 

// CHECK: rcrl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x9c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rcrl $0, -485498096(%edx,%eax,4) 

// CHECK: rcrl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x9c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrl $0, 485498096(%edx,%eax,4) 

// CHECK: rcrl $0, 485498096(%edx) 
// CHECK: encoding: [0xc1,0x9a,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrl $0, 485498096(%edx) 

// CHECK: rcrl $0, 485498096 
// CHECK: encoding: [0xc1,0x1d,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrl $0, 485498096 

// CHECK: rcrl $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc1,0x5c,0x02,0x40,0x00]        
rcrl $0, 64(%edx,%eax) 

// CHECK: rcrl $0, %eax 
// CHECK: encoding: [0xc1,0xd8,0x00]        
rcrl $0, %eax 

// CHECK: rcrl $0, (%edx) 
// CHECK: encoding: [0xc1,0x1a,0x00]        
rcrl $0, (%edx) 

// CHECK: rcrw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x9c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rcrw $0, -485498096(%edx,%eax,4) 

// CHECK: rcrw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x9c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrw $0, 485498096(%edx,%eax,4) 

// CHECK: rcrw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc1,0x9a,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrw $0, 485498096(%edx) 

// CHECK: rcrw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x1d,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrw $0, 485498096 

// CHECK: rcrw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc1,0x5c,0x02,0x40,0x00]        
rcrw $0, 64(%edx,%eax) 

// CHECK: rcrw $0, (%edx) 
// CHECK: encoding: [0x66,0xc1,0x1a,0x00]        
rcrw $0, (%edx) 

// CHECK: rep insb %dx, %es:(%edi) 
// CHECK: encoding: [0xf3,0x6c]       
rep insb %dx, %es:(%edi) 

// CHECK: rep insl %dx, %es:(%edi) 
// CHECK: encoding: [0xf3,0x6d]       
rep insl %dx, %es:(%edi) 

// CHECK: rep insw %dx, %es:(%edi) 
// CHECK: encoding: [0xf3,0x66,0x6d]       
rep insw %dx, %es:(%edi) 

// CHECK: repne insb %dx, %es:(%edi) 
// CHECK: encoding: [0xf2,0x6c]       
repne insb %dx, %es:(%edi) 

// CHECK: repne insl %dx, %es:(%edi) 
// CHECK: encoding: [0xf2,0x6d]       
repne insl %dx, %es:(%edi) 

// CHECK: repne insw %dx, %es:(%edi) 
// CHECK: encoding: [0xf2,0x66,0x6d]       
repne insw %dx, %es:(%edi) 

// CHECK: repne outsb %es:(%esi), %dx 
// CHECK: encoding: [0xf2,0x26,0x6e]       
repne outsb %es:(%esi), %dx 

// CHECK: repne outsl %es:(%esi), %dx 
// CHECK: encoding: [0xf2,0x26,0x6f]       
repne outsl %es:(%esi), %dx 

// CHECK: repne outsw %es:(%esi), %dx 
// CHECK: encoding: [0xf2,0x26,0x66,0x6f]       
repne outsw %es:(%esi), %dx 

// CHECK: rep outsb %es:(%esi), %dx 
// CHECK: encoding: [0xf3,0x26,0x6e]       
rep outsb %es:(%esi), %dx 

// CHECK: rep outsl %es:(%esi), %dx 
// CHECK: encoding: [0xf3,0x26,0x6f]       
rep outsl %es:(%esi), %dx 

// CHECK: rep outsw %es:(%esi), %dx 
// CHECK: encoding: [0xf3,0x26,0x66,0x6f]       
rep outsw %es:(%esi), %dx 

// CHECK: rolb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rolb $0, -485498096(%edx,%eax,4) 

// CHECK: rolb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rolb $0, 485498096(%edx,%eax,4) 

// CHECK: rolb $0, 485498096(%edx) 
// CHECK: encoding: [0xc0,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rolb $0, 485498096(%edx) 

// CHECK: rolb $0, 485498096 
// CHECK: encoding: [0xc0,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
rolb $0, 485498096 

// CHECK: rolb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc0,0x44,0x02,0x40,0x00]        
rolb $0, 64(%edx,%eax) 

// CHECK: rolb $0, (%edx) 
// CHECK: encoding: [0xc0,0x02,0x00]        
rolb $0, (%edx) 

// CHECK: roll $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
roll $0, -485498096(%edx,%eax,4) 

// CHECK: roll $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
roll $0, 485498096(%edx,%eax,4) 

// CHECK: roll $0, 485498096(%edx) 
// CHECK: encoding: [0xc1,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
roll $0, 485498096(%edx) 

// CHECK: roll $0, 485498096 
// CHECK: encoding: [0xc1,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
roll $0, 485498096 

// CHECK: roll $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc1,0x44,0x02,0x40,0x00]        
roll $0, 64(%edx,%eax) 

// CHECK: roll $0, %eax 
// CHECK: encoding: [0xc1,0xc0,0x00]        
roll $0, %eax 

// CHECK: roll $0, (%edx) 
// CHECK: encoding: [0xc1,0x02,0x00]        
roll $0, (%edx) 

// CHECK: rolw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rolw $0, -485498096(%edx,%eax,4) 

// CHECK: rolw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rolw $0, 485498096(%edx,%eax,4) 

// CHECK: rolw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc1,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rolw $0, 485498096(%edx) 

// CHECK: rolw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
rolw $0, 485498096 

// CHECK: rolw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc1,0x44,0x02,0x40,0x00]        
rolw $0, 64(%edx,%eax) 

// CHECK: rolw $0, (%edx) 
// CHECK: encoding: [0x66,0xc1,0x02,0x00]        
rolw $0, (%edx) 

// CHECK: rorb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rorb $0, -485498096(%edx,%eax,4) 

// CHECK: rorb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rorb $0, 485498096(%edx,%eax,4) 

// CHECK: rorb $0, 485498096(%edx) 
// CHECK: encoding: [0xc0,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]        
rorb $0, 485498096(%edx) 

// CHECK: rorb $0, 485498096 
// CHECK: encoding: [0xc0,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]        
rorb $0, 485498096 

// CHECK: rorb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc0,0x4c,0x02,0x40,0x00]        
rorb $0, 64(%edx,%eax) 

// CHECK: rorb $0, (%edx) 
// CHECK: encoding: [0xc0,0x0a,0x00]        
rorb $0, (%edx) 

// CHECK: rorl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rorl $0, -485498096(%edx,%eax,4) 

// CHECK: rorl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rorl $0, 485498096(%edx,%eax,4) 

// CHECK: rorl $0, 485498096(%edx) 
// CHECK: encoding: [0xc1,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]        
rorl $0, 485498096(%edx) 

// CHECK: rorl $0, 485498096 
// CHECK: encoding: [0xc1,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]        
rorl $0, 485498096 

// CHECK: rorl $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc1,0x4c,0x02,0x40,0x00]        
rorl $0, 64(%edx,%eax) 

// CHECK: rorl $0, %eax 
// CHECK: encoding: [0xc1,0xc8,0x00]        
rorl $0, %eax 

// CHECK: rorl $0, (%edx) 
// CHECK: encoding: [0xc1,0x0a,0x00]        
rorl $0, (%edx) 

// CHECK: rorw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
rorw $0, -485498096(%edx,%eax,4) 

// CHECK: rorw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
rorw $0, 485498096(%edx,%eax,4) 

// CHECK: rorw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc1,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]        
rorw $0, 485498096(%edx) 

// CHECK: rorw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]        
rorw $0, 485498096 

// CHECK: rorw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc1,0x4c,0x02,0x40,0x00]        
rorw $0, 64(%edx,%eax) 

// CHECK: rorw $0, (%edx) 
// CHECK: encoding: [0x66,0xc1,0x0a,0x00]        
rorw $0, (%edx) 

// CHECK: sarb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0xbc,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
sarb $0, -485498096(%edx,%eax,4) 

// CHECK: sarb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0xbc,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
sarb $0, 485498096(%edx,%eax,4) 

// CHECK: sarb $0, 485498096(%edx) 
// CHECK: encoding: [0xc0,0xba,0xf0,0x1c,0xf0,0x1c,0x00]        
sarb $0, 485498096(%edx) 

// CHECK: sarb $0, 485498096 
// CHECK: encoding: [0xc0,0x3d,0xf0,0x1c,0xf0,0x1c,0x00]        
sarb $0, 485498096 

// CHECK: sarb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc0,0x7c,0x02,0x40,0x00]        
sarb $0, 64(%edx,%eax) 

// CHECK: sarb $0, (%edx) 
// CHECK: encoding: [0xc0,0x3a,0x00]        
sarb $0, (%edx) 

// CHECK: sarl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0xbc,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
sarl $0, -485498096(%edx,%eax,4) 

// CHECK: sarl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0xbc,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
sarl $0, 485498096(%edx,%eax,4) 

// CHECK: sarl $0, 485498096(%edx) 
// CHECK: encoding: [0xc1,0xba,0xf0,0x1c,0xf0,0x1c,0x00]        
sarl $0, 485498096(%edx) 

// CHECK: sarl $0, 485498096 
// CHECK: encoding: [0xc1,0x3d,0xf0,0x1c,0xf0,0x1c,0x00]        
sarl $0, 485498096 

// CHECK: sarl $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc1,0x7c,0x02,0x40,0x00]        
sarl $0, 64(%edx,%eax) 

// CHECK: sarl $0, %eax 
// CHECK: encoding: [0xc1,0xf8,0x00]        
sarl $0, %eax 

// CHECK: sarl $0, (%edx) 
// CHECK: encoding: [0xc1,0x3a,0x00]        
sarl $0, (%edx) 

// CHECK: sarw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0xbc,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
sarw $0, -485498096(%edx,%eax,4) 

// CHECK: sarw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0xbc,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
sarw $0, 485498096(%edx,%eax,4) 

// CHECK: sarw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc1,0xba,0xf0,0x1c,0xf0,0x1c,0x00]        
sarw $0, 485498096(%edx) 

// CHECK: sarw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x3d,0xf0,0x1c,0xf0,0x1c,0x00]        
sarw $0, 485498096 

// CHECK: sarw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc1,0x7c,0x02,0x40,0x00]        
sarw $0, 64(%edx,%eax) 

// CHECK: sarw $0, (%edx) 
// CHECK: encoding: [0x66,0xc1,0x3a,0x00]        
sarw $0, (%edx) 

// CHECK: shlb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
shlb $0, -485498096(%edx,%eax,4) 

// CHECK: shlb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
shlb $0, 485498096(%edx,%eax,4) 

// CHECK: shlb $0, 485498096(%edx) 
// CHECK: encoding: [0xc0,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]        
shlb $0, 485498096(%edx) 

// CHECK: shlb $0, 485498096 
// CHECK: encoding: [0xc0,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shlb $0, 485498096 

// CHECK: shlb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc0,0x64,0x02,0x40,0x00]        
shlb $0, 64(%edx,%eax) 

// CHECK: shlb $0, (%edx) 
// CHECK: encoding: [0xc0,0x22,0x00]        
shlb $0, (%edx) 

// CHECK: shll $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
shll $0, -485498096(%edx,%eax,4) 

// CHECK: shll $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
shll $0, 485498096(%edx,%eax,4) 

// CHECK: shll $0, 485498096(%edx) 
// CHECK: encoding: [0xc1,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]        
shll $0, 485498096(%edx) 

// CHECK: shll $0, 485498096 
// CHECK: encoding: [0xc1,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shll $0, 485498096 

// CHECK: shll $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc1,0x64,0x02,0x40,0x00]        
shll $0, 64(%edx,%eax) 

// CHECK: shll $0, %eax 
// CHECK: encoding: [0xc1,0xe0,0x00]        
shll $0, %eax 

// CHECK: shll $0, (%edx) 
// CHECK: encoding: [0xc1,0x22,0x00]        
shll $0, (%edx) 

// CHECK: shlw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
shlw $0, -485498096(%edx,%eax,4) 

// CHECK: shlw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
shlw $0, 485498096(%edx,%eax,4) 

// CHECK: shlw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc1,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]        
shlw $0, 485498096(%edx) 

// CHECK: shlw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shlw $0, 485498096 

// CHECK: shlw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc1,0x64,0x02,0x40,0x00]        
shlw $0, 64(%edx,%eax) 

// CHECK: shlw $0, (%edx) 
// CHECK: encoding: [0x66,0xc1,0x22,0x00]        
shlw $0, (%edx) 

// CHECK: shrb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0xac,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
shrb $0, -485498096(%edx,%eax,4) 

// CHECK: shrb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc0,0xac,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
shrb $0, 485498096(%edx,%eax,4) 

// CHECK: shrb $0, 485498096(%edx) 
// CHECK: encoding: [0xc0,0xaa,0xf0,0x1c,0xf0,0x1c,0x00]        
shrb $0, 485498096(%edx) 

// CHECK: shrb $0, 485498096 
// CHECK: encoding: [0xc0,0x2d,0xf0,0x1c,0xf0,0x1c,0x00]        
shrb $0, 485498096 

// CHECK: shrb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc0,0x6c,0x02,0x40,0x00]        
shrb $0, 64(%edx,%eax) 

// CHECK: shrb $0, (%edx) 
// CHECK: encoding: [0xc0,0x2a,0x00]        
shrb $0, (%edx) 

// CHECK: shrl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0xac,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
shrl $0, -485498096(%edx,%eax,4) 

// CHECK: shrl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc1,0xac,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
shrl $0, 485498096(%edx,%eax,4) 

// CHECK: shrl $0, 485498096(%edx) 
// CHECK: encoding: [0xc1,0xaa,0xf0,0x1c,0xf0,0x1c,0x00]        
shrl $0, 485498096(%edx) 

// CHECK: shrl $0, 485498096 
// CHECK: encoding: [0xc1,0x2d,0xf0,0x1c,0xf0,0x1c,0x00]        
shrl $0, 485498096 

// CHECK: shrl $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc1,0x6c,0x02,0x40,0x00]        
shrl $0, 64(%edx,%eax) 

// CHECK: shrl $0, %eax 
// CHECK: encoding: [0xc1,0xe8,0x00]        
shrl $0, %eax 

// CHECK: shrl $0, (%edx) 
// CHECK: encoding: [0xc1,0x2a,0x00]        
shrl $0, (%edx) 

// CHECK: shrw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0xac,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
shrw $0, -485498096(%edx,%eax,4) 

// CHECK: shrw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc1,0xac,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
shrw $0, 485498096(%edx,%eax,4) 

// CHECK: shrw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc1,0xaa,0xf0,0x1c,0xf0,0x1c,0x00]        
shrw $0, 485498096(%edx) 

// CHECK: shrw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x2d,0xf0,0x1c,0xf0,0x1c,0x00]        
shrw $0, 485498096 

// CHECK: shrw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc1,0x6c,0x02,0x40,0x00]        
shrw $0, 64(%edx,%eax) 

// CHECK: shrw $0, (%edx) 
// CHECK: encoding: [0x66,0xc1,0x2a,0x00]        
shrw $0, (%edx) 

