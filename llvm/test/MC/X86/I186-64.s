// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: enter $0, $0 
// CHECK: encoding: [0xc8,0x00,0x00,0x00]        
enter $0, $0 

// CHECK: imull $0, %r13d, %r13d 
// CHECK: encoding: [0x45,0x6b,0xed,0x00]       
imull $0, %r13d, %r13d 

// CHECK: insb %dx, %es:(%rdi) 
// CHECK: encoding: [0x6c]        
insb %dx, %es:(%rdi) 

// CHECK: insl %dx, %es:(%rdi) 
// CHECK: encoding: [0x6d]        
insl %dx, %es:(%rdi) 

// CHECK: insw %dx, %es:(%rdi) 
// CHECK: encoding: [0x66,0x6d]        
insw %dx, %es:(%rdi) 

// CHECK: leave 
// CHECK: encoding: [0xc9]          
leave 

// CHECK: outsb %gs:(%rsi), %dx 
// CHECK: encoding: [0x65,0x6e]        
outsb %gs:(%rsi), %dx 

// CHECK: outsl %gs:(%rsi), %dx 
// CHECK: encoding: [0x65,0x6f]        
outsl %gs:(%rsi), %dx 

// CHECK: outsw %gs:(%rsi), %dx 
// CHECK: encoding: [0x65,0x66,0x6f]        
outsw %gs:(%rsi), %dx 

// CHECK: pushq $0 
// CHECK: encoding: [0x6a,0x00]         
pushq $0 

// CHECK: rclb $0, 485498096 
// CHECK: encoding: [0xc0,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rclb $0, 485498096 

// CHECK: rclb $0, 64(%rdx) 
// CHECK: encoding: [0xc0,0x52,0x40,0x00]        
rclb $0, 64(%rdx) 

// CHECK: rclb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x54,0x82,0x40,0x00]        
rclb $0, 64(%rdx,%rax,4) 

// CHECK: rclb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x54,0x82,0xc0,0x00]        
rclb $0, -64(%rdx,%rax,4) 

// CHECK: rclb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc0,0x54,0x02,0x40,0x00]        
rclb $0, 64(%rdx,%rax) 

// CHECK: rclb $0, %r14b 
// CHECK: encoding: [0x41,0xc0,0xd6,0x00]        
rclb $0, %r14b 

// CHECK: rclb $0, (%rdx) 
// CHECK: encoding: [0xc0,0x12,0x00]        
rclb $0, (%rdx) 

// CHECK: rcll $0, 485498096 
// CHECK: encoding: [0xc1,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rcll $0, 485498096 

// CHECK: rcll $0, 64(%rdx) 
// CHECK: encoding: [0xc1,0x52,0x40,0x00]        
rcll $0, 64(%rdx) 

// CHECK: rcll $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x54,0x82,0x40,0x00]        
rcll $0, 64(%rdx,%rax,4) 

// CHECK: rcll $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x54,0x82,0xc0,0x00]        
rcll $0, -64(%rdx,%rax,4) 

// CHECK: rcll $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc1,0x54,0x02,0x40,0x00]        
rcll $0, 64(%rdx,%rax) 

// CHECK: rcll $0, %r13d 
// CHECK: encoding: [0x41,0xc1,0xd5,0x00]        
rcll $0, %r13d 

// CHECK: rcll $0, (%rdx) 
// CHECK: encoding: [0xc1,0x12,0x00]        
rcll $0, (%rdx) 

// CHECK: rclq $0, 485498096 
// CHECK: encoding: [0x48,0xc1,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rclq $0, 485498096 

// CHECK: rclq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc1,0x52,0x40,0x00]        
rclq $0, 64(%rdx) 

// CHECK: rclq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x54,0x82,0x40,0x00]        
rclq $0, 64(%rdx,%rax,4) 

// CHECK: rclq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x54,0x82,0xc0,0x00]        
rclq $0, -64(%rdx,%rax,4) 

// CHECK: rclq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc1,0x54,0x02,0x40,0x00]        
rclq $0, 64(%rdx,%rax) 

// CHECK: rclq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc1,0x12,0x00]        
rclq $0, (%rdx) 

// CHECK: rclw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rclw $0, 485498096 

// CHECK: rclw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc1,0x52,0x40,0x00]        
rclw $0, 64(%rdx) 

// CHECK: rclw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x54,0x82,0x40,0x00]        
rclw $0, 64(%rdx,%rax,4) 

// CHECK: rclw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x54,0x82,0xc0,0x00]        
rclw $0, -64(%rdx,%rax,4) 

// CHECK: rclw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc1,0x54,0x02,0x40,0x00]        
rclw $0, 64(%rdx,%rax) 

// CHECK: rclw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xc1,0xd6,0x00]        
rclw $0, %r14w 

// CHECK: rclw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc1,0x12,0x00]        
rclw $0, (%rdx) 

// CHECK: rcrb $0, 485498096 
// CHECK: encoding: [0xc0,0x1c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrb $0, 485498096 

// CHECK: rcrb $0, 64(%rdx) 
// CHECK: encoding: [0xc0,0x5a,0x40,0x00]        
rcrb $0, 64(%rdx) 

// CHECK: rcrb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x5c,0x82,0x40,0x00]        
rcrb $0, 64(%rdx,%rax,4) 

// CHECK: rcrb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x5c,0x82,0xc0,0x00]        
rcrb $0, -64(%rdx,%rax,4) 

// CHECK: rcrb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc0,0x5c,0x02,0x40,0x00]        
rcrb $0, 64(%rdx,%rax) 

// CHECK: rcrb $0, %r14b 
// CHECK: encoding: [0x41,0xc0,0xde,0x00]        
rcrb $0, %r14b 

// CHECK: rcrb $0, (%rdx) 
// CHECK: encoding: [0xc0,0x1a,0x00]        
rcrb $0, (%rdx) 

// CHECK: rcrl $0, 485498096 
// CHECK: encoding: [0xc1,0x1c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrl $0, 485498096 

// CHECK: rcrl $0, 64(%rdx) 
// CHECK: encoding: [0xc1,0x5a,0x40,0x00]        
rcrl $0, 64(%rdx) 

// CHECK: rcrl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x5c,0x82,0x40,0x00]        
rcrl $0, 64(%rdx,%rax,4) 

// CHECK: rcrl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x5c,0x82,0xc0,0x00]        
rcrl $0, -64(%rdx,%rax,4) 

// CHECK: rcrl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc1,0x5c,0x02,0x40,0x00]        
rcrl $0, 64(%rdx,%rax) 

// CHECK: rcrl $0, %r13d 
// CHECK: encoding: [0x41,0xc1,0xdd,0x00]        
rcrl $0, %r13d 

// CHECK: rcrl $0, (%rdx) 
// CHECK: encoding: [0xc1,0x1a,0x00]        
rcrl $0, (%rdx) 

// CHECK: rcrq $0, 485498096 
// CHECK: encoding: [0x48,0xc1,0x1c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrq $0, 485498096 

// CHECK: rcrq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc1,0x5a,0x40,0x00]        
rcrq $0, 64(%rdx) 

// CHECK: rcrq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x5c,0x82,0x40,0x00]        
rcrq $0, 64(%rdx,%rax,4) 

// CHECK: rcrq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x5c,0x82,0xc0,0x00]        
rcrq $0, -64(%rdx,%rax,4) 

// CHECK: rcrq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc1,0x5c,0x02,0x40,0x00]        
rcrq $0, 64(%rdx,%rax) 

// CHECK: rcrq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc1,0x1a,0x00]        
rcrq $0, (%rdx) 

// CHECK: rcrw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x1c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rcrw $0, 485498096 

// CHECK: rcrw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc1,0x5a,0x40,0x00]        
rcrw $0, 64(%rdx) 

// CHECK: rcrw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x5c,0x82,0x40,0x00]        
rcrw $0, 64(%rdx,%rax,4) 

// CHECK: rcrw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x5c,0x82,0xc0,0x00]        
rcrw $0, -64(%rdx,%rax,4) 

// CHECK: rcrw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc1,0x5c,0x02,0x40,0x00]        
rcrw $0, 64(%rdx,%rax) 

// CHECK: rcrw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xc1,0xde,0x00]        
rcrw $0, %r14w 

// CHECK: rcrw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc1,0x1a,0x00]        
rcrw $0, (%rdx) 

// CHECK: rep insb %dx, %es:(%rdi) 
// CHECK: encoding: [0xf3,0x6c]       
rep insb %dx, %es:(%rdi) 

// CHECK: rep insl %dx, %es:(%rdi) 
// CHECK: encoding: [0xf3,0x6d]       
rep insl %dx, %es:(%rdi) 

// CHECK: rep insw %dx, %es:(%rdi) 
// CHECK: encoding: [0xf3,0x66,0x6d]       
rep insw %dx, %es:(%rdi) 

// CHECK: repne insb %dx, %es:(%rdi) 
// CHECK: encoding: [0xf2,0x6c]       
repne insb %dx, %es:(%rdi) 

// CHECK: repne insl %dx, %es:(%rdi) 
// CHECK: encoding: [0xf2,0x6d]       
repne insl %dx, %es:(%rdi) 

// CHECK: repne insw %dx, %es:(%rdi) 
// CHECK: encoding: [0xf2,0x66,0x6d]       
repne insw %dx, %es:(%rdi) 

// CHECK: repne outsb %gs:(%rsi), %dx 
// CHECK: encoding: [0xf2,0x65,0x6e]       
repne outsb %gs:(%rsi), %dx 

// CHECK: repne outsl %gs:(%rsi), %dx 
// CHECK: encoding: [0xf2,0x65,0x6f]       
repne outsl %gs:(%rsi), %dx 

// CHECK: repne outsw %gs:(%rsi), %dx 
// CHECK: encoding: [0xf2,0x65,0x66,0x6f]       
repne outsw %gs:(%rsi), %dx 

// CHECK: rep outsb %gs:(%rsi), %dx 
// CHECK: encoding: [0xf3,0x65,0x6e]       
rep outsb %gs:(%rsi), %dx 

// CHECK: rep outsl %gs:(%rsi), %dx 
// CHECK: encoding: [0xf3,0x65,0x6f]       
rep outsl %gs:(%rsi), %dx 

// CHECK: rep outsw %gs:(%rsi), %dx 
// CHECK: encoding: [0xf3,0x65,0x66,0x6f]       
rep outsw %gs:(%rsi), %dx 

// CHECK: rolb $0, 485498096 
// CHECK: encoding: [0xc0,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rolb $0, 485498096 

// CHECK: rolb $0, 64(%rdx) 
// CHECK: encoding: [0xc0,0x42,0x40,0x00]        
rolb $0, 64(%rdx) 

// CHECK: rolb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x44,0x82,0x40,0x00]        
rolb $0, 64(%rdx,%rax,4) 

// CHECK: rolb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x44,0x82,0xc0,0x00]        
rolb $0, -64(%rdx,%rax,4) 

// CHECK: rolb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc0,0x44,0x02,0x40,0x00]        
rolb $0, 64(%rdx,%rax) 

// CHECK: rolb $0, %r14b 
// CHECK: encoding: [0x41,0xc0,0xc6,0x00]        
rolb $0, %r14b 

// CHECK: rolb $0, (%rdx) 
// CHECK: encoding: [0xc0,0x02,0x00]        
rolb $0, (%rdx) 

// CHECK: roll $0, 485498096 
// CHECK: encoding: [0xc1,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
roll $0, 485498096 

// CHECK: roll $0, 64(%rdx) 
// CHECK: encoding: [0xc1,0x42,0x40,0x00]        
roll $0, 64(%rdx) 

// CHECK: roll $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x44,0x82,0x40,0x00]        
roll $0, 64(%rdx,%rax,4) 

// CHECK: roll $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x44,0x82,0xc0,0x00]        
roll $0, -64(%rdx,%rax,4) 

// CHECK: roll $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc1,0x44,0x02,0x40,0x00]        
roll $0, 64(%rdx,%rax) 

// CHECK: roll $0, %r13d 
// CHECK: encoding: [0x41,0xc1,0xc5,0x00]        
roll $0, %r13d 

// CHECK: roll $0, (%rdx) 
// CHECK: encoding: [0xc1,0x02,0x00]        
roll $0, (%rdx) 

// CHECK: rolq $0, 485498096 
// CHECK: encoding: [0x48,0xc1,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rolq $0, 485498096 

// CHECK: rolq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc1,0x42,0x40,0x00]        
rolq $0, 64(%rdx) 

// CHECK: rolq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x44,0x82,0x40,0x00]        
rolq $0, 64(%rdx,%rax,4) 

// CHECK: rolq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x44,0x82,0xc0,0x00]        
rolq $0, -64(%rdx,%rax,4) 

// CHECK: rolq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc1,0x44,0x02,0x40,0x00]        
rolq $0, 64(%rdx,%rax) 

// CHECK: rolq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc1,0x02,0x00]        
rolq $0, (%rdx) 

// CHECK: rolw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rolw $0, 485498096 

// CHECK: rolw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc1,0x42,0x40,0x00]        
rolw $0, 64(%rdx) 

// CHECK: rolw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x44,0x82,0x40,0x00]        
rolw $0, 64(%rdx,%rax,4) 

// CHECK: rolw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x44,0x82,0xc0,0x00]        
rolw $0, -64(%rdx,%rax,4) 

// CHECK: rolw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc1,0x44,0x02,0x40,0x00]        
rolw $0, 64(%rdx,%rax) 

// CHECK: rolw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xc1,0xc6,0x00]        
rolw $0, %r14w 

// CHECK: rolw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc1,0x02,0x00]        
rolw $0, (%rdx) 

// CHECK: rorb $0, 485498096 
// CHECK: encoding: [0xc0,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rorb $0, 485498096 

// CHECK: rorb $0, 64(%rdx) 
// CHECK: encoding: [0xc0,0x4a,0x40,0x00]        
rorb $0, 64(%rdx) 

// CHECK: rorb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x4c,0x82,0x40,0x00]        
rorb $0, 64(%rdx,%rax,4) 

// CHECK: rorb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x4c,0x82,0xc0,0x00]        
rorb $0, -64(%rdx,%rax,4) 

// CHECK: rorb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc0,0x4c,0x02,0x40,0x00]        
rorb $0, 64(%rdx,%rax) 

// CHECK: rorb $0, %r14b 
// CHECK: encoding: [0x41,0xc0,0xce,0x00]        
rorb $0, %r14b 

// CHECK: rorb $0, (%rdx) 
// CHECK: encoding: [0xc0,0x0a,0x00]        
rorb $0, (%rdx) 

// CHECK: rorl $0, 485498096 
// CHECK: encoding: [0xc1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rorl $0, 485498096 

// CHECK: rorl $0, 64(%rdx) 
// CHECK: encoding: [0xc1,0x4a,0x40,0x00]        
rorl $0, 64(%rdx) 

// CHECK: rorl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x4c,0x82,0x40,0x00]        
rorl $0, 64(%rdx,%rax,4) 

// CHECK: rorl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x4c,0x82,0xc0,0x00]        
rorl $0, -64(%rdx,%rax,4) 

// CHECK: rorl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc1,0x4c,0x02,0x40,0x00]        
rorl $0, 64(%rdx,%rax) 

// CHECK: rorl $0, %r13d 
// CHECK: encoding: [0x41,0xc1,0xcd,0x00]        
rorl $0, %r13d 

// CHECK: rorl $0, (%rdx) 
// CHECK: encoding: [0xc1,0x0a,0x00]        
rorl $0, (%rdx) 

// CHECK: rorq $0, 485498096 
// CHECK: encoding: [0x48,0xc1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rorq $0, 485498096 

// CHECK: rorq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc1,0x4a,0x40,0x00]        
rorq $0, 64(%rdx) 

// CHECK: rorq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x4c,0x82,0x40,0x00]        
rorq $0, 64(%rdx,%rax,4) 

// CHECK: rorq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x4c,0x82,0xc0,0x00]        
rorq $0, -64(%rdx,%rax,4) 

// CHECK: rorq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc1,0x4c,0x02,0x40,0x00]        
rorq $0, 64(%rdx,%rax) 

// CHECK: rorq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc1,0x0a,0x00]        
rorq $0, (%rdx) 

// CHECK: rorw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
rorw $0, 485498096 

// CHECK: rorw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc1,0x4a,0x40,0x00]        
rorw $0, 64(%rdx) 

// CHECK: rorw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x4c,0x82,0x40,0x00]        
rorw $0, 64(%rdx,%rax,4) 

// CHECK: rorw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x4c,0x82,0xc0,0x00]        
rorw $0, -64(%rdx,%rax,4) 

// CHECK: rorw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc1,0x4c,0x02,0x40,0x00]        
rorw $0, 64(%rdx,%rax) 

// CHECK: rorw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xc1,0xce,0x00]        
rorw $0, %r14w 

// CHECK: rorw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc1,0x0a,0x00]        
rorw $0, (%rdx) 

// CHECK: sarb $0, 485498096 
// CHECK: encoding: [0xc0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
sarb $0, 485498096 

// CHECK: sarb $0, 64(%rdx) 
// CHECK: encoding: [0xc0,0x7a,0x40,0x00]        
sarb $0, 64(%rdx) 

// CHECK: sarb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x7c,0x82,0x40,0x00]        
sarb $0, 64(%rdx,%rax,4) 

// CHECK: sarb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x7c,0x82,0xc0,0x00]        
sarb $0, -64(%rdx,%rax,4) 

// CHECK: sarb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc0,0x7c,0x02,0x40,0x00]        
sarb $0, 64(%rdx,%rax) 

// CHECK: sarb $0, %r14b 
// CHECK: encoding: [0x41,0xc0,0xfe,0x00]        
sarb $0, %r14b 

// CHECK: sarb $0, (%rdx) 
// CHECK: encoding: [0xc0,0x3a,0x00]        
sarb $0, (%rdx) 

// CHECK: sarl $0, 485498096 
// CHECK: encoding: [0xc1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
sarl $0, 485498096 

// CHECK: sarl $0, 64(%rdx) 
// CHECK: encoding: [0xc1,0x7a,0x40,0x00]        
sarl $0, 64(%rdx) 

// CHECK: sarl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x7c,0x82,0x40,0x00]        
sarl $0, 64(%rdx,%rax,4) 

// CHECK: sarl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x7c,0x82,0xc0,0x00]        
sarl $0, -64(%rdx,%rax,4) 

// CHECK: sarl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc1,0x7c,0x02,0x40,0x00]        
sarl $0, 64(%rdx,%rax) 

// CHECK: sarl $0, %r13d 
// CHECK: encoding: [0x41,0xc1,0xfd,0x00]        
sarl $0, %r13d 

// CHECK: sarl $0, (%rdx) 
// CHECK: encoding: [0xc1,0x3a,0x00]        
sarl $0, (%rdx) 

// CHECK: sarq $0, 485498096 
// CHECK: encoding: [0x48,0xc1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
sarq $0, 485498096 

// CHECK: sarq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc1,0x7a,0x40,0x00]        
sarq $0, 64(%rdx) 

// CHECK: sarq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x7c,0x82,0x40,0x00]        
sarq $0, 64(%rdx,%rax,4) 

// CHECK: sarq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x7c,0x82,0xc0,0x00]        
sarq $0, -64(%rdx,%rax,4) 

// CHECK: sarq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc1,0x7c,0x02,0x40,0x00]        
sarq $0, 64(%rdx,%rax) 

// CHECK: sarq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc1,0x3a,0x00]        
sarq $0, (%rdx) 

// CHECK: sarw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
sarw $0, 485498096 

// CHECK: sarw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc1,0x7a,0x40,0x00]        
sarw $0, 64(%rdx) 

// CHECK: sarw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x7c,0x82,0x40,0x00]        
sarw $0, 64(%rdx,%rax,4) 

// CHECK: sarw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x7c,0x82,0xc0,0x00]        
sarw $0, -64(%rdx,%rax,4) 

// CHECK: sarw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc1,0x7c,0x02,0x40,0x00]        
sarw $0, 64(%rdx,%rax) 

// CHECK: sarw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xc1,0xfe,0x00]        
sarw $0, %r14w 

// CHECK: sarw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc1,0x3a,0x00]        
sarw $0, (%rdx) 

// CHECK: shlb $0, 485498096 
// CHECK: encoding: [0xc0,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shlb $0, 485498096 

// CHECK: shlb $0, 64(%rdx) 
// CHECK: encoding: [0xc0,0x62,0x40,0x00]        
shlb $0, 64(%rdx) 

// CHECK: shlb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x64,0x82,0x40,0x00]        
shlb $0, 64(%rdx,%rax,4) 

// CHECK: shlb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x64,0x82,0xc0,0x00]        
shlb $0, -64(%rdx,%rax,4) 

// CHECK: shlb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc0,0x64,0x02,0x40,0x00]        
shlb $0, 64(%rdx,%rax) 

// CHECK: shlb $0, %r14b 
// CHECK: encoding: [0x41,0xc0,0xe6,0x00]        
shlb $0, %r14b 

// CHECK: shlb $0, (%rdx) 
// CHECK: encoding: [0xc0,0x22,0x00]        
shlb $0, (%rdx) 

// CHECK: shll $0, 485498096 
// CHECK: encoding: [0xc1,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shll $0, 485498096 

// CHECK: shll $0, 64(%rdx) 
// CHECK: encoding: [0xc1,0x62,0x40,0x00]        
shll $0, 64(%rdx) 

// CHECK: shll $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x64,0x82,0x40,0x00]        
shll $0, 64(%rdx,%rax,4) 

// CHECK: shll $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x64,0x82,0xc0,0x00]        
shll $0, -64(%rdx,%rax,4) 

// CHECK: shll $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc1,0x64,0x02,0x40,0x00]        
shll $0, 64(%rdx,%rax) 

// CHECK: shll $0, %r13d 
// CHECK: encoding: [0x41,0xc1,0xe5,0x00]        
shll $0, %r13d 

// CHECK: shll $0, (%rdx) 
// CHECK: encoding: [0xc1,0x22,0x00]        
shll $0, (%rdx) 

// CHECK: shlq $0, 485498096 
// CHECK: encoding: [0x48,0xc1,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shlq $0, 485498096 

// CHECK: shlq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc1,0x62,0x40,0x00]        
shlq $0, 64(%rdx) 

// CHECK: shlq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x64,0x82,0x40,0x00]        
shlq $0, 64(%rdx,%rax,4) 

// CHECK: shlq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x64,0x82,0xc0,0x00]        
shlq $0, -64(%rdx,%rax,4) 

// CHECK: shlq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc1,0x64,0x02,0x40,0x00]        
shlq $0, 64(%rdx,%rax) 

// CHECK: shlq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc1,0x22,0x00]        
shlq $0, (%rdx) 

// CHECK: shlw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shlw $0, 485498096 

// CHECK: shlw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc1,0x62,0x40,0x00]        
shlw $0, 64(%rdx) 

// CHECK: shlw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x64,0x82,0x40,0x00]        
shlw $0, 64(%rdx,%rax,4) 

// CHECK: shlw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x64,0x82,0xc0,0x00]        
shlw $0, -64(%rdx,%rax,4) 

// CHECK: shlw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc1,0x64,0x02,0x40,0x00]        
shlw $0, 64(%rdx,%rax) 

// CHECK: shlw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xc1,0xe6,0x00]        
shlw $0, %r14w 

// CHECK: shlw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc1,0x22,0x00]        
shlw $0, (%rdx) 

// CHECK: shrb $0, 485498096 
// CHECK: encoding: [0xc0,0x2c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shrb $0, 485498096 

// CHECK: shrb $0, 64(%rdx) 
// CHECK: encoding: [0xc0,0x6a,0x40,0x00]        
shrb $0, 64(%rdx) 

// CHECK: shrb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x6c,0x82,0x40,0x00]        
shrb $0, 64(%rdx,%rax,4) 

// CHECK: shrb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc0,0x6c,0x82,0xc0,0x00]        
shrb $0, -64(%rdx,%rax,4) 

// CHECK: shrb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc0,0x6c,0x02,0x40,0x00]        
shrb $0, 64(%rdx,%rax) 

// CHECK: shrb $0, %r14b 
// CHECK: encoding: [0x41,0xc0,0xee,0x00]        
shrb $0, %r14b 

// CHECK: shrb $0, (%rdx) 
// CHECK: encoding: [0xc0,0x2a,0x00]        
shrb $0, (%rdx) 

// CHECK: shrl $0, 485498096 
// CHECK: encoding: [0xc1,0x2c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shrl $0, 485498096 

// CHECK: shrl $0, 64(%rdx) 
// CHECK: encoding: [0xc1,0x6a,0x40,0x00]        
shrl $0, 64(%rdx) 

// CHECK: shrl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x6c,0x82,0x40,0x00]        
shrl $0, 64(%rdx,%rax,4) 

// CHECK: shrl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc1,0x6c,0x82,0xc0,0x00]        
shrl $0, -64(%rdx,%rax,4) 

// CHECK: shrl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc1,0x6c,0x02,0x40,0x00]        
shrl $0, 64(%rdx,%rax) 

// CHECK: shrl $0, %r13d 
// CHECK: encoding: [0x41,0xc1,0xed,0x00]        
shrl $0, %r13d 

// CHECK: shrl $0, (%rdx) 
// CHECK: encoding: [0xc1,0x2a,0x00]        
shrl $0, (%rdx) 

// CHECK: shrq $0, 485498096 
// CHECK: encoding: [0x48,0xc1,0x2c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shrq $0, 485498096 

// CHECK: shrq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc1,0x6a,0x40,0x00]        
shrq $0, 64(%rdx) 

// CHECK: shrq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x6c,0x82,0x40,0x00]        
shrq $0, 64(%rdx,%rax,4) 

// CHECK: shrq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc1,0x6c,0x82,0xc0,0x00]        
shrq $0, -64(%rdx,%rax,4) 

// CHECK: shrq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc1,0x6c,0x02,0x40,0x00]        
shrq $0, 64(%rdx,%rax) 

// CHECK: shrq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc1,0x2a,0x00]        
shrq $0, (%rdx) 

// CHECK: shrw $0, 485498096 
// CHECK: encoding: [0x66,0xc1,0x2c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
shrw $0, 485498096 

// CHECK: shrw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc1,0x6a,0x40,0x00]        
shrw $0, 64(%rdx) 

// CHECK: shrw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x6c,0x82,0x40,0x00]        
shrw $0, 64(%rdx,%rax,4) 

// CHECK: shrw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc1,0x6c,0x82,0xc0,0x00]        
shrw $0, -64(%rdx,%rax,4) 

// CHECK: shrw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc1,0x6c,0x02,0x40,0x00]        
shrw $0, 64(%rdx,%rax) 

// CHECK: shrw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xc1,0xee,0x00]        
shrw $0, %r14w 

// CHECK: shrw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc1,0x2a,0x00]        
shrw $0, (%rdx) 

