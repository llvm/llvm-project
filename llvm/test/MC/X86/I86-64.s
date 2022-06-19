// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: adcb $0, 485498096 
// CHECK: encoding: [0x80,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
adcb $0, 485498096 

// CHECK: adcb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x52,0x40,0x00]        
adcb $0, 64(%rdx) 

// CHECK: adcb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x54,0x82,0x40,0x00]        
adcb $0, 64(%rdx,%rax,4) 

// CHECK: adcb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x54,0x82,0xc0,0x00]        
adcb $0, -64(%rdx,%rax,4) 

// CHECK: adcb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x54,0x02,0x40,0x00]        
adcb $0, 64(%rdx,%rax) 

// CHECK: adcb $0, %al 
// CHECK: encoding: [0x14,0x00]        
adcb $0, %al 

// CHECK: adcb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xd6,0x00]        
adcb $0, %r14b 

// CHECK: adcb $0, (%rdx) 
// CHECK: encoding: [0x80,0x12,0x00]        
adcb $0, (%rdx) 

// CHECK: adcb 485498096, %r14b 
// CHECK: encoding: [0x44,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
adcb 485498096, %r14b 

// CHECK: adcb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x12,0x72,0x40]        
adcb 64(%rdx), %r14b 

// CHECK: adcb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x12,0x74,0x82,0x40]        
adcb 64(%rdx,%rax,4), %r14b 

// CHECK: adcb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x12,0x74,0x82,0xc0]        
adcb -64(%rdx,%rax,4), %r14b 

// CHECK: adcb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x12,0x74,0x02,0x40]        
adcb 64(%rdx,%rax), %r14b 

// CHECK: adcb %r14b, 485498096 
// CHECK: encoding: [0x44,0x10,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
adcb %r14b, 485498096 

// CHECK: adcb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x10,0x72,0x40]        
adcb %r14b, 64(%rdx) 

// CHECK: adcb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x10,0x74,0x82,0x40]        
adcb %r14b, 64(%rdx,%rax,4) 

// CHECK: adcb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x10,0x74,0x82,0xc0]        
adcb %r14b, -64(%rdx,%rax,4) 

// CHECK: adcb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x10,0x74,0x02,0x40]        
adcb %r14b, 64(%rdx,%rax) 

// CHECK: adcb %r14b, %r14b 
// CHECK: encoding: [0x45,0x10,0xf6]        
adcb %r14b, %r14b 

// CHECK: adcb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x10,0x32]        
adcb %r14b, (%rdx) 

// CHECK: adcb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x12,0x32]        
adcb (%rdx), %r14b 

// CHECK: adcl $0, 485498096 
// CHECK: encoding: [0x83,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
adcl $0, 485498096 

// CHECK: adcl $0, 64(%rdx) 
// CHECK: encoding: [0x83,0x52,0x40,0x00]        
adcl $0, 64(%rdx) 

// CHECK: adcl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x54,0x82,0x40,0x00]        
adcl $0, 64(%rdx,%rax,4) 

// CHECK: adcl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x54,0x82,0xc0,0x00]        
adcl $0, -64(%rdx,%rax,4) 

// CHECK: adcl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x83,0x54,0x02,0x40,0x00]        
adcl $0, 64(%rdx,%rax) 

// CHECK: adcl $0, %eax 
// CHECK: encoding: [0x83,0xd0,0x00]        
adcl $0, %eax 

// CHECK: adcl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xd5,0x00]        
adcl $0, %r13d 

// CHECK: adcl $0, (%rdx) 
// CHECK: encoding: [0x83,0x12,0x00]        
adcl $0, (%rdx) 

// CHECK: adcl %r13d, %r13d 
// CHECK: encoding: [0x45,0x11,0xed]        
adcl %r13d, %r13d 

// CHECK: adcq $0, 485498096 
// CHECK: encoding: [0x48,0x83,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
adcq $0, 485498096 

// CHECK: adcq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0x83,0x52,0x40,0x00]        
adcq $0, 64(%rdx) 

// CHECK: adcq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x54,0x82,0x40,0x00]        
adcq $0, 64(%rdx,%rax,4) 

// CHECK: adcq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x54,0x82,0xc0,0x00]        
adcq $0, -64(%rdx,%rax,4) 

// CHECK: adcq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x83,0x54,0x02,0x40,0x00]        
adcq $0, 64(%rdx,%rax) 

// CHECK: adcq $0, (%rdx) 
// CHECK: encoding: [0x48,0x83,0x12,0x00]        
adcq $0, (%rdx) 

// CHECK: adcw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
adcw $0, 485498096 

// CHECK: adcw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0x83,0x52,0x40,0x00]        
adcw $0, 64(%rdx) 

// CHECK: adcw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x54,0x82,0x40,0x00]        
adcw $0, 64(%rdx,%rax,4) 

// CHECK: adcw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x54,0x82,0xc0,0x00]        
adcw $0, -64(%rdx,%rax,4) 

// CHECK: adcw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x83,0x54,0x02,0x40,0x00]        
adcw $0, 64(%rdx,%rax) 

// CHECK: adcw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0x83,0xd6,0x00]        
adcw $0, %r14w 

// CHECK: adcw $0, (%rdx) 
// CHECK: encoding: [0x66,0x83,0x12,0x00]        
adcw $0, (%rdx) 

// CHECK: adcw 485498096, %r14w 
// CHECK: encoding: [0x66,0x44,0x13,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
adcw 485498096, %r14w 

// CHECK: adcw 64(%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x13,0x72,0x40]        
adcw 64(%rdx), %r14w 

// CHECK: adcw 64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x13,0x74,0x82,0x40]        
adcw 64(%rdx,%rax,4), %r14w 

// CHECK: adcw -64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x13,0x74,0x82,0xc0]        
adcw -64(%rdx,%rax,4), %r14w 

// CHECK: adcw 64(%rdx,%rax), %r14w 
// CHECK: encoding: [0x66,0x44,0x13,0x74,0x02,0x40]        
adcw 64(%rdx,%rax), %r14w 

// CHECK: adcw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x11,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
adcw %r14w, 485498096 

// CHECK: adcw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x11,0x72,0x40]        
adcw %r14w, 64(%rdx) 

// CHECK: adcw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x11,0x74,0x82,0x40]        
adcw %r14w, 64(%rdx,%rax,4) 

// CHECK: adcw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x11,0x74,0x82,0xc0]        
adcw %r14w, -64(%rdx,%rax,4) 

// CHECK: adcw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x11,0x74,0x02,0x40]        
adcw %r14w, 64(%rdx,%rax) 

// CHECK: adcw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x11,0xf6]        
adcw %r14w, %r14w 

// CHECK: adcw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x11,0x32]        
adcw %r14w, (%rdx) 

// CHECK: adcw (%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x13,0x32]        
adcw (%rdx), %r14w 

// CHECK: addb $0, 485498096 
// CHECK: encoding: [0x80,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
addb $0, 485498096 

// CHECK: addb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x42,0x40,0x00]        
addb $0, 64(%rdx) 

// CHECK: addb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x44,0x82,0x40,0x00]        
addb $0, 64(%rdx,%rax,4) 

// CHECK: addb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x44,0x82,0xc0,0x00]        
addb $0, -64(%rdx,%rax,4) 

// CHECK: addb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x44,0x02,0x40,0x00]        
addb $0, 64(%rdx,%rax) 

// CHECK: addb $0, %al 
// CHECK: encoding: [0x04,0x00]        
addb $0, %al 

// CHECK: addb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xc6,0x00]        
addb $0, %r14b 

// CHECK: addb $0, (%rdx) 
// CHECK: encoding: [0x80,0x02,0x00]        
addb $0, (%rdx) 

// CHECK: addb 485498096, %r14b 
// CHECK: encoding: [0x44,0x02,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
addb 485498096, %r14b 

// CHECK: addb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x02,0x72,0x40]        
addb 64(%rdx), %r14b 

// CHECK: addb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x02,0x74,0x82,0x40]        
addb 64(%rdx,%rax,4), %r14b 

// CHECK: addb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x02,0x74,0x82,0xc0]        
addb -64(%rdx,%rax,4), %r14b 

// CHECK: addb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x02,0x74,0x02,0x40]        
addb 64(%rdx,%rax), %r14b 

// CHECK: addb %r14b, 485498096 
// CHECK: encoding: [0x44,0x00,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
addb %r14b, 485498096 

// CHECK: addb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x00,0x72,0x40]        
addb %r14b, 64(%rdx) 

// CHECK: addb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x00,0x74,0x82,0x40]        
addb %r14b, 64(%rdx,%rax,4) 

// CHECK: addb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x00,0x74,0x82,0xc0]        
addb %r14b, -64(%rdx,%rax,4) 

// CHECK: addb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x00,0x74,0x02,0x40]        
addb %r14b, 64(%rdx,%rax) 

// CHECK: addb %r14b, %r14b 
// CHECK: encoding: [0x45,0x00,0xf6]        
addb %r14b, %r14b 

// CHECK: addb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x00,0x32]        
addb %r14b, (%rdx) 

// CHECK: addb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x02,0x32]        
addb (%rdx), %r14b 

// CHECK: addl $0, 485498096 
// CHECK: encoding: [0x83,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
addl $0, 485498096 

// CHECK: addl $0, 64(%rdx) 
// CHECK: encoding: [0x83,0x42,0x40,0x00]        
addl $0, 64(%rdx) 

// CHECK: addl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x44,0x82,0x40,0x00]        
addl $0, 64(%rdx,%rax,4) 

// CHECK: addl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x44,0x82,0xc0,0x00]        
addl $0, -64(%rdx,%rax,4) 

// CHECK: addl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x83,0x44,0x02,0x40,0x00]        
addl $0, 64(%rdx,%rax) 

// CHECK: addl $0, %eax 
// CHECK: encoding: [0x83,0xc0,0x00]        
addl $0, %eax 

// CHECK: addl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xc5,0x00]        
addl $0, %r13d 

// CHECK: addl $0, (%rdx) 
// CHECK: encoding: [0x83,0x02,0x00]        
addl $0, (%rdx) 

// CHECK: addl %r13d, %r13d 
// CHECK: encoding: [0x45,0x01,0xed]        
addl %r13d, %r13d 

// CHECK: addq $0, 485498096 
// CHECK: encoding: [0x48,0x83,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
addq $0, 485498096 

// CHECK: addq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0x83,0x42,0x40,0x00]        
addq $0, 64(%rdx) 

// CHECK: addq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x44,0x82,0x40,0x00]        
addq $0, 64(%rdx,%rax,4) 

// CHECK: addq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x44,0x82,0xc0,0x00]        
addq $0, -64(%rdx,%rax,4) 

// CHECK: addq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x83,0x44,0x02,0x40,0x00]        
addq $0, 64(%rdx,%rax) 

// CHECK: addq $0, (%rdx) 
// CHECK: encoding: [0x48,0x83,0x02,0x00]        
addq $0, (%rdx) 

// CHECK: addw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
addw $0, 485498096 

// CHECK: addw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0x83,0x42,0x40,0x00]        
addw $0, 64(%rdx) 

// CHECK: addw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x44,0x82,0x40,0x00]        
addw $0, 64(%rdx,%rax,4) 

// CHECK: addw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x44,0x82,0xc0,0x00]        
addw $0, -64(%rdx,%rax,4) 

// CHECK: addw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x83,0x44,0x02,0x40,0x00]        
addw $0, 64(%rdx,%rax) 

// CHECK: addw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0x83,0xc6,0x00]        
addw $0, %r14w 

// CHECK: addw $0, (%rdx) 
// CHECK: encoding: [0x66,0x83,0x02,0x00]        
addw $0, (%rdx) 

// CHECK: addw 485498096, %r14w 
// CHECK: encoding: [0x66,0x44,0x03,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
addw 485498096, %r14w 

// CHECK: addw 64(%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x03,0x72,0x40]        
addw 64(%rdx), %r14w 

// CHECK: addw 64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x03,0x74,0x82,0x40]        
addw 64(%rdx,%rax,4), %r14w 

// CHECK: addw -64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x03,0x74,0x82,0xc0]        
addw -64(%rdx,%rax,4), %r14w 

// CHECK: addw 64(%rdx,%rax), %r14w 
// CHECK: encoding: [0x66,0x44,0x03,0x74,0x02,0x40]        
addw 64(%rdx,%rax), %r14w 

// CHECK: addw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x01,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
addw %r14w, 485498096 

// CHECK: addw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x01,0x72,0x40]        
addw %r14w, 64(%rdx) 

// CHECK: addw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x01,0x74,0x82,0x40]        
addw %r14w, 64(%rdx,%rax,4) 

// CHECK: addw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x01,0x74,0x82,0xc0]        
addw %r14w, -64(%rdx,%rax,4) 

// CHECK: addw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x01,0x74,0x02,0x40]        
addw %r14w, 64(%rdx,%rax) 

// CHECK: addw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x01,0xf6]        
addw %r14w, %r14w 

// CHECK: addw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x01,0x32]        
addw %r14w, (%rdx) 

// CHECK: addw (%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x03,0x32]        
addw (%rdx), %r14w 

// CHECK: andb $0, 485498096 
// CHECK: encoding: [0x80,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
andb $0, 485498096 

// CHECK: andb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x62,0x40,0x00]        
andb $0, 64(%rdx) 

// CHECK: andb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x64,0x82,0x40,0x00]        
andb $0, 64(%rdx,%rax,4) 

// CHECK: andb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x64,0x82,0xc0,0x00]        
andb $0, -64(%rdx,%rax,4) 

// CHECK: andb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x64,0x02,0x40,0x00]        
andb $0, 64(%rdx,%rax) 

// CHECK: andb $0, %al 
// CHECK: encoding: [0x24,0x00]        
andb $0, %al 

// CHECK: andb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xe6,0x00]        
andb $0, %r14b 

// CHECK: andb $0, (%rdx) 
// CHECK: encoding: [0x80,0x22,0x00]        
andb $0, (%rdx) 

// CHECK: andb 485498096, %r14b 
// CHECK: encoding: [0x44,0x22,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
andb 485498096, %r14b 

// CHECK: andb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x22,0x72,0x40]        
andb 64(%rdx), %r14b 

// CHECK: andb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x22,0x74,0x82,0x40]        
andb 64(%rdx,%rax,4), %r14b 

// CHECK: andb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x22,0x74,0x82,0xc0]        
andb -64(%rdx,%rax,4), %r14b 

// CHECK: andb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x22,0x74,0x02,0x40]        
andb 64(%rdx,%rax), %r14b 

// CHECK: andb %r14b, 485498096 
// CHECK: encoding: [0x44,0x20,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
andb %r14b, 485498096 

// CHECK: andb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x20,0x72,0x40]        
andb %r14b, 64(%rdx) 

// CHECK: andb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x20,0x74,0x82,0x40]        
andb %r14b, 64(%rdx,%rax,4) 

// CHECK: andb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x20,0x74,0x82,0xc0]        
andb %r14b, -64(%rdx,%rax,4) 

// CHECK: andb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x20,0x74,0x02,0x40]        
andb %r14b, 64(%rdx,%rax) 

// CHECK: andb %r14b, %r14b 
// CHECK: encoding: [0x45,0x20,0xf6]        
andb %r14b, %r14b 

// CHECK: andb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x20,0x32]        
andb %r14b, (%rdx) 

// CHECK: andb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x22,0x32]        
andb (%rdx), %r14b 

// CHECK: andl $0, 485498096 
// CHECK: encoding: [0x83,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
andl $0, 485498096 

// CHECK: andl $0, 64(%rdx) 
// CHECK: encoding: [0x83,0x62,0x40,0x00]        
andl $0, 64(%rdx) 

// CHECK: andl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x64,0x82,0x40,0x00]        
andl $0, 64(%rdx,%rax,4) 

// CHECK: andl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x64,0x82,0xc0,0x00]        
andl $0, -64(%rdx,%rax,4) 

// CHECK: andl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x83,0x64,0x02,0x40,0x00]        
andl $0, 64(%rdx,%rax) 

// CHECK: andl $0, %eax 
// CHECK: encoding: [0x83,0xe0,0x00]        
andl $0, %eax 

// CHECK: andl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xe5,0x00]        
andl $0, %r13d 

// CHECK: andl $0, (%rdx) 
// CHECK: encoding: [0x83,0x22,0x00]        
andl $0, (%rdx) 

// CHECK: andl %r13d, %r13d 
// CHECK: encoding: [0x45,0x21,0xed]        
andl %r13d, %r13d 

// CHECK: andq $0, 485498096 
// CHECK: encoding: [0x48,0x83,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
andq $0, 485498096 

// CHECK: andq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0x83,0x62,0x40,0x00]        
andq $0, 64(%rdx) 

// CHECK: andq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x64,0x82,0x40,0x00]        
andq $0, 64(%rdx,%rax,4) 

// CHECK: andq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x64,0x82,0xc0,0x00]        
andq $0, -64(%rdx,%rax,4) 

// CHECK: andq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x83,0x64,0x02,0x40,0x00]        
andq $0, 64(%rdx,%rax) 

// CHECK: andq $0, (%rdx) 
// CHECK: encoding: [0x48,0x83,0x22,0x00]        
andq $0, (%rdx) 

// CHECK: andw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
andw $0, 485498096 

// CHECK: andw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0x83,0x62,0x40,0x00]        
andw $0, 64(%rdx) 

// CHECK: andw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x64,0x82,0x40,0x00]        
andw $0, 64(%rdx,%rax,4) 

// CHECK: andw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x64,0x82,0xc0,0x00]        
andw $0, -64(%rdx,%rax,4) 

// CHECK: andw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x83,0x64,0x02,0x40,0x00]        
andw $0, 64(%rdx,%rax) 

// CHECK: andw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0x83,0xe6,0x00]        
andw $0, %r14w 

// CHECK: andw $0, (%rdx) 
// CHECK: encoding: [0x66,0x83,0x22,0x00]        
andw $0, (%rdx) 

// CHECK: andw 485498096, %r14w 
// CHECK: encoding: [0x66,0x44,0x23,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
andw 485498096, %r14w 

// CHECK: andw 64(%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x23,0x72,0x40]        
andw 64(%rdx), %r14w 

// CHECK: andw 64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x23,0x74,0x82,0x40]        
andw 64(%rdx,%rax,4), %r14w 

// CHECK: andw -64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x23,0x74,0x82,0xc0]        
andw -64(%rdx,%rax,4), %r14w 

// CHECK: andw 64(%rdx,%rax), %r14w 
// CHECK: encoding: [0x66,0x44,0x23,0x74,0x02,0x40]        
andw 64(%rdx,%rax), %r14w 

// CHECK: andw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x21,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
andw %r14w, 485498096 

// CHECK: andw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x21,0x72,0x40]        
andw %r14w, 64(%rdx) 

// CHECK: andw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x21,0x74,0x82,0x40]        
andw %r14w, 64(%rdx,%rax,4) 

// CHECK: andw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x21,0x74,0x82,0xc0]        
andw %r14w, -64(%rdx,%rax,4) 

// CHECK: andw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x21,0x74,0x02,0x40]        
andw %r14w, 64(%rdx,%rax) 

// CHECK: andw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x21,0xf6]        
andw %r14w, %r14w 

// CHECK: andw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x21,0x32]        
andw %r14w, (%rdx) 

// CHECK: andw (%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x23,0x32]        
andw (%rdx), %r14w 

// CHECK: callq 64 
// CHECK: encoding: [0xe8,A,A,A,A]         
callq 64 

// CHECK: callw 64 
// CHECK: encoding: [0x66,0xe8,A,A]         
callw 64 

// CHECK: cbtw 
// CHECK: encoding: [0x66,0x98]          
cbtw 

// CHECK: cwtl 
// CHECK: encoding: [0x98]          
cwtl 

// CHECK: cltq 
// CHECK: encoding: [0x48,0x98]          
cltq 

// CHECK: clc 
// CHECK: encoding: [0xf8]          
clc 

// CHECK: cld 
// CHECK: encoding: [0xfc]          
cld 

// CHECK: cli 
// CHECK: encoding: [0xfa]          
cli 

// CHECK: cwtd 
// CHECK: encoding: [0x66,0x99]          
cwtd 

// CHECK: cltd 
// CHECK: encoding: [0x99]          
cltd 

// CHECK: cqto 
// CHECK: encoding: [0x48,0x99]          
cqto 

// CHECK: cmc 
// CHECK: encoding: [0xf5]          
cmc 

// CHECK: cmpb $0, 485498096 
// CHECK: encoding: [0x80,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpb $0, 485498096 

// CHECK: cmpb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x7a,0x40,0x00]        
cmpb $0, 64(%rdx) 

// CHECK: cmpb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x7c,0x82,0x40,0x00]        
cmpb $0, 64(%rdx,%rax,4) 

// CHECK: cmpb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x7c,0x82,0xc0,0x00]        
cmpb $0, -64(%rdx,%rax,4) 

// CHECK: cmpb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x7c,0x02,0x40,0x00]        
cmpb $0, 64(%rdx,%rax) 

// CHECK: cmpb $0, %al 
// CHECK: encoding: [0x3c,0x00]        
cmpb $0, %al 

// CHECK: cmpb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xfe,0x00]        
cmpb $0, %r14b 

// CHECK: cmpb $0, (%rdx) 
// CHECK: encoding: [0x80,0x3a,0x00]        
cmpb $0, (%rdx) 

// CHECK: cmpb 485498096, %r14b 
// CHECK: encoding: [0x44,0x3a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
cmpb 485498096, %r14b 

// CHECK: cmpb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x3a,0x72,0x40]        
cmpb 64(%rdx), %r14b 

// CHECK: cmpb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x3a,0x74,0x82,0x40]        
cmpb 64(%rdx,%rax,4), %r14b 

// CHECK: cmpb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x3a,0x74,0x82,0xc0]        
cmpb -64(%rdx,%rax,4), %r14b 

// CHECK: cmpb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x3a,0x74,0x02,0x40]        
cmpb 64(%rdx,%rax), %r14b 

// CHECK: cmpb %r14b, 485498096 
// CHECK: encoding: [0x44,0x38,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
cmpb %r14b, 485498096 

// CHECK: cmpb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x38,0x72,0x40]        
cmpb %r14b, 64(%rdx) 

// CHECK: cmpb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x38,0x74,0x82,0x40]        
cmpb %r14b, 64(%rdx,%rax,4) 

// CHECK: cmpb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x38,0x74,0x82,0xc0]        
cmpb %r14b, -64(%rdx,%rax,4) 

// CHECK: cmpb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x38,0x74,0x02,0x40]        
cmpb %r14b, 64(%rdx,%rax) 

// CHECK: cmpb %r14b, %r14b 
// CHECK: encoding: [0x45,0x38,0xf6]        
cmpb %r14b, %r14b 

// CHECK: cmpb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x38,0x32]        
cmpb %r14b, (%rdx) 

// CHECK: cmpb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x3a,0x32]        
cmpb (%rdx), %r14b 

// CHECK: cmpl $0, 485498096 
// CHECK: encoding: [0x83,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpl $0, 485498096 

// CHECK: cmpl $0, 64(%rdx) 
// CHECK: encoding: [0x83,0x7a,0x40,0x00]        
cmpl $0, 64(%rdx) 

// CHECK: cmpl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x7c,0x82,0x40,0x00]        
cmpl $0, 64(%rdx,%rax,4) 

// CHECK: cmpl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x7c,0x82,0xc0,0x00]        
cmpl $0, -64(%rdx,%rax,4) 

// CHECK: cmpl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x83,0x7c,0x02,0x40,0x00]        
cmpl $0, 64(%rdx,%rax) 

// CHECK: cmpl $0, %eax 
// CHECK: encoding: [0x83,0xf8,0x00]        
cmpl $0, %eax 

// CHECK: cmpl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xfd,0x00]        
cmpl $0, %r13d 

// CHECK: cmpl $0, (%rdx) 
// CHECK: encoding: [0x83,0x3a,0x00]        
cmpl $0, (%rdx) 

// CHECK: cmpl %r13d, %r13d 
// CHECK: encoding: [0x45,0x39,0xed]        
cmpl %r13d, %r13d 

// CHECK: cmpq $0, 485498096 
// CHECK: encoding: [0x48,0x83,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpq $0, 485498096 

// CHECK: cmpq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0x83,0x7a,0x40,0x00]        
cmpq $0, 64(%rdx) 

// CHECK: cmpq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x7c,0x82,0x40,0x00]        
cmpq $0, 64(%rdx,%rax,4) 

// CHECK: cmpq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x7c,0x82,0xc0,0x00]        
cmpq $0, -64(%rdx,%rax,4) 

// CHECK: cmpq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x83,0x7c,0x02,0x40,0x00]        
cmpq $0, 64(%rdx,%rax) 

// CHECK: cmpq $0, (%rdx) 
// CHECK: encoding: [0x48,0x83,0x3a,0x00]        
cmpq $0, (%rdx) 

// CHECK: cmpsb %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x65,0xa6]        
cmpsb %es:(%rdi), %gs:(%rsi) 

// CHECK: cmpsl %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x65,0xa7]        
cmpsl %es:(%rdi), %gs:(%rsi) 

// CHECK: cmpsq %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x65,0x48,0xa7]        
cmpsq %es:(%rdi), %gs:(%rsi) 

// CHECK: cmpsw %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0x65,0x66,0xa7]        
cmpsw %es:(%rdi), %gs:(%rsi) 

// CHECK: cmpw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpw $0, 485498096 

// CHECK: cmpw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0x83,0x7a,0x40,0x00]        
cmpw $0, 64(%rdx) 

// CHECK: cmpw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x7c,0x82,0x40,0x00]        
cmpw $0, 64(%rdx,%rax,4) 

// CHECK: cmpw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x7c,0x82,0xc0,0x00]        
cmpw $0, -64(%rdx,%rax,4) 

// CHECK: cmpw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x83,0x7c,0x02,0x40,0x00]        
cmpw $0, 64(%rdx,%rax) 

// CHECK: cmpw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0x83,0xfe,0x00]        
cmpw $0, %r14w 

// CHECK: cmpw $0, (%rdx) 
// CHECK: encoding: [0x66,0x83,0x3a,0x00]        
cmpw $0, (%rdx) 

// CHECK: cmpw 485498096, %r14w 
// CHECK: encoding: [0x66,0x44,0x3b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
cmpw 485498096, %r14w 

// CHECK: cmpw 64(%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x3b,0x72,0x40]        
cmpw 64(%rdx), %r14w 

// CHECK: cmpw 64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x3b,0x74,0x82,0x40]        
cmpw 64(%rdx,%rax,4), %r14w 

// CHECK: cmpw -64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x3b,0x74,0x82,0xc0]        
cmpw -64(%rdx,%rax,4), %r14w 

// CHECK: cmpw 64(%rdx,%rax), %r14w 
// CHECK: encoding: [0x66,0x44,0x3b,0x74,0x02,0x40]        
cmpw 64(%rdx,%rax), %r14w 

// CHECK: cmpw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x39,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
cmpw %r14w, 485498096 

// CHECK: cmpw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x39,0x72,0x40]        
cmpw %r14w, 64(%rdx) 

// CHECK: cmpw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x39,0x74,0x82,0x40]        
cmpw %r14w, 64(%rdx,%rax,4) 

// CHECK: cmpw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x39,0x74,0x82,0xc0]        
cmpw %r14w, -64(%rdx,%rax,4) 

// CHECK: cmpw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x39,0x74,0x02,0x40]        
cmpw %r14w, 64(%rdx,%rax) 

// CHECK: cmpw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x39,0xf6]        
cmpw %r14w, %r14w 

// CHECK: cmpw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x39,0x32]        
cmpw %r14w, (%rdx) 

// CHECK: cmpw (%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x3b,0x32]        
cmpw (%rdx), %r14w 

// CHECK: cwtd 
// CHECK: encoding: [0x66,0x99]          
cwtd 

// CHECK: decb 485498096 
// CHECK: encoding: [0xfe,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
decb 485498096 

// CHECK: decb 64(%rdx) 
// CHECK: encoding: [0xfe,0x4a,0x40]         
decb 64(%rdx) 

// CHECK: decb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xfe,0x4c,0x82,0x40]         
decb 64(%rdx,%rax,4) 

// CHECK: decb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xfe,0x4c,0x82,0xc0]         
decb -64(%rdx,%rax,4) 

// CHECK: decb 64(%rdx,%rax) 
// CHECK: encoding: [0xfe,0x4c,0x02,0x40]         
decb 64(%rdx,%rax) 

// CHECK: decb %r14b 
// CHECK: encoding: [0x41,0xfe,0xce]         
decb %r14b 

// CHECK: decb (%rdx) 
// CHECK: encoding: [0xfe,0x0a]         
decb (%rdx) 

// CHECK: decl 485498096 
// CHECK: encoding: [0xff,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
decl 485498096 

// CHECK: decl 64(%rdx) 
// CHECK: encoding: [0xff,0x4a,0x40]         
decl 64(%rdx) 

// CHECK: decl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xff,0x4c,0x82,0x40]         
decl 64(%rdx,%rax,4) 

// CHECK: decl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xff,0x4c,0x82,0xc0]         
decl -64(%rdx,%rax,4) 

// CHECK: decl 64(%rdx,%rax) 
// CHECK: encoding: [0xff,0x4c,0x02,0x40]         
decl 64(%rdx,%rax) 

// CHECK: decl %r13d 
// CHECK: encoding: [0x41,0xff,0xcd]         
decl %r13d 

// CHECK: decl (%rdx) 
// CHECK: encoding: [0xff,0x0a]         
decl (%rdx) 

// CHECK: decq 485498096 
// CHECK: encoding: [0x48,0xff,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
decq 485498096 

// CHECK: decq 64(%rdx) 
// CHECK: encoding: [0x48,0xff,0x4a,0x40]         
decq 64(%rdx) 

// CHECK: decq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xff,0x4c,0x82,0x40]         
decq 64(%rdx,%rax,4) 

// CHECK: decq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xff,0x4c,0x82,0xc0]         
decq -64(%rdx,%rax,4) 

// CHECK: decq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xff,0x4c,0x02,0x40]         
decq 64(%rdx,%rax) 

// CHECK: decq (%rdx) 
// CHECK: encoding: [0x48,0xff,0x0a]         
decq (%rdx) 

// CHECK: decw 485498096 
// CHECK: encoding: [0x66,0xff,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
decw 485498096 

// CHECK: decw 64(%rdx) 
// CHECK: encoding: [0x66,0xff,0x4a,0x40]         
decw 64(%rdx) 

// CHECK: decw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xff,0x4c,0x82,0x40]         
decw 64(%rdx,%rax,4) 

// CHECK: decw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xff,0x4c,0x82,0xc0]         
decw -64(%rdx,%rax,4) 

// CHECK: decw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xff,0x4c,0x02,0x40]         
decw 64(%rdx,%rax) 

// CHECK: decw %r14w 
// CHECK: encoding: [0x66,0x41,0xff,0xce]         
decw %r14w 

// CHECK: decw (%rdx) 
// CHECK: encoding: [0x66,0xff,0x0a]         
decw (%rdx) 

// CHECK: divb 485498096 
// CHECK: encoding: [0xf6,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
divb 485498096 

// CHECK: divb 64(%rdx) 
// CHECK: encoding: [0xf6,0x72,0x40]         
divb 64(%rdx) 

// CHECK: divb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x74,0x82,0x40]         
divb 64(%rdx,%rax,4) 

// CHECK: divb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x74,0x82,0xc0]         
divb -64(%rdx,%rax,4) 

// CHECK: divb 64(%rdx,%rax) 
// CHECK: encoding: [0xf6,0x74,0x02,0x40]         
divb 64(%rdx,%rax) 

// CHECK: divb %r14b 
// CHECK: encoding: [0x41,0xf6,0xf6]         
divb %r14b 

// CHECK: divb (%rdx) 
// CHECK: encoding: [0xf6,0x32]         
divb (%rdx) 

// CHECK: divl 485498096 
// CHECK: encoding: [0xf7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
divl 485498096 

// CHECK: divl 64(%rdx) 
// CHECK: encoding: [0xf7,0x72,0x40]         
divl 64(%rdx) 

// CHECK: divl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x74,0x82,0x40]         
divl 64(%rdx,%rax,4) 

// CHECK: divl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x74,0x82,0xc0]         
divl -64(%rdx,%rax,4) 

// CHECK: divl 64(%rdx,%rax) 
// CHECK: encoding: [0xf7,0x74,0x02,0x40]         
divl 64(%rdx,%rax) 

// CHECK: divl %r13d 
// CHECK: encoding: [0x41,0xf7,0xf5]         
divl %r13d 

// CHECK: divl (%rdx) 
// CHECK: encoding: [0xf7,0x32]         
divl (%rdx) 

// CHECK: divq 485498096 
// CHECK: encoding: [0x48,0xf7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
divq 485498096 

// CHECK: divq 64(%rdx) 
// CHECK: encoding: [0x48,0xf7,0x72,0x40]         
divq 64(%rdx) 

// CHECK: divq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x74,0x82,0x40]         
divq 64(%rdx,%rax,4) 

// CHECK: divq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x74,0x82,0xc0]         
divq -64(%rdx,%rax,4) 

// CHECK: divq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xf7,0x74,0x02,0x40]         
divq 64(%rdx,%rax) 

// CHECK: divq (%rdx) 
// CHECK: encoding: [0x48,0xf7,0x32]         
divq (%rdx) 

// CHECK: divw 485498096 
// CHECK: encoding: [0x66,0xf7,0x34,0x25,0xf0,0x1c,0xf0,0x1c]         
divw 485498096 

// CHECK: divw 64(%rdx) 
// CHECK: encoding: [0x66,0xf7,0x72,0x40]         
divw 64(%rdx) 

// CHECK: divw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x74,0x82,0x40]         
divw 64(%rdx,%rax,4) 

// CHECK: divw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x74,0x82,0xc0]         
divw -64(%rdx,%rax,4) 

// CHECK: divw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xf7,0x74,0x02,0x40]         
divw 64(%rdx,%rax) 

// CHECK: divw %r14w 
// CHECK: encoding: [0x66,0x41,0xf7,0xf6]         
divw %r14w 

// CHECK: divw (%rdx) 
// CHECK: encoding: [0x66,0xf7,0x32]         
divw (%rdx) 

// CHECK: hlt 
// CHECK: encoding: [0xf4]          
hlt 

// CHECK: idivb 485498096 
// CHECK: encoding: [0xf6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
idivb 485498096 

// CHECK: idivb 64(%rdx) 
// CHECK: encoding: [0xf6,0x7a,0x40]         
idivb 64(%rdx) 

// CHECK: idivb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x7c,0x82,0x40]         
idivb 64(%rdx,%rax,4) 

// CHECK: idivb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x7c,0x82,0xc0]         
idivb -64(%rdx,%rax,4) 

// CHECK: idivb 64(%rdx,%rax) 
// CHECK: encoding: [0xf6,0x7c,0x02,0x40]         
idivb 64(%rdx,%rax) 

// CHECK: idivb %r14b 
// CHECK: encoding: [0x41,0xf6,0xfe]         
idivb %r14b 

// CHECK: idivb (%rdx) 
// CHECK: encoding: [0xf6,0x3a]         
idivb (%rdx) 

// CHECK: idivl 485498096 
// CHECK: encoding: [0xf7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
idivl 485498096 

// CHECK: idivl 64(%rdx) 
// CHECK: encoding: [0xf7,0x7a,0x40]         
idivl 64(%rdx) 

// CHECK: idivl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x7c,0x82,0x40]         
idivl 64(%rdx,%rax,4) 

// CHECK: idivl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x7c,0x82,0xc0]         
idivl -64(%rdx,%rax,4) 

// CHECK: idivl 64(%rdx,%rax) 
// CHECK: encoding: [0xf7,0x7c,0x02,0x40]         
idivl 64(%rdx,%rax) 

// CHECK: idivl %r13d 
// CHECK: encoding: [0x41,0xf7,0xfd]         
idivl %r13d 

// CHECK: idivl (%rdx) 
// CHECK: encoding: [0xf7,0x3a]         
idivl (%rdx) 

// CHECK: idivq 485498096 
// CHECK: encoding: [0x48,0xf7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
idivq 485498096 

// CHECK: idivq 64(%rdx) 
// CHECK: encoding: [0x48,0xf7,0x7a,0x40]         
idivq 64(%rdx) 

// CHECK: idivq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x7c,0x82,0x40]         
idivq 64(%rdx,%rax,4) 

// CHECK: idivq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x7c,0x82,0xc0]         
idivq -64(%rdx,%rax,4) 

// CHECK: idivq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xf7,0x7c,0x02,0x40]         
idivq 64(%rdx,%rax) 

// CHECK: idivq (%rdx) 
// CHECK: encoding: [0x48,0xf7,0x3a]         
idivq (%rdx) 

// CHECK: idivw 485498096 
// CHECK: encoding: [0x66,0xf7,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
idivw 485498096 

// CHECK: idivw 64(%rdx) 
// CHECK: encoding: [0x66,0xf7,0x7a,0x40]         
idivw 64(%rdx) 

// CHECK: idivw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x7c,0x82,0x40]         
idivw 64(%rdx,%rax,4) 

// CHECK: idivw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x7c,0x82,0xc0]         
idivw -64(%rdx,%rax,4) 

// CHECK: idivw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xf7,0x7c,0x02,0x40]         
idivw 64(%rdx,%rax) 

// CHECK: idivw %r14w 
// CHECK: encoding: [0x66,0x41,0xf7,0xfe]         
idivw %r14w 

// CHECK: idivw (%rdx) 
// CHECK: encoding: [0x66,0xf7,0x3a]         
idivw (%rdx) 

// CHECK: imulb 485498096 
// CHECK: encoding: [0xf6,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
imulb 485498096 

// CHECK: imulb 64(%rdx) 
// CHECK: encoding: [0xf6,0x6a,0x40]         
imulb 64(%rdx) 

// CHECK: imulb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x6c,0x82,0x40]         
imulb 64(%rdx,%rax,4) 

// CHECK: imulb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x6c,0x82,0xc0]         
imulb -64(%rdx,%rax,4) 

// CHECK: imulb 64(%rdx,%rax) 
// CHECK: encoding: [0xf6,0x6c,0x02,0x40]         
imulb 64(%rdx,%rax) 

// CHECK: imulb %r14b 
// CHECK: encoding: [0x41,0xf6,0xee]         
imulb %r14b 

// CHECK: imulb (%rdx) 
// CHECK: encoding: [0xf6,0x2a]         
imulb (%rdx) 

// CHECK: imull 485498096 
// CHECK: encoding: [0xf7,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
imull 485498096 

// CHECK: imull 64(%rdx) 
// CHECK: encoding: [0xf7,0x6a,0x40]         
imull 64(%rdx) 

// CHECK: imull 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x6c,0x82,0x40]         
imull 64(%rdx,%rax,4) 

// CHECK: imull -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x6c,0x82,0xc0]         
imull -64(%rdx,%rax,4) 

// CHECK: imull 64(%rdx,%rax) 
// CHECK: encoding: [0xf7,0x6c,0x02,0x40]         
imull 64(%rdx,%rax) 

// CHECK: imull %r13d 
// CHECK: encoding: [0x41,0xf7,0xed]         
imull %r13d 

// CHECK: imull %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0xaf,0xed]        
imull %r13d, %r13d 

// CHECK: imull (%rdx) 
// CHECK: encoding: [0xf7,0x2a]         
imull (%rdx) 

// CHECK: imulq 485498096 
// CHECK: encoding: [0x48,0xf7,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
imulq 485498096 

// CHECK: imulq 64(%rdx) 
// CHECK: encoding: [0x48,0xf7,0x6a,0x40]         
imulq 64(%rdx) 

// CHECK: imulq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x6c,0x82,0x40]         
imulq 64(%rdx,%rax,4) 

// CHECK: imulq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x6c,0x82,0xc0]         
imulq -64(%rdx,%rax,4) 

// CHECK: imulq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xf7,0x6c,0x02,0x40]         
imulq 64(%rdx,%rax) 

// CHECK: imulq (%rdx) 
// CHECK: encoding: [0x48,0xf7,0x2a]         
imulq (%rdx) 

// CHECK: imulw 485498096 
// CHECK: encoding: [0x66,0xf7,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
imulw 485498096 

// CHECK: imulw 64(%rdx) 
// CHECK: encoding: [0x66,0xf7,0x6a,0x40]         
imulw 64(%rdx) 

// CHECK: imulw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x6c,0x82,0x40]         
imulw 64(%rdx,%rax,4) 

// CHECK: imulw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x6c,0x82,0xc0]         
imulw -64(%rdx,%rax,4) 

// CHECK: imulw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xf7,0x6c,0x02,0x40]         
imulw 64(%rdx,%rax) 

// CHECK: imulw %r14w 
// CHECK: encoding: [0x66,0x41,0xf7,0xee]         
imulw %r14w 

// CHECK: imulw (%rdx) 
// CHECK: encoding: [0x66,0xf7,0x2a]         
imulw (%rdx) 

// CHECK: inb $0, %al 
// CHECK: encoding: [0xe4,0x00]        
inb $0, %al 

// CHECK: inb %dx, %al 
// CHECK: encoding: [0xec]        
inb %dx, %al 

// CHECK: incb 485498096 
// CHECK: encoding: [0xfe,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
incb 485498096 

// CHECK: incb 64(%rdx) 
// CHECK: encoding: [0xfe,0x42,0x40]         
incb 64(%rdx) 

// CHECK: incb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xfe,0x44,0x82,0x40]         
incb 64(%rdx,%rax,4) 

// CHECK: incb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xfe,0x44,0x82,0xc0]         
incb -64(%rdx,%rax,4) 

// CHECK: incb 64(%rdx,%rax) 
// CHECK: encoding: [0xfe,0x44,0x02,0x40]         
incb 64(%rdx,%rax) 

// CHECK: incb %r14b 
// CHECK: encoding: [0x41,0xfe,0xc6]         
incb %r14b 

// CHECK: incb (%rdx) 
// CHECK: encoding: [0xfe,0x02]         
incb (%rdx) 

// CHECK: incl 485498096 
// CHECK: encoding: [0xff,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
incl 485498096 

// CHECK: incl 64(%rdx) 
// CHECK: encoding: [0xff,0x42,0x40]         
incl 64(%rdx) 

// CHECK: incl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xff,0x44,0x82,0x40]         
incl 64(%rdx,%rax,4) 

// CHECK: incl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xff,0x44,0x82,0xc0]         
incl -64(%rdx,%rax,4) 

// CHECK: incl 64(%rdx,%rax) 
// CHECK: encoding: [0xff,0x44,0x02,0x40]         
incl 64(%rdx,%rax) 

// CHECK: incl %r13d 
// CHECK: encoding: [0x41,0xff,0xc5]         
incl %r13d 

// CHECK: incl (%rdx) 
// CHECK: encoding: [0xff,0x02]         
incl (%rdx) 

// CHECK: incq 485498096 
// CHECK: encoding: [0x48,0xff,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
incq 485498096 

// CHECK: incq 64(%rdx) 
// CHECK: encoding: [0x48,0xff,0x42,0x40]         
incq 64(%rdx) 

// CHECK: incq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xff,0x44,0x82,0x40]         
incq 64(%rdx,%rax,4) 

// CHECK: incq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xff,0x44,0x82,0xc0]         
incq -64(%rdx,%rax,4) 

// CHECK: incq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xff,0x44,0x02,0x40]         
incq 64(%rdx,%rax) 

// CHECK: incq (%rdx) 
// CHECK: encoding: [0x48,0xff,0x02]         
incq (%rdx) 

// CHECK: incw 485498096 
// CHECK: encoding: [0x66,0xff,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
incw 485498096 

// CHECK: incw 64(%rdx) 
// CHECK: encoding: [0x66,0xff,0x42,0x40]         
incw 64(%rdx) 

// CHECK: incw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xff,0x44,0x82,0x40]         
incw 64(%rdx,%rax,4) 

// CHECK: incw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xff,0x44,0x82,0xc0]         
incw -64(%rdx,%rax,4) 

// CHECK: incw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xff,0x44,0x02,0x40]         
incw 64(%rdx,%rax) 

// CHECK: incw %r14w 
// CHECK: encoding: [0x66,0x41,0xff,0xc6]         
incw %r14w 

// CHECK: incw (%rdx) 
// CHECK: encoding: [0x66,0xff,0x02]         
incw (%rdx) 

// CHECK: inl $0, %eax 
// CHECK: encoding: [0xe5,0x00]        
inl $0, %eax 

// CHECK: inl %dx, %eax 
// CHECK: encoding: [0xed]        
inl %dx, %eax 

// CHECK: int $0 
// CHECK: encoding: [0xcd,0x00]         
int $0 

// CHECK: int3 
// CHECK: encoding: [0xcc]          
int3 

// CHECK: iretl 
// CHECK: encoding: [0xcf]          
iretl 

// CHECK: iretq 
// CHECK: encoding: [0x48,0xcf]          
iretq 

// CHECK: iretw 
// CHECK: encoding: [0x66,0xcf]          
iretw 

// CHECK: ja 64 
// CHECK: encoding: [0x77,A]         
ja 64 

// CHECK: jae 64 
// CHECK: encoding: [0x73,A]         
jae 64 

// CHECK: jb 64 
// CHECK: encoding: [0x72,A]         
jb 64 

// CHECK: jbe 64 
// CHECK: encoding: [0x76,A]         
jbe 64 

// CHECK: je 64 
// CHECK: encoding: [0x74,A]         
je 64 

// CHECK: jg 64 
// CHECK: encoding: [0x7f,A]         
jg 64 

// CHECK: jge 64 
// CHECK: encoding: [0x7d,A]         
jge 64 

// CHECK: jl 64 
// CHECK: encoding: [0x7c,A]         
jl 64 

// CHECK: jle 64 
// CHECK: encoding: [0x7e,A]         
jle 64 

// CHECK: jmp 64 
// CHECK: encoding: [0xeb,A]         
jmp 64 

// CHECK: jne 64 
// CHECK: encoding: [0x75,A]         
jne 64 

// CHECK: jno 64 
// CHECK: encoding: [0x71,A]         
jno 64 

// CHECK: jnp 64 
// CHECK: encoding: [0x7b,A]         
jnp 64 

// CHECK: jns 64 
// CHECK: encoding: [0x79,A]         
jns 64 

// CHECK: jo 64 
// CHECK: encoding: [0x70,A]         
jo 64 

// CHECK: jp 64 
// CHECK: encoding: [0x7a,A]         
jp 64 

// CHECK: js 64 
// CHECK: encoding: [0x78,A]         
js 64 

// CHECK: leal 485498096, %r13d 
// CHECK: encoding: [0x44,0x8d,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
leal 485498096, %r13d 

// CHECK: leal 64(%rdx), %r13d 
// CHECK: encoding: [0x44,0x8d,0x6a,0x40]        
leal 64(%rdx), %r13d 

// CHECK: leal 64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x8d,0x6c,0x82,0x40]        
leal 64(%rdx,%rax,4), %r13d 

// CHECK: leal -64(%rdx,%rax,4), %r13d 
// CHECK: encoding: [0x44,0x8d,0x6c,0x82,0xc0]        
leal -64(%rdx,%rax,4), %r13d 

// CHECK: leal 64(%rdx,%rax), %r13d 
// CHECK: encoding: [0x44,0x8d,0x6c,0x02,0x40]        
leal 64(%rdx,%rax), %r13d 

// CHECK: leal (%rdx), %r13d 
// CHECK: encoding: [0x44,0x8d,0x2a]        
leal (%rdx), %r13d 

// CHECK: lodsb %gs:(%rsi), %al 
// CHECK: encoding: [0x65,0xac]        
lodsb %gs:(%rsi), %al 

// CHECK: lodsw %gs:(%rsi), %ax 
// CHECK: encoding: [0x65,0x66,0xad]        
lodsw %gs:(%rsi), %ax 

// CHECK: loop 64 
// CHECK: encoding: [0xe2,A]         
loop 64 

// CHECK: loope 64 
// CHECK: encoding: [0xe1,A]         
loope 64 

// CHECK: loopne 64 
// CHECK: encoding: [0xe0,A]         
loopne 64 

// CHECK: lretl $0 
// CHECK: encoding: [0xca,0x00,0x00]         
lretl $0 

// CHECK: lretl 
// CHECK: encoding: [0xcb]          
lretl 

// CHECK: movb $0, 485498096 
// CHECK: encoding: [0xc6,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
movb $0, 485498096 

// CHECK: movb $0, 64(%rdx) 
// CHECK: encoding: [0xc6,0x42,0x40,0x00]        
movb $0, 64(%rdx) 

// CHECK: movb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc6,0x44,0x82,0x40,0x00]        
movb $0, 64(%rdx,%rax,4) 

// CHECK: movb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc6,0x44,0x82,0xc0,0x00]        
movb $0, -64(%rdx,%rax,4) 

// CHECK: movb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc6,0x44,0x02,0x40,0x00]        
movb $0, 64(%rdx,%rax) 

// CHECK: movb $0, %r14b 
// CHECK: encoding: [0x41,0xb6,0x00]        
movb $0, %r14b 

// CHECK: movb $0, (%rdx) 
// CHECK: encoding: [0xc6,0x02,0x00]        
movb $0, (%rdx) 

// CHECK: movb 485498096, %r14b 
// CHECK: encoding: [0x44,0x8a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
movb 485498096, %r14b 

// CHECK: movb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x8a,0x72,0x40]        
movb 64(%rdx), %r14b 

// CHECK: movb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x8a,0x74,0x82,0x40]        
movb 64(%rdx,%rax,4), %r14b 

// CHECK: movb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x8a,0x74,0x82,0xc0]        
movb -64(%rdx,%rax,4), %r14b 

// CHECK: movb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x8a,0x74,0x02,0x40]        
movb 64(%rdx,%rax), %r14b 

// CHECK: movb %r14b, 485498096 
// CHECK: encoding: [0x44,0x88,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
movb %r14b, 485498096 

// CHECK: movb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x88,0x72,0x40]        
movb %r14b, 64(%rdx) 

// CHECK: movb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x88,0x74,0x82,0x40]        
movb %r14b, 64(%rdx,%rax,4) 

// CHECK: movb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x88,0x74,0x82,0xc0]        
movb %r14b, -64(%rdx,%rax,4) 

// CHECK: movb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x88,0x74,0x02,0x40]        
movb %r14b, 64(%rdx,%rax) 

// CHECK: movb %r14b, %r14b 
// CHECK: encoding: [0x45,0x88,0xf6]        
movb %r14b, %r14b 

// CHECK: movb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x88,0x32]        
movb %r14b, (%rdx) 

// CHECK: movb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x8a,0x32]        
movb (%rdx), %r14b 

// CHECK: movl $0, 485498096 
// CHECK: encoding: [0xc7,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
movl $0, 485498096 

// CHECK: movl $0, 64(%rdx) 
// CHECK: encoding: [0xc7,0x42,0x40,0x00,0x00,0x00,0x00]        
movl $0, 64(%rdx) 

// CHECK: movl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc7,0x44,0x82,0x40,0x00,0x00,0x00,0x00]        
movl $0, 64(%rdx,%rax,4) 

// CHECK: movl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc7,0x44,0x82,0xc0,0x00,0x00,0x00,0x00]        
movl $0, -64(%rdx,%rax,4) 

// CHECK: movl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xc7,0x44,0x02,0x40,0x00,0x00,0x00,0x00]        
movl $0, 64(%rdx,%rax) 

// CHECK: movl $0, %r13d 
// CHECK: encoding: [0x41,0xbd,0x00,0x00,0x00,0x00]        
movl $0, %r13d 

// CHECK: movl $0, (%rdx) 
// CHECK: encoding: [0xc7,0x02,0x00,0x00,0x00,0x00]        
movl $0, (%rdx) 

// CHECK: movl %es, %r13d 
// CHECK: encoding: [0x41,0x8c,0xc5]        
movl %es, %r13d 

// CHECK: movl %r11d, %es 
// CHECK: encoding: [0x41,0x8e,0xc3]        
movl %r11d, %es 

// CHECK: movl %r13d, %r13d 
// CHECK: encoding: [0x45,0x89,0xed]        
movl %r13d, %r13d 

// CHECK: movq $0, 485498096 
// CHECK: encoding: [0x48,0xc7,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
movq $0, 485498096 

// CHECK: movq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xc7,0x42,0x40,0x00,0x00,0x00,0x00]        
movq $0, 64(%rdx) 

// CHECK: movq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc7,0x44,0x82,0x40,0x00,0x00,0x00,0x00]        
movq $0, 64(%rdx,%rax,4) 

// CHECK: movq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xc7,0x44,0x82,0xc0,0x00,0x00,0x00,0x00]        
movq $0, -64(%rdx,%rax,4) 

// CHECK: movq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xc7,0x44,0x02,0x40,0x00,0x00,0x00,0x00]        
movq $0, 64(%rdx,%rax) 

// CHECK: movq $0, (%rdx) 
// CHECK: encoding: [0x48,0xc7,0x02,0x00,0x00,0x00,0x00]        
movq $0, (%rdx) 

// CHECK: movsb %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x65,0xa4]        
movsb %gs:(%rsi), %es:(%rdi) 

// CHECK: movsl %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x65,0xa5]        
movsl %gs:(%rsi), %es:(%rdi) 

// CHECK: movsq %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x65,0x48,0xa5]        
movsq %gs:(%rsi), %es:(%rdi) 

// CHECK: movsw %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0x65,0x66,0xa5]        
movsw %gs:(%rsi), %es:(%rdi) 

// CHECK: movw $0, 485498096 
// CHECK: encoding: [0x66,0xc7,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
movw $0, 485498096 

// CHECK: movw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xc7,0x42,0x40,0x00,0x00]        
movw $0, 64(%rdx) 

// CHECK: movw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc7,0x44,0x82,0x40,0x00,0x00]        
movw $0, 64(%rdx,%rax,4) 

// CHECK: movw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xc7,0x44,0x82,0xc0,0x00,0x00]        
movw $0, -64(%rdx,%rax,4) 

// CHECK: movw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xc7,0x44,0x02,0x40,0x00,0x00]        
movw $0, 64(%rdx,%rax) 

// CHECK: movw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xbe,0x00,0x00]        
movw $0, %r14w 

// CHECK: movw $0, (%rdx) 
// CHECK: encoding: [0x66,0xc7,0x02,0x00,0x00]        
movw $0, (%rdx) 

// CHECK: movw 485498096, %es 
// CHECK: encoding: [0x8e,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
movw 485498096, %es 

// CHECK: movw 485498096, %r14w 
// CHECK: encoding: [0x66,0x44,0x8b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
movw 485498096, %r14w 

// CHECK: movw 64(%rdx), %es 
// CHECK: encoding: [0x8e,0x42,0x40]        
movw 64(%rdx), %es 

// CHECK: movw 64(%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x8b,0x72,0x40]        
movw 64(%rdx), %r14w 

// CHECK: movw 64(%rdx,%rax,4), %es 
// CHECK: encoding: [0x8e,0x44,0x82,0x40]        
movw 64(%rdx,%rax,4), %es 

// CHECK: movw -64(%rdx,%rax,4), %es 
// CHECK: encoding: [0x8e,0x44,0x82,0xc0]        
movw -64(%rdx,%rax,4), %es 

// CHECK: movw 64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x8b,0x74,0x82,0x40]        
movw 64(%rdx,%rax,4), %r14w 

// CHECK: movw -64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x8b,0x74,0x82,0xc0]        
movw -64(%rdx,%rax,4), %r14w 

// CHECK: movw 64(%rdx,%rax), %es 
// CHECK: encoding: [0x8e,0x44,0x02,0x40]        
movw 64(%rdx,%rax), %es 

// CHECK: movw 64(%rdx,%rax), %r14w 
// CHECK: encoding: [0x66,0x44,0x8b,0x74,0x02,0x40]        
movw 64(%rdx,%rax), %r14w 

// CHECK: movw %es, 485498096 
// CHECK: encoding: [0x8c,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
movw %es, 485498096 

// CHECK: movw %es, 64(%rdx) 
// CHECK: encoding: [0x8c,0x42,0x40]        
movw %es, 64(%rdx) 

// CHECK: movw %es, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x8c,0x44,0x82,0x40]        
movw %es, 64(%rdx,%rax,4) 

// CHECK: movw %es, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x8c,0x44,0x82,0xc0]        
movw %es, -64(%rdx,%rax,4) 

// CHECK: movw %es, 64(%rdx,%rax) 
// CHECK: encoding: [0x8c,0x44,0x02,0x40]        
movw %es, 64(%rdx,%rax) 

// CHECK: movw %es, (%rdx) 
// CHECK: encoding: [0x8c,0x02]        
movw %es, (%rdx) 

// CHECK: movw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x89,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
movw %r14w, 485498096 

// CHECK: movw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x89,0x72,0x40]        
movw %r14w, 64(%rdx) 

// CHECK: movw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x89,0x74,0x82,0x40]        
movw %r14w, 64(%rdx,%rax,4) 

// CHECK: movw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x89,0x74,0x82,0xc0]        
movw %r14w, -64(%rdx,%rax,4) 

// CHECK: movw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x89,0x74,0x02,0x40]        
movw %r14w, 64(%rdx,%rax) 

// CHECK: movw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x89,0xf6]        
movw %r14w, %r14w 

// CHECK: movw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x89,0x32]        
movw %r14w, (%rdx) 

// CHECK: movw (%rdx), %es 
// CHECK: encoding: [0x8e,0x02]        
movw (%rdx), %es 

// CHECK: movw (%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x8b,0x32]        
movw (%rdx), %r14w 

// CHECK: mulb 485498096 
// CHECK: encoding: [0xf6,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
mulb 485498096 

// CHECK: mulb 64(%rdx) 
// CHECK: encoding: [0xf6,0x62,0x40]         
mulb 64(%rdx) 

// CHECK: mulb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x64,0x82,0x40]         
mulb 64(%rdx,%rax,4) 

// CHECK: mulb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x64,0x82,0xc0]         
mulb -64(%rdx,%rax,4) 

// CHECK: mulb 64(%rdx,%rax) 
// CHECK: encoding: [0xf6,0x64,0x02,0x40]         
mulb 64(%rdx,%rax) 

// CHECK: mulb %r14b 
// CHECK: encoding: [0x41,0xf6,0xe6]         
mulb %r14b 

// CHECK: mulb (%rdx) 
// CHECK: encoding: [0xf6,0x22]         
mulb (%rdx) 

// CHECK: mull 485498096 
// CHECK: encoding: [0xf7,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
mull 485498096 

// CHECK: mull 64(%rdx) 
// CHECK: encoding: [0xf7,0x62,0x40]         
mull 64(%rdx) 

// CHECK: mull 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x64,0x82,0x40]         
mull 64(%rdx,%rax,4) 

// CHECK: mull -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x64,0x82,0xc0]         
mull -64(%rdx,%rax,4) 

// CHECK: mull 64(%rdx,%rax) 
// CHECK: encoding: [0xf7,0x64,0x02,0x40]         
mull 64(%rdx,%rax) 

// CHECK: mull %r13d 
// CHECK: encoding: [0x41,0xf7,0xe5]         
mull %r13d 

// CHECK: mull (%rdx) 
// CHECK: encoding: [0xf7,0x22]         
mull (%rdx) 

// CHECK: mulq 485498096 
// CHECK: encoding: [0x48,0xf7,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
mulq 485498096 

// CHECK: mulq 64(%rdx) 
// CHECK: encoding: [0x48,0xf7,0x62,0x40]         
mulq 64(%rdx) 

// CHECK: mulq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x64,0x82,0x40]         
mulq 64(%rdx,%rax,4) 

// CHECK: mulq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x64,0x82,0xc0]         
mulq -64(%rdx,%rax,4) 

// CHECK: mulq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xf7,0x64,0x02,0x40]         
mulq 64(%rdx,%rax) 

// CHECK: mulq (%rdx) 
// CHECK: encoding: [0x48,0xf7,0x22]         
mulq (%rdx) 

// CHECK: mulw 485498096 
// CHECK: encoding: [0x66,0xf7,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
mulw 485498096 

// CHECK: mulw 64(%rdx) 
// CHECK: encoding: [0x66,0xf7,0x62,0x40]         
mulw 64(%rdx) 

// CHECK: mulw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x64,0x82,0x40]         
mulw 64(%rdx,%rax,4) 

// CHECK: mulw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x64,0x82,0xc0]         
mulw -64(%rdx,%rax,4) 

// CHECK: mulw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xf7,0x64,0x02,0x40]         
mulw 64(%rdx,%rax) 

// CHECK: mulw %r14w 
// CHECK: encoding: [0x66,0x41,0xf7,0xe6]         
mulw %r14w 

// CHECK: mulw (%rdx) 
// CHECK: encoding: [0x66,0xf7,0x22]         
mulw (%rdx) 

// CHECK: negb 485498096 
// CHECK: encoding: [0xf6,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
negb 485498096 

// CHECK: negb 64(%rdx) 
// CHECK: encoding: [0xf6,0x5a,0x40]         
negb 64(%rdx) 

// CHECK: negb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x5c,0x82,0x40]         
negb 64(%rdx,%rax,4) 

// CHECK: negb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x5c,0x82,0xc0]         
negb -64(%rdx,%rax,4) 

// CHECK: negb 64(%rdx,%rax) 
// CHECK: encoding: [0xf6,0x5c,0x02,0x40]         
negb 64(%rdx,%rax) 

// CHECK: negb %r14b 
// CHECK: encoding: [0x41,0xf6,0xde]         
negb %r14b 

// CHECK: negb (%rdx) 
// CHECK: encoding: [0xf6,0x1a]         
negb (%rdx) 

// CHECK: negl 485498096 
// CHECK: encoding: [0xf7,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
negl 485498096 

// CHECK: negl 64(%rdx) 
// CHECK: encoding: [0xf7,0x5a,0x40]         
negl 64(%rdx) 

// CHECK: negl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x5c,0x82,0x40]         
negl 64(%rdx,%rax,4) 

// CHECK: negl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x5c,0x82,0xc0]         
negl -64(%rdx,%rax,4) 

// CHECK: negl 64(%rdx,%rax) 
// CHECK: encoding: [0xf7,0x5c,0x02,0x40]         
negl 64(%rdx,%rax) 

// CHECK: negl %r13d 
// CHECK: encoding: [0x41,0xf7,0xdd]         
negl %r13d 

// CHECK: negl (%rdx) 
// CHECK: encoding: [0xf7,0x1a]         
negl (%rdx) 

// CHECK: negq 485498096 
// CHECK: encoding: [0x48,0xf7,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
negq 485498096 

// CHECK: negq 64(%rdx) 
// CHECK: encoding: [0x48,0xf7,0x5a,0x40]         
negq 64(%rdx) 

// CHECK: negq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x5c,0x82,0x40]         
negq 64(%rdx,%rax,4) 

// CHECK: negq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x5c,0x82,0xc0]         
negq -64(%rdx,%rax,4) 

// CHECK: negq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xf7,0x5c,0x02,0x40]         
negq 64(%rdx,%rax) 

// CHECK: negq (%rdx) 
// CHECK: encoding: [0x48,0xf7,0x1a]         
negq (%rdx) 

// CHECK: negw 485498096 
// CHECK: encoding: [0x66,0xf7,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
negw 485498096 

// CHECK: negw 64(%rdx) 
// CHECK: encoding: [0x66,0xf7,0x5a,0x40]         
negw 64(%rdx) 

// CHECK: negw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x5c,0x82,0x40]         
negw 64(%rdx,%rax,4) 

// CHECK: negw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x5c,0x82,0xc0]         
negw -64(%rdx,%rax,4) 

// CHECK: negw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xf7,0x5c,0x02,0x40]         
negw 64(%rdx,%rax) 

// CHECK: negw %r14w 
// CHECK: encoding: [0x66,0x41,0xf7,0xde]         
negw %r14w 

// CHECK: negw (%rdx) 
// CHECK: encoding: [0x66,0xf7,0x1a]         
negw (%rdx) 

// CHECK: nop 
// CHECK: encoding: [0x90]          
nop 

// CHECK: notb 485498096 
// CHECK: encoding: [0xf6,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
notb 485498096 

// CHECK: notb 64(%rdx) 
// CHECK: encoding: [0xf6,0x52,0x40]         
notb 64(%rdx) 

// CHECK: notb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x54,0x82,0x40]         
notb 64(%rdx,%rax,4) 

// CHECK: notb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x54,0x82,0xc0]         
notb -64(%rdx,%rax,4) 

// CHECK: notb 64(%rdx,%rax) 
// CHECK: encoding: [0xf6,0x54,0x02,0x40]         
notb 64(%rdx,%rax) 

// CHECK: notb %r14b 
// CHECK: encoding: [0x41,0xf6,0xd6]         
notb %r14b 

// CHECK: notb (%rdx) 
// CHECK: encoding: [0xf6,0x12]         
notb (%rdx) 

// CHECK: notl 485498096 
// CHECK: encoding: [0xf7,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
notl 485498096 

// CHECK: notl 64(%rdx) 
// CHECK: encoding: [0xf7,0x52,0x40]         
notl 64(%rdx) 

// CHECK: notl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x54,0x82,0x40]         
notl 64(%rdx,%rax,4) 

// CHECK: notl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x54,0x82,0xc0]         
notl -64(%rdx,%rax,4) 

// CHECK: notl 64(%rdx,%rax) 
// CHECK: encoding: [0xf7,0x54,0x02,0x40]         
notl 64(%rdx,%rax) 

// CHECK: notl %r13d 
// CHECK: encoding: [0x41,0xf7,0xd5]         
notl %r13d 

// CHECK: notl (%rdx) 
// CHECK: encoding: [0xf7,0x12]         
notl (%rdx) 

// CHECK: notq 485498096 
// CHECK: encoding: [0x48,0xf7,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
notq 485498096 

// CHECK: notq 64(%rdx) 
// CHECK: encoding: [0x48,0xf7,0x52,0x40]         
notq 64(%rdx) 

// CHECK: notq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x54,0x82,0x40]         
notq 64(%rdx,%rax,4) 

// CHECK: notq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x54,0x82,0xc0]         
notq -64(%rdx,%rax,4) 

// CHECK: notq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xf7,0x54,0x02,0x40]         
notq 64(%rdx,%rax) 

// CHECK: notq (%rdx) 
// CHECK: encoding: [0x48,0xf7,0x12]         
notq (%rdx) 

// CHECK: notw 485498096 
// CHECK: encoding: [0x66,0xf7,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
notw 485498096 

// CHECK: notw 64(%rdx) 
// CHECK: encoding: [0x66,0xf7,0x52,0x40]         
notw 64(%rdx) 

// CHECK: notw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x54,0x82,0x40]         
notw 64(%rdx,%rax,4) 

// CHECK: notw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x54,0x82,0xc0]         
notw -64(%rdx,%rax,4) 

// CHECK: notw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xf7,0x54,0x02,0x40]         
notw 64(%rdx,%rax) 

// CHECK: notw %r14w 
// CHECK: encoding: [0x66,0x41,0xf7,0xd6]         
notw %r14w 

// CHECK: notw (%rdx) 
// CHECK: encoding: [0x66,0xf7,0x12]         
notw (%rdx) 

// CHECK: orb $0, 485498096 
// CHECK: encoding: [0x80,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
orb $0, 485498096 

// CHECK: orb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x4a,0x40,0x00]        
orb $0, 64(%rdx) 

// CHECK: orb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x4c,0x82,0x40,0x00]        
orb $0, 64(%rdx,%rax,4) 

// CHECK: orb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x4c,0x82,0xc0,0x00]        
orb $0, -64(%rdx,%rax,4) 

// CHECK: orb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x4c,0x02,0x40,0x00]        
orb $0, 64(%rdx,%rax) 

// CHECK: orb $0, %al 
// CHECK: encoding: [0x0c,0x00]        
orb $0, %al 

// CHECK: orb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xce,0x00]        
orb $0, %r14b 

// CHECK: orb $0, (%rdx) 
// CHECK: encoding: [0x80,0x0a,0x00]        
orb $0, (%rdx) 

// CHECK: orb 485498096, %r14b 
// CHECK: encoding: [0x44,0x0a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
orb 485498096, %r14b 

// CHECK: orb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x0a,0x72,0x40]        
orb 64(%rdx), %r14b 

// CHECK: orb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x0a,0x74,0x82,0x40]        
orb 64(%rdx,%rax,4), %r14b 

// CHECK: orb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x0a,0x74,0x82,0xc0]        
orb -64(%rdx,%rax,4), %r14b 

// CHECK: orb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x0a,0x74,0x02,0x40]        
orb 64(%rdx,%rax), %r14b 

// CHECK: orb %r14b, 485498096 
// CHECK: encoding: [0x44,0x08,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
orb %r14b, 485498096 

// CHECK: orb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x08,0x72,0x40]        
orb %r14b, 64(%rdx) 

// CHECK: orb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x08,0x74,0x82,0x40]        
orb %r14b, 64(%rdx,%rax,4) 

// CHECK: orb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x08,0x74,0x82,0xc0]        
orb %r14b, -64(%rdx,%rax,4) 

// CHECK: orb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x08,0x74,0x02,0x40]        
orb %r14b, 64(%rdx,%rax) 

// CHECK: orb %r14b, %r14b 
// CHECK: encoding: [0x45,0x08,0xf6]        
orb %r14b, %r14b 

// CHECK: orb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x08,0x32]        
orb %r14b, (%rdx) 

// CHECK: orb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x0a,0x32]        
orb (%rdx), %r14b 

// CHECK: orl $0, 485498096 
// CHECK: encoding: [0x83,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
orl $0, 485498096 

// CHECK: orl $0, 64(%rdx) 
// CHECK: encoding: [0x83,0x4a,0x40,0x00]        
orl $0, 64(%rdx) 

// CHECK: orl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x4c,0x82,0x40,0x00]        
orl $0, 64(%rdx,%rax,4) 

// CHECK: orl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x4c,0x82,0xc0,0x00]        
orl $0, -64(%rdx,%rax,4) 

// CHECK: orl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x83,0x4c,0x02,0x40,0x00]        
orl $0, 64(%rdx,%rax) 

// CHECK: orl $0, %eax 
// CHECK: encoding: [0x83,0xc8,0x00]        
orl $0, %eax 

// CHECK: orl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xcd,0x00]        
orl $0, %r13d 

// CHECK: orl $0, (%rdx) 
// CHECK: encoding: [0x83,0x0a,0x00]        
orl $0, (%rdx) 

// CHECK: orl %r13d, %r13d 
// CHECK: encoding: [0x45,0x09,0xed]        
orl %r13d, %r13d 

// CHECK: orq $0, 485498096 
// CHECK: encoding: [0x48,0x83,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
orq $0, 485498096 

// CHECK: orq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0x83,0x4a,0x40,0x00]        
orq $0, 64(%rdx) 

// CHECK: orq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x4c,0x82,0x40,0x00]        
orq $0, 64(%rdx,%rax,4) 

// CHECK: orq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x4c,0x82,0xc0,0x00]        
orq $0, -64(%rdx,%rax,4) 

// CHECK: orq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x83,0x4c,0x02,0x40,0x00]        
orq $0, 64(%rdx,%rax) 

// CHECK: orq $0, (%rdx) 
// CHECK: encoding: [0x48,0x83,0x0a,0x00]        
orq $0, (%rdx) 

// CHECK: orw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
orw $0, 485498096 

// CHECK: orw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0x83,0x4a,0x40,0x00]        
orw $0, 64(%rdx) 

// CHECK: orw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x4c,0x82,0x40,0x00]        
orw $0, 64(%rdx,%rax,4) 

// CHECK: orw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x4c,0x82,0xc0,0x00]        
orw $0, -64(%rdx,%rax,4) 

// CHECK: orw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x83,0x4c,0x02,0x40,0x00]        
orw $0, 64(%rdx,%rax) 

// CHECK: orw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0x83,0xce,0x00]        
orw $0, %r14w 

// CHECK: orw $0, (%rdx) 
// CHECK: encoding: [0x66,0x83,0x0a,0x00]        
orw $0, (%rdx) 

// CHECK: orw 485498096, %r14w 
// CHECK: encoding: [0x66,0x44,0x0b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
orw 485498096, %r14w 

// CHECK: orw 64(%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x0b,0x72,0x40]        
orw 64(%rdx), %r14w 

// CHECK: orw 64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x0b,0x74,0x82,0x40]        
orw 64(%rdx,%rax,4), %r14w 

// CHECK: orw -64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x0b,0x74,0x82,0xc0]        
orw -64(%rdx,%rax,4), %r14w 

// CHECK: orw 64(%rdx,%rax), %r14w 
// CHECK: encoding: [0x66,0x44,0x0b,0x74,0x02,0x40]        
orw 64(%rdx,%rax), %r14w 

// CHECK: orw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x09,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
orw %r14w, 485498096 

// CHECK: orw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x09,0x72,0x40]        
orw %r14w, 64(%rdx) 

// CHECK: orw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x09,0x74,0x82,0x40]        
orw %r14w, 64(%rdx,%rax,4) 

// CHECK: orw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x09,0x74,0x82,0xc0]        
orw %r14w, -64(%rdx,%rax,4) 

// CHECK: orw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x09,0x74,0x02,0x40]        
orw %r14w, 64(%rdx,%rax) 

// CHECK: orw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x09,0xf6]        
orw %r14w, %r14w 

// CHECK: orw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x09,0x32]        
orw %r14w, (%rdx) 

// CHECK: orw (%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x0b,0x32]        
orw (%rdx), %r14w 

// CHECK: outb %al, $0 
// CHECK: encoding: [0xe6,0x00]        
outb %al, $0 

// CHECK: outb %al, %dx 
// CHECK: encoding: [0xee]        
outb %al, %dx 

// CHECK: outl %eax, $0 
// CHECK: encoding: [0xe7,0x00]        
outl %eax, $0 

// CHECK: outl %eax, %dx 
// CHECK: encoding: [0xef]        
outl %eax, %dx 

// CHECK: pause 
// CHECK: encoding: [0xf3,0x90]          
pause 

// CHECK: popfq 
// CHECK: encoding: [0x9d]          
popfq 

// CHECK: popfw 
// CHECK: encoding: [0x66,0x9d]          
popfw 

// CHECK: pushfq 
// CHECK: encoding: [0x9c]          
pushfq 

// CHECK: pushfw 
// CHECK: encoding: [0x66,0x9c]          
pushfw 

// CHECK: rclb 485498096 
// CHECK: encoding: [0xd0,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
rclb 485498096 

// CHECK: rclb 64(%rdx) 
// CHECK: encoding: [0xd0,0x52,0x40]         
rclb 64(%rdx) 

// CHECK: rclb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x54,0x82,0x40]         
rclb 64(%rdx,%rax,4) 

// CHECK: rclb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x54,0x82,0xc0]         
rclb -64(%rdx,%rax,4) 

// CHECK: rclb 64(%rdx,%rax) 
// CHECK: encoding: [0xd0,0x54,0x02,0x40]         
rclb 64(%rdx,%rax) 

// CHECK: rclb %cl, 485498096 
// CHECK: encoding: [0xd2,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
rclb %cl, 485498096 

// CHECK: rclb %cl, 64(%rdx) 
// CHECK: encoding: [0xd2,0x52,0x40]        
rclb %cl, 64(%rdx) 

// CHECK: rclb %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x54,0x82,0x40]        
rclb %cl, 64(%rdx,%rax,4) 

// CHECK: rclb %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x54,0x82,0xc0]        
rclb %cl, -64(%rdx,%rax,4) 

// CHECK: rclb %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd2,0x54,0x02,0x40]        
rclb %cl, 64(%rdx,%rax) 

// CHECK: rclb %cl, %r14b 
// CHECK: encoding: [0x41,0xd2,0xd6]        
rclb %cl, %r14b 

// CHECK: rclb %cl, (%rdx) 
// CHECK: encoding: [0xd2,0x12]        
rclb %cl, (%rdx) 

// CHECK: rclb %r14b 
// CHECK: encoding: [0x41,0xd0,0xd6]         
rclb %r14b 

// CHECK: rclb (%rdx) 
// CHECK: encoding: [0xd0,0x12]         
rclb (%rdx) 

// CHECK: rcll 485498096 
// CHECK: encoding: [0xd1,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
rcll 485498096 

// CHECK: rcll 64(%rdx) 
// CHECK: encoding: [0xd1,0x52,0x40]         
rcll 64(%rdx) 

// CHECK: rcll 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x54,0x82,0x40]         
rcll 64(%rdx,%rax,4) 

// CHECK: rcll -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x54,0x82,0xc0]         
rcll -64(%rdx,%rax,4) 

// CHECK: rcll 64(%rdx,%rax) 
// CHECK: encoding: [0xd1,0x54,0x02,0x40]         
rcll 64(%rdx,%rax) 

// CHECK: rcll %cl, 485498096 
// CHECK: encoding: [0xd3,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
rcll %cl, 485498096 

// CHECK: rcll %cl, 64(%rdx) 
// CHECK: encoding: [0xd3,0x52,0x40]        
rcll %cl, 64(%rdx) 

// CHECK: rcll %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x54,0x82,0x40]        
rcll %cl, 64(%rdx,%rax,4) 

// CHECK: rcll %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x54,0x82,0xc0]        
rcll %cl, -64(%rdx,%rax,4) 

// CHECK: rcll %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd3,0x54,0x02,0x40]        
rcll %cl, 64(%rdx,%rax) 

// CHECK: rcll %cl, %r13d 
// CHECK: encoding: [0x41,0xd3,0xd5]        
rcll %cl, %r13d 

// CHECK: rcll %cl, (%rdx) 
// CHECK: encoding: [0xd3,0x12]        
rcll %cl, (%rdx) 

// CHECK: rcll %r13d 
// CHECK: encoding: [0x41,0xd1,0xd5]         
rcll %r13d 

// CHECK: rcll (%rdx) 
// CHECK: encoding: [0xd1,0x12]         
rcll (%rdx) 

// CHECK: rclq 485498096 
// CHECK: encoding: [0x48,0xd1,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
rclq 485498096 

// CHECK: rclq 64(%rdx) 
// CHECK: encoding: [0x48,0xd1,0x52,0x40]         
rclq 64(%rdx) 

// CHECK: rclq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x54,0x82,0x40]         
rclq 64(%rdx,%rax,4) 

// CHECK: rclq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x54,0x82,0xc0]         
rclq -64(%rdx,%rax,4) 

// CHECK: rclq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd1,0x54,0x02,0x40]         
rclq 64(%rdx,%rax) 

// CHECK: rclq %cl, 485498096 
// CHECK: encoding: [0x48,0xd3,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
rclq %cl, 485498096 

// CHECK: rclq %cl, 64(%rdx) 
// CHECK: encoding: [0x48,0xd3,0x52,0x40]        
rclq %cl, 64(%rdx) 

// CHECK: rclq %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x54,0x82,0x40]        
rclq %cl, 64(%rdx,%rax,4) 

// CHECK: rclq %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x54,0x82,0xc0]        
rclq %cl, -64(%rdx,%rax,4) 

// CHECK: rclq %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd3,0x54,0x02,0x40]        
rclq %cl, 64(%rdx,%rax) 

// CHECK: rclq %cl, (%rdx) 
// CHECK: encoding: [0x48,0xd3,0x12]        
rclq %cl, (%rdx) 

// CHECK: rclq (%rdx) 
// CHECK: encoding: [0x48,0xd1,0x12]         
rclq (%rdx) 

// CHECK: rclw 485498096 
// CHECK: encoding: [0x66,0xd1,0x14,0x25,0xf0,0x1c,0xf0,0x1c]         
rclw 485498096 

// CHECK: rclw 64(%rdx) 
// CHECK: encoding: [0x66,0xd1,0x52,0x40]         
rclw 64(%rdx) 

// CHECK: rclw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x54,0x82,0x40]         
rclw 64(%rdx,%rax,4) 

// CHECK: rclw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x54,0x82,0xc0]         
rclw -64(%rdx,%rax,4) 

// CHECK: rclw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd1,0x54,0x02,0x40]         
rclw 64(%rdx,%rax) 

// CHECK: rclw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
rclw %cl, 485498096 

// CHECK: rclw %cl, 64(%rdx) 
// CHECK: encoding: [0x66,0xd3,0x52,0x40]        
rclw %cl, 64(%rdx) 

// CHECK: rclw %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x54,0x82,0x40]        
rclw %cl, 64(%rdx,%rax,4) 

// CHECK: rclw %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x54,0x82,0xc0]        
rclw %cl, -64(%rdx,%rax,4) 

// CHECK: rclw %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd3,0x54,0x02,0x40]        
rclw %cl, 64(%rdx,%rax) 

// CHECK: rclw %cl, %r14w 
// CHECK: encoding: [0x66,0x41,0xd3,0xd6]        
rclw %cl, %r14w 

// CHECK: rclw %cl, (%rdx) 
// CHECK: encoding: [0x66,0xd3,0x12]        
rclw %cl, (%rdx) 

// CHECK: rclw %r14w 
// CHECK: encoding: [0x66,0x41,0xd1,0xd6]         
rclw %r14w 

// CHECK: rclw (%rdx) 
// CHECK: encoding: [0x66,0xd1,0x12]         
rclw (%rdx) 

// CHECK: rcrb 485498096 
// CHECK: encoding: [0xd0,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
rcrb 485498096 

// CHECK: rcrb 64(%rdx) 
// CHECK: encoding: [0xd0,0x5a,0x40]         
rcrb 64(%rdx) 

// CHECK: rcrb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x5c,0x82,0x40]         
rcrb 64(%rdx,%rax,4) 

// CHECK: rcrb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x5c,0x82,0xc0]         
rcrb -64(%rdx,%rax,4) 

// CHECK: rcrb 64(%rdx,%rax) 
// CHECK: encoding: [0xd0,0x5c,0x02,0x40]         
rcrb 64(%rdx,%rax) 

// CHECK: rcrb %cl, 485498096 
// CHECK: encoding: [0xd2,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
rcrb %cl, 485498096 

// CHECK: rcrb %cl, 64(%rdx) 
// CHECK: encoding: [0xd2,0x5a,0x40]        
rcrb %cl, 64(%rdx) 

// CHECK: rcrb %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x5c,0x82,0x40]        
rcrb %cl, 64(%rdx,%rax,4) 

// CHECK: rcrb %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x5c,0x82,0xc0]        
rcrb %cl, -64(%rdx,%rax,4) 

// CHECK: rcrb %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd2,0x5c,0x02,0x40]        
rcrb %cl, 64(%rdx,%rax) 

// CHECK: rcrb %cl, %r14b 
// CHECK: encoding: [0x41,0xd2,0xde]        
rcrb %cl, %r14b 

// CHECK: rcrb %cl, (%rdx) 
// CHECK: encoding: [0xd2,0x1a]        
rcrb %cl, (%rdx) 

// CHECK: rcrb %r14b 
// CHECK: encoding: [0x41,0xd0,0xde]         
rcrb %r14b 

// CHECK: rcrb (%rdx) 
// CHECK: encoding: [0xd0,0x1a]         
rcrb (%rdx) 

// CHECK: rcrl 485498096 
// CHECK: encoding: [0xd1,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
rcrl 485498096 

// CHECK: rcrl 64(%rdx) 
// CHECK: encoding: [0xd1,0x5a,0x40]         
rcrl 64(%rdx) 

// CHECK: rcrl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x5c,0x82,0x40]         
rcrl 64(%rdx,%rax,4) 

// CHECK: rcrl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x5c,0x82,0xc0]         
rcrl -64(%rdx,%rax,4) 

// CHECK: rcrl 64(%rdx,%rax) 
// CHECK: encoding: [0xd1,0x5c,0x02,0x40]         
rcrl 64(%rdx,%rax) 

// CHECK: rcrl %cl, 485498096 
// CHECK: encoding: [0xd3,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
rcrl %cl, 485498096 

// CHECK: rcrl %cl, 64(%rdx) 
// CHECK: encoding: [0xd3,0x5a,0x40]        
rcrl %cl, 64(%rdx) 

// CHECK: rcrl %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x5c,0x82,0x40]        
rcrl %cl, 64(%rdx,%rax,4) 

// CHECK: rcrl %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x5c,0x82,0xc0]        
rcrl %cl, -64(%rdx,%rax,4) 

// CHECK: rcrl %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd3,0x5c,0x02,0x40]        
rcrl %cl, 64(%rdx,%rax) 

// CHECK: rcrl %cl, %r13d 
// CHECK: encoding: [0x41,0xd3,0xdd]        
rcrl %cl, %r13d 

// CHECK: rcrl %cl, (%rdx) 
// CHECK: encoding: [0xd3,0x1a]        
rcrl %cl, (%rdx) 

// CHECK: rcrl %r13d 
// CHECK: encoding: [0x41,0xd1,0xdd]         
rcrl %r13d 

// CHECK: rcrl (%rdx) 
// CHECK: encoding: [0xd1,0x1a]         
rcrl (%rdx) 

// CHECK: rcrq 485498096 
// CHECK: encoding: [0x48,0xd1,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
rcrq 485498096 

// CHECK: rcrq 64(%rdx) 
// CHECK: encoding: [0x48,0xd1,0x5a,0x40]         
rcrq 64(%rdx) 

// CHECK: rcrq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x5c,0x82,0x40]         
rcrq 64(%rdx,%rax,4) 

// CHECK: rcrq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x5c,0x82,0xc0]         
rcrq -64(%rdx,%rax,4) 

// CHECK: rcrq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd1,0x5c,0x02,0x40]         
rcrq 64(%rdx,%rax) 

// CHECK: rcrq %cl, 485498096 
// CHECK: encoding: [0x48,0xd3,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
rcrq %cl, 485498096 

// CHECK: rcrq %cl, 64(%rdx) 
// CHECK: encoding: [0x48,0xd3,0x5a,0x40]        
rcrq %cl, 64(%rdx) 

// CHECK: rcrq %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x5c,0x82,0x40]        
rcrq %cl, 64(%rdx,%rax,4) 

// CHECK: rcrq %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x5c,0x82,0xc0]        
rcrq %cl, -64(%rdx,%rax,4) 

// CHECK: rcrq %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd3,0x5c,0x02,0x40]        
rcrq %cl, 64(%rdx,%rax) 

// CHECK: rcrq %cl, (%rdx) 
// CHECK: encoding: [0x48,0xd3,0x1a]        
rcrq %cl, (%rdx) 

// CHECK: rcrq (%rdx) 
// CHECK: encoding: [0x48,0xd1,0x1a]         
rcrq (%rdx) 

// CHECK: rcrw 485498096 
// CHECK: encoding: [0x66,0xd1,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]         
rcrw 485498096 

// CHECK: rcrw 64(%rdx) 
// CHECK: encoding: [0x66,0xd1,0x5a,0x40]         
rcrw 64(%rdx) 

// CHECK: rcrw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x5c,0x82,0x40]         
rcrw 64(%rdx,%rax,4) 

// CHECK: rcrw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x5c,0x82,0xc0]         
rcrw -64(%rdx,%rax,4) 

// CHECK: rcrw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd1,0x5c,0x02,0x40]         
rcrw 64(%rdx,%rax) 

// CHECK: rcrw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
rcrw %cl, 485498096 

// CHECK: rcrw %cl, 64(%rdx) 
// CHECK: encoding: [0x66,0xd3,0x5a,0x40]        
rcrw %cl, 64(%rdx) 

// CHECK: rcrw %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x5c,0x82,0x40]        
rcrw %cl, 64(%rdx,%rax,4) 

// CHECK: rcrw %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x5c,0x82,0xc0]        
rcrw %cl, -64(%rdx,%rax,4) 

// CHECK: rcrw %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd3,0x5c,0x02,0x40]        
rcrw %cl, 64(%rdx,%rax) 

// CHECK: rcrw %cl, %r14w 
// CHECK: encoding: [0x66,0x41,0xd3,0xde]        
rcrw %cl, %r14w 

// CHECK: rcrw %cl, (%rdx) 
// CHECK: encoding: [0x66,0xd3,0x1a]        
rcrw %cl, (%rdx) 

// CHECK: rcrw %r14w 
// CHECK: encoding: [0x66,0x41,0xd1,0xde]         
rcrw %r14w 

// CHECK: rcrw (%rdx) 
// CHECK: encoding: [0x66,0xd1,0x1a]         
rcrw (%rdx) 

// CHECK: rep cmpsb %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x65,0xa6]       
rep cmpsb %es:(%rdi), %gs:(%rsi) 

// CHECK: rep cmpsl %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x65,0xa7]       
rep cmpsl %es:(%rdi), %gs:(%rsi) 

// CHECK: rep cmpsq %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x65,0x48,0xa7]       
rep cmpsq %es:(%rdi), %gs:(%rsi) 

// CHECK: rep cmpsw %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf3,0x65,0x66,0xa7]       
rep cmpsw %es:(%rdi), %gs:(%rsi) 

// CHECK: rep lodsb %gs:(%rsi), %al 
// CHECK: encoding: [0xf3,0x65,0xac]       
rep lodsb %gs:(%rsi), %al 

// CHECK: rep lodsw %gs:(%rsi), %ax 
// CHECK: encoding: [0xf3,0x65,0x66,0xad]       
rep lodsw %gs:(%rsi), %ax 

// CHECK: rep movsb %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x65,0xa4]       
rep movsb %gs:(%rsi), %es:(%rdi) 

// CHECK: rep movsl %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x65,0xa5]       
rep movsl %gs:(%rsi), %es:(%rdi) 

// CHECK: rep movsq %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x65,0x48,0xa5]       
rep movsq %gs:(%rsi), %es:(%rdi) 

// CHECK: rep movsw %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf3,0x65,0x66,0xa5]       
rep movsw %gs:(%rsi), %es:(%rdi) 

// CHECK: repne cmpsb %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x65,0xa6]       
repne cmpsb %es:(%rdi), %gs:(%rsi) 

// CHECK: repne cmpsl %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x65,0xa7]       
repne cmpsl %es:(%rdi), %gs:(%rsi) 

// CHECK: repne cmpsq %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x65,0x48,0xa7]       
repne cmpsq %es:(%rdi), %gs:(%rsi) 

// CHECK: repne cmpsw %es:(%rdi), %gs:(%rsi) 
// CHECK: encoding: [0xf2,0x65,0x66,0xa7]       
repne cmpsw %es:(%rdi), %gs:(%rsi) 

// CHECK: repne lodsb %gs:(%rsi), %al 
// CHECK: encoding: [0xf2,0x65,0xac]       
repne lodsb %gs:(%rsi), %al 

// CHECK: repne lodsw %gs:(%rsi), %ax 
// CHECK: encoding: [0xf2,0x65,0x66,0xad]       
repne lodsw %gs:(%rsi), %ax 

// CHECK: repne movsb %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x65,0xa4]       
repne movsb %gs:(%rsi), %es:(%rdi) 

// CHECK: repne movsl %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x65,0xa5]       
repne movsl %gs:(%rsi), %es:(%rdi) 

// CHECK: repne movsq %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x65,0x48,0xa5]       
repne movsq %gs:(%rsi), %es:(%rdi) 

// CHECK: repne movsw %gs:(%rsi), %es:(%rdi) 
// CHECK: encoding: [0xf2,0x65,0x66,0xa5]       
repne movsw %gs:(%rsi), %es:(%rdi) 

// CHECK: repne scasb %es:(%rdi), %al 
// CHECK: encoding: [0xf2,0xae]       
repne scasb %es:(%rdi), %al 

// CHECK: repne scasw %es:(%rdi), %ax 
// CHECK: encoding: [0xf2,0x66,0xaf]       
repne scasw %es:(%rdi), %ax 

// CHECK: repne stosb %al, %es:(%rdi) 
// CHECK: encoding: [0xf2,0xaa]       
repne stosb %al, %es:(%rdi) 

// CHECK: repne stosw %ax, %es:(%rdi) 
// CHECK: encoding: [0xf2,0x66,0xab]       
repne stosw %ax, %es:(%rdi) 

// CHECK: rep scasb %es:(%rdi), %al 
// CHECK: encoding: [0xf3,0xae]       
rep scasb %es:(%rdi), %al 

// CHECK: rep scasw %es:(%rdi), %ax 
// CHECK: encoding: [0xf3,0x66,0xaf]       
rep scasw %es:(%rdi), %ax 

// CHECK: rep stosb %al, %es:(%rdi) 
// CHECK: encoding: [0xf3,0xaa]       
rep stosb %al, %es:(%rdi) 

// CHECK: rep stosw %ax, %es:(%rdi) 
// CHECK: encoding: [0xf3,0x66,0xab]       
rep stosw %ax, %es:(%rdi) 

// CHECK: rolb 485498096 
// CHECK: encoding: [0xd0,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
rolb 485498096 

// CHECK: rolb 64(%rdx) 
// CHECK: encoding: [0xd0,0x42,0x40]         
rolb 64(%rdx) 

// CHECK: rolb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x44,0x82,0x40]         
rolb 64(%rdx,%rax,4) 

// CHECK: rolb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x44,0x82,0xc0]         
rolb -64(%rdx,%rax,4) 

// CHECK: rolb 64(%rdx,%rax) 
// CHECK: encoding: [0xd0,0x44,0x02,0x40]         
rolb 64(%rdx,%rax) 

// CHECK: rolb %cl, 485498096 
// CHECK: encoding: [0xd2,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
rolb %cl, 485498096 

// CHECK: rolb %cl, 64(%rdx) 
// CHECK: encoding: [0xd2,0x42,0x40]        
rolb %cl, 64(%rdx) 

// CHECK: rolb %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x44,0x82,0x40]        
rolb %cl, 64(%rdx,%rax,4) 

// CHECK: rolb %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x44,0x82,0xc0]        
rolb %cl, -64(%rdx,%rax,4) 

// CHECK: rolb %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd2,0x44,0x02,0x40]        
rolb %cl, 64(%rdx,%rax) 

// CHECK: rolb %cl, %r14b 
// CHECK: encoding: [0x41,0xd2,0xc6]        
rolb %cl, %r14b 

// CHECK: rolb %cl, (%rdx) 
// CHECK: encoding: [0xd2,0x02]        
rolb %cl, (%rdx) 

// CHECK: rolb %r14b 
// CHECK: encoding: [0x41,0xd0,0xc6]         
rolb %r14b 

// CHECK: rolb (%rdx) 
// CHECK: encoding: [0xd0,0x02]         
rolb (%rdx) 

// CHECK: roll 485498096 
// CHECK: encoding: [0xd1,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
roll 485498096 

// CHECK: roll 64(%rdx) 
// CHECK: encoding: [0xd1,0x42,0x40]         
roll 64(%rdx) 

// CHECK: roll 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x44,0x82,0x40]         
roll 64(%rdx,%rax,4) 

// CHECK: roll -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x44,0x82,0xc0]         
roll -64(%rdx,%rax,4) 

// CHECK: roll 64(%rdx,%rax) 
// CHECK: encoding: [0xd1,0x44,0x02,0x40]         
roll 64(%rdx,%rax) 

// CHECK: roll %cl, 485498096 
// CHECK: encoding: [0xd3,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
roll %cl, 485498096 

// CHECK: roll %cl, 64(%rdx) 
// CHECK: encoding: [0xd3,0x42,0x40]        
roll %cl, 64(%rdx) 

// CHECK: roll %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x44,0x82,0x40]        
roll %cl, 64(%rdx,%rax,4) 

// CHECK: roll %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x44,0x82,0xc0]        
roll %cl, -64(%rdx,%rax,4) 

// CHECK: roll %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd3,0x44,0x02,0x40]        
roll %cl, 64(%rdx,%rax) 

// CHECK: roll %cl, %r13d 
// CHECK: encoding: [0x41,0xd3,0xc5]        
roll %cl, %r13d 

// CHECK: roll %cl, (%rdx) 
// CHECK: encoding: [0xd3,0x02]        
roll %cl, (%rdx) 

// CHECK: roll %r13d 
// CHECK: encoding: [0x41,0xd1,0xc5]         
roll %r13d 

// CHECK: roll (%rdx) 
// CHECK: encoding: [0xd1,0x02]         
roll (%rdx) 

// CHECK: rolq 485498096 
// CHECK: encoding: [0x48,0xd1,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
rolq 485498096 

// CHECK: rolq 64(%rdx) 
// CHECK: encoding: [0x48,0xd1,0x42,0x40]         
rolq 64(%rdx) 

// CHECK: rolq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x44,0x82,0x40]         
rolq 64(%rdx,%rax,4) 

// CHECK: rolq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x44,0x82,0xc0]         
rolq -64(%rdx,%rax,4) 

// CHECK: rolq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd1,0x44,0x02,0x40]         
rolq 64(%rdx,%rax) 

// CHECK: rolq %cl, 485498096 
// CHECK: encoding: [0x48,0xd3,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
rolq %cl, 485498096 

// CHECK: rolq %cl, 64(%rdx) 
// CHECK: encoding: [0x48,0xd3,0x42,0x40]        
rolq %cl, 64(%rdx) 

// CHECK: rolq %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x44,0x82,0x40]        
rolq %cl, 64(%rdx,%rax,4) 

// CHECK: rolq %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x44,0x82,0xc0]        
rolq %cl, -64(%rdx,%rax,4) 

// CHECK: rolq %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd3,0x44,0x02,0x40]        
rolq %cl, 64(%rdx,%rax) 

// CHECK: rolq %cl, (%rdx) 
// CHECK: encoding: [0x48,0xd3,0x02]        
rolq %cl, (%rdx) 

// CHECK: rolq (%rdx) 
// CHECK: encoding: [0x48,0xd1,0x02]         
rolq (%rdx) 

// CHECK: rolw 485498096 
// CHECK: encoding: [0x66,0xd1,0x04,0x25,0xf0,0x1c,0xf0,0x1c]         
rolw 485498096 

// CHECK: rolw 64(%rdx) 
// CHECK: encoding: [0x66,0xd1,0x42,0x40]         
rolw 64(%rdx) 

// CHECK: rolw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x44,0x82,0x40]         
rolw 64(%rdx,%rax,4) 

// CHECK: rolw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x44,0x82,0xc0]         
rolw -64(%rdx,%rax,4) 

// CHECK: rolw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd1,0x44,0x02,0x40]         
rolw 64(%rdx,%rax) 

// CHECK: rolw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
rolw %cl, 485498096 

// CHECK: rolw %cl, 64(%rdx) 
// CHECK: encoding: [0x66,0xd3,0x42,0x40]        
rolw %cl, 64(%rdx) 

// CHECK: rolw %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x44,0x82,0x40]        
rolw %cl, 64(%rdx,%rax,4) 

// CHECK: rolw %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x44,0x82,0xc0]        
rolw %cl, -64(%rdx,%rax,4) 

// CHECK: rolw %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd3,0x44,0x02,0x40]        
rolw %cl, 64(%rdx,%rax) 

// CHECK: rolw %cl, %r14w 
// CHECK: encoding: [0x66,0x41,0xd3,0xc6]        
rolw %cl, %r14w 

// CHECK: rolw %cl, (%rdx) 
// CHECK: encoding: [0x66,0xd3,0x02]        
rolw %cl, (%rdx) 

// CHECK: rolw %r14w 
// CHECK: encoding: [0x66,0x41,0xd1,0xc6]         
rolw %r14w 

// CHECK: rolw (%rdx) 
// CHECK: encoding: [0x66,0xd1,0x02]         
rolw (%rdx) 

// CHECK: rorb 485498096 
// CHECK: encoding: [0xd0,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
rorb 485498096 

// CHECK: rorb 64(%rdx) 
// CHECK: encoding: [0xd0,0x4a,0x40]         
rorb 64(%rdx) 

// CHECK: rorb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x4c,0x82,0x40]         
rorb 64(%rdx,%rax,4) 

// CHECK: rorb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x4c,0x82,0xc0]         
rorb -64(%rdx,%rax,4) 

// CHECK: rorb 64(%rdx,%rax) 
// CHECK: encoding: [0xd0,0x4c,0x02,0x40]         
rorb 64(%rdx,%rax) 

// CHECK: rorb %cl, 485498096 
// CHECK: encoding: [0xd2,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
rorb %cl, 485498096 

// CHECK: rorb %cl, 64(%rdx) 
// CHECK: encoding: [0xd2,0x4a,0x40]        
rorb %cl, 64(%rdx) 

// CHECK: rorb %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x4c,0x82,0x40]        
rorb %cl, 64(%rdx,%rax,4) 

// CHECK: rorb %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x4c,0x82,0xc0]        
rorb %cl, -64(%rdx,%rax,4) 

// CHECK: rorb %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd2,0x4c,0x02,0x40]        
rorb %cl, 64(%rdx,%rax) 

// CHECK: rorb %cl, %r14b 
// CHECK: encoding: [0x41,0xd2,0xce]        
rorb %cl, %r14b 

// CHECK: rorb %cl, (%rdx) 
// CHECK: encoding: [0xd2,0x0a]        
rorb %cl, (%rdx) 

// CHECK: rorb %r14b 
// CHECK: encoding: [0x41,0xd0,0xce]         
rorb %r14b 

// CHECK: rorb (%rdx) 
// CHECK: encoding: [0xd0,0x0a]         
rorb (%rdx) 

// CHECK: rorl 485498096 
// CHECK: encoding: [0xd1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
rorl 485498096 

// CHECK: rorl 64(%rdx) 
// CHECK: encoding: [0xd1,0x4a,0x40]         
rorl 64(%rdx) 

// CHECK: rorl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x4c,0x82,0x40]         
rorl 64(%rdx,%rax,4) 

// CHECK: rorl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x4c,0x82,0xc0]         
rorl -64(%rdx,%rax,4) 

// CHECK: rorl 64(%rdx,%rax) 
// CHECK: encoding: [0xd1,0x4c,0x02,0x40]         
rorl 64(%rdx,%rax) 

// CHECK: rorl %cl, 485498096 
// CHECK: encoding: [0xd3,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
rorl %cl, 485498096 

// CHECK: rorl %cl, 64(%rdx) 
// CHECK: encoding: [0xd3,0x4a,0x40]        
rorl %cl, 64(%rdx) 

// CHECK: rorl %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x4c,0x82,0x40]        
rorl %cl, 64(%rdx,%rax,4) 

// CHECK: rorl %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x4c,0x82,0xc0]        
rorl %cl, -64(%rdx,%rax,4) 

// CHECK: rorl %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd3,0x4c,0x02,0x40]        
rorl %cl, 64(%rdx,%rax) 

// CHECK: rorl %cl, %r13d 
// CHECK: encoding: [0x41,0xd3,0xcd]        
rorl %cl, %r13d 

// CHECK: rorl %cl, (%rdx) 
// CHECK: encoding: [0xd3,0x0a]        
rorl %cl, (%rdx) 

// CHECK: rorl %r13d 
// CHECK: encoding: [0x41,0xd1,0xcd]         
rorl %r13d 

// CHECK: rorl (%rdx) 
// CHECK: encoding: [0xd1,0x0a]         
rorl (%rdx) 

// CHECK: rorq 485498096 
// CHECK: encoding: [0x48,0xd1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
rorq 485498096 

// CHECK: rorq 64(%rdx) 
// CHECK: encoding: [0x48,0xd1,0x4a,0x40]         
rorq 64(%rdx) 

// CHECK: rorq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x4c,0x82,0x40]         
rorq 64(%rdx,%rax,4) 

// CHECK: rorq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x4c,0x82,0xc0]         
rorq -64(%rdx,%rax,4) 

// CHECK: rorq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd1,0x4c,0x02,0x40]         
rorq 64(%rdx,%rax) 

// CHECK: rorq %cl, 485498096 
// CHECK: encoding: [0x48,0xd3,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
rorq %cl, 485498096 

// CHECK: rorq %cl, 64(%rdx) 
// CHECK: encoding: [0x48,0xd3,0x4a,0x40]        
rorq %cl, 64(%rdx) 

// CHECK: rorq %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x4c,0x82,0x40]        
rorq %cl, 64(%rdx,%rax,4) 

// CHECK: rorq %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x4c,0x82,0xc0]        
rorq %cl, -64(%rdx,%rax,4) 

// CHECK: rorq %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd3,0x4c,0x02,0x40]        
rorq %cl, 64(%rdx,%rax) 

// CHECK: rorq %cl, (%rdx) 
// CHECK: encoding: [0x48,0xd3,0x0a]        
rorq %cl, (%rdx) 

// CHECK: rorq (%rdx) 
// CHECK: encoding: [0x48,0xd1,0x0a]         
rorq (%rdx) 

// CHECK: rorw 485498096 
// CHECK: encoding: [0x66,0xd1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]         
rorw 485498096 

// CHECK: rorw 64(%rdx) 
// CHECK: encoding: [0x66,0xd1,0x4a,0x40]         
rorw 64(%rdx) 

// CHECK: rorw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x4c,0x82,0x40]         
rorw 64(%rdx,%rax,4) 

// CHECK: rorw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x4c,0x82,0xc0]         
rorw -64(%rdx,%rax,4) 

// CHECK: rorw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd1,0x4c,0x02,0x40]         
rorw 64(%rdx,%rax) 

// CHECK: rorw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
rorw %cl, 485498096 

// CHECK: rorw %cl, 64(%rdx) 
// CHECK: encoding: [0x66,0xd3,0x4a,0x40]        
rorw %cl, 64(%rdx) 

// CHECK: rorw %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x4c,0x82,0x40]        
rorw %cl, 64(%rdx,%rax,4) 

// CHECK: rorw %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x4c,0x82,0xc0]        
rorw %cl, -64(%rdx,%rax,4) 

// CHECK: rorw %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd3,0x4c,0x02,0x40]        
rorw %cl, 64(%rdx,%rax) 

// CHECK: rorw %cl, %r14w 
// CHECK: encoding: [0x66,0x41,0xd3,0xce]        
rorw %cl, %r14w 

// CHECK: rorw %cl, (%rdx) 
// CHECK: encoding: [0x66,0xd3,0x0a]        
rorw %cl, (%rdx) 

// CHECK: rorw %r14w 
// CHECK: encoding: [0x66,0x41,0xd1,0xce]         
rorw %r14w 

// CHECK: rorw (%rdx) 
// CHECK: encoding: [0x66,0xd1,0x0a]         
rorw (%rdx) 

// CHECK: sarb 485498096 
// CHECK: encoding: [0xd0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
sarb 485498096 

// CHECK: sarb 64(%rdx) 
// CHECK: encoding: [0xd0,0x7a,0x40]         
sarb 64(%rdx) 

// CHECK: sarb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x7c,0x82,0x40]         
sarb 64(%rdx,%rax,4) 

// CHECK: sarb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x7c,0x82,0xc0]         
sarb -64(%rdx,%rax,4) 

// CHECK: sarb 64(%rdx,%rax) 
// CHECK: encoding: [0xd0,0x7c,0x02,0x40]         
sarb 64(%rdx,%rax) 

// CHECK: sarb %cl, 485498096 
// CHECK: encoding: [0xd2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sarb %cl, 485498096 

// CHECK: sarb %cl, 64(%rdx) 
// CHECK: encoding: [0xd2,0x7a,0x40]        
sarb %cl, 64(%rdx) 

// CHECK: sarb %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x7c,0x82,0x40]        
sarb %cl, 64(%rdx,%rax,4) 

// CHECK: sarb %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x7c,0x82,0xc0]        
sarb %cl, -64(%rdx,%rax,4) 

// CHECK: sarb %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd2,0x7c,0x02,0x40]        
sarb %cl, 64(%rdx,%rax) 

// CHECK: sarb %cl, %r14b 
// CHECK: encoding: [0x41,0xd2,0xfe]        
sarb %cl, %r14b 

// CHECK: sarb %cl, (%rdx) 
// CHECK: encoding: [0xd2,0x3a]        
sarb %cl, (%rdx) 

// CHECK: sarb %r14b 
// CHECK: encoding: [0x41,0xd0,0xfe]         
sarb %r14b 

// CHECK: sarb (%rdx) 
// CHECK: encoding: [0xd0,0x3a]         
sarb (%rdx) 

// CHECK: sarl 485498096 
// CHECK: encoding: [0xd1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
sarl 485498096 

// CHECK: sarl 64(%rdx) 
// CHECK: encoding: [0xd1,0x7a,0x40]         
sarl 64(%rdx) 

// CHECK: sarl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x7c,0x82,0x40]         
sarl 64(%rdx,%rax,4) 

// CHECK: sarl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x7c,0x82,0xc0]         
sarl -64(%rdx,%rax,4) 

// CHECK: sarl 64(%rdx,%rax) 
// CHECK: encoding: [0xd1,0x7c,0x02,0x40]         
sarl 64(%rdx,%rax) 

// CHECK: sarl %cl, 485498096 
// CHECK: encoding: [0xd3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sarl %cl, 485498096 

// CHECK: sarl %cl, 64(%rdx) 
// CHECK: encoding: [0xd3,0x7a,0x40]        
sarl %cl, 64(%rdx) 

// CHECK: sarl %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x7c,0x82,0x40]        
sarl %cl, 64(%rdx,%rax,4) 

// CHECK: sarl %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x7c,0x82,0xc0]        
sarl %cl, -64(%rdx,%rax,4) 

// CHECK: sarl %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd3,0x7c,0x02,0x40]        
sarl %cl, 64(%rdx,%rax) 

// CHECK: sarl %cl, %r13d 
// CHECK: encoding: [0x41,0xd3,0xfd]        
sarl %cl, %r13d 

// CHECK: sarl %cl, (%rdx) 
// CHECK: encoding: [0xd3,0x3a]        
sarl %cl, (%rdx) 

// CHECK: sarl %r13d 
// CHECK: encoding: [0x41,0xd1,0xfd]         
sarl %r13d 

// CHECK: sarl (%rdx) 
// CHECK: encoding: [0xd1,0x3a]         
sarl (%rdx) 

// CHECK: sarq 485498096 
// CHECK: encoding: [0x48,0xd1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
sarq 485498096 

// CHECK: sarq 64(%rdx) 
// CHECK: encoding: [0x48,0xd1,0x7a,0x40]         
sarq 64(%rdx) 

// CHECK: sarq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x7c,0x82,0x40]         
sarq 64(%rdx,%rax,4) 

// CHECK: sarq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x7c,0x82,0xc0]         
sarq -64(%rdx,%rax,4) 

// CHECK: sarq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd1,0x7c,0x02,0x40]         
sarq 64(%rdx,%rax) 

// CHECK: sarq %cl, 485498096 
// CHECK: encoding: [0x48,0xd3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sarq %cl, 485498096 

// CHECK: sarq %cl, 64(%rdx) 
// CHECK: encoding: [0x48,0xd3,0x7a,0x40]        
sarq %cl, 64(%rdx) 

// CHECK: sarq %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x7c,0x82,0x40]        
sarq %cl, 64(%rdx,%rax,4) 

// CHECK: sarq %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x7c,0x82,0xc0]        
sarq %cl, -64(%rdx,%rax,4) 

// CHECK: sarq %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd3,0x7c,0x02,0x40]        
sarq %cl, 64(%rdx,%rax) 

// CHECK: sarq %cl, (%rdx) 
// CHECK: encoding: [0x48,0xd3,0x3a]        
sarq %cl, (%rdx) 

// CHECK: sarq (%rdx) 
// CHECK: encoding: [0x48,0xd1,0x3a]         
sarq (%rdx) 

// CHECK: sarw 485498096 
// CHECK: encoding: [0x66,0xd1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]         
sarw 485498096 

// CHECK: sarw 64(%rdx) 
// CHECK: encoding: [0x66,0xd1,0x7a,0x40]         
sarw 64(%rdx) 

// CHECK: sarw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x7c,0x82,0x40]         
sarw 64(%rdx,%rax,4) 

// CHECK: sarw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x7c,0x82,0xc0]         
sarw -64(%rdx,%rax,4) 

// CHECK: sarw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd1,0x7c,0x02,0x40]         
sarw 64(%rdx,%rax) 

// CHECK: sarw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sarw %cl, 485498096 

// CHECK: sarw %cl, 64(%rdx) 
// CHECK: encoding: [0x66,0xd3,0x7a,0x40]        
sarw %cl, 64(%rdx) 

// CHECK: sarw %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x7c,0x82,0x40]        
sarw %cl, 64(%rdx,%rax,4) 

// CHECK: sarw %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x7c,0x82,0xc0]        
sarw %cl, -64(%rdx,%rax,4) 

// CHECK: sarw %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd3,0x7c,0x02,0x40]        
sarw %cl, 64(%rdx,%rax) 

// CHECK: sarw %cl, %r14w 
// CHECK: encoding: [0x66,0x41,0xd3,0xfe]        
sarw %cl, %r14w 

// CHECK: sarw %cl, (%rdx) 
// CHECK: encoding: [0x66,0xd3,0x3a]        
sarw %cl, (%rdx) 

// CHECK: sarw %r14w 
// CHECK: encoding: [0x66,0x41,0xd1,0xfe]         
sarw %r14w 

// CHECK: sarw (%rdx) 
// CHECK: encoding: [0x66,0xd1,0x3a]         
sarw (%rdx) 

// CHECK: sbbb $0, 485498096 
// CHECK: encoding: [0x80,0x1c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
sbbb $0, 485498096 

// CHECK: sbbb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x5a,0x40,0x00]        
sbbb $0, 64(%rdx) 

// CHECK: sbbb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x5c,0x82,0x40,0x00]        
sbbb $0, 64(%rdx,%rax,4) 

// CHECK: sbbb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x5c,0x82,0xc0,0x00]        
sbbb $0, -64(%rdx,%rax,4) 

// CHECK: sbbb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x5c,0x02,0x40,0x00]        
sbbb $0, 64(%rdx,%rax) 

// CHECK: sbbb $0, %al 
// CHECK: encoding: [0x1c,0x00]        
sbbb $0, %al 

// CHECK: sbbb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xde,0x00]        
sbbb $0, %r14b 

// CHECK: sbbb $0, (%rdx) 
// CHECK: encoding: [0x80,0x1a,0x00]        
sbbb $0, (%rdx) 

// CHECK: sbbb 485498096, %r14b 
// CHECK: encoding: [0x44,0x1a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
sbbb 485498096, %r14b 

// CHECK: sbbb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x1a,0x72,0x40]        
sbbb 64(%rdx), %r14b 

// CHECK: sbbb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x1a,0x74,0x82,0x40]        
sbbb 64(%rdx,%rax,4), %r14b 

// CHECK: sbbb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x1a,0x74,0x82,0xc0]        
sbbb -64(%rdx,%rax,4), %r14b 

// CHECK: sbbb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x1a,0x74,0x02,0x40]        
sbbb 64(%rdx,%rax), %r14b 

// CHECK: sbbb %r14b, 485498096 
// CHECK: encoding: [0x44,0x18,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
sbbb %r14b, 485498096 

// CHECK: sbbb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x18,0x72,0x40]        
sbbb %r14b, 64(%rdx) 

// CHECK: sbbb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x18,0x74,0x82,0x40]        
sbbb %r14b, 64(%rdx,%rax,4) 

// CHECK: sbbb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x18,0x74,0x82,0xc0]        
sbbb %r14b, -64(%rdx,%rax,4) 

// CHECK: sbbb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x18,0x74,0x02,0x40]        
sbbb %r14b, 64(%rdx,%rax) 

// CHECK: sbbb %r14b, %r14b 
// CHECK: encoding: [0x45,0x18,0xf6]        
sbbb %r14b, %r14b 

// CHECK: sbbb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x18,0x32]        
sbbb %r14b, (%rdx) 

// CHECK: sbbb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x1a,0x32]        
sbbb (%rdx), %r14b 

// CHECK: sbbl $0, %eax 
// CHECK: encoding: [0x83,0xd8,0x00]        
sbbl $0, %eax 

// CHECK: sbbl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xdd,0x00]        
sbbl $0, %r13d 

// CHECK: sbbl %r13d, %r13d 
// CHECK: encoding: [0x45,0x19,0xed]        
sbbl %r13d, %r13d 

// CHECK: scasb %es:(%rdi), %al 
// CHECK: encoding: [0xae]        
scasb %es:(%rdi), %al 

// CHECK: scasw %es:(%rdi), %ax 
// CHECK: encoding: [0x66,0xaf]        
scasw %es:(%rdi), %ax 

// CHECK: shlb 485498096 
// CHECK: encoding: [0xd0,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
shlb 485498096 

// CHECK: shlb 64(%rdx) 
// CHECK: encoding: [0xd0,0x62,0x40]         
shlb 64(%rdx) 

// CHECK: shlb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x64,0x82,0x40]         
shlb 64(%rdx,%rax,4) 

// CHECK: shlb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x64,0x82,0xc0]         
shlb -64(%rdx,%rax,4) 

// CHECK: shlb 64(%rdx,%rax) 
// CHECK: encoding: [0xd0,0x64,0x02,0x40]         
shlb 64(%rdx,%rax) 

// CHECK: shlb %cl, 485498096 
// CHECK: encoding: [0xd2,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
shlb %cl, 485498096 

// CHECK: shlb %cl, 64(%rdx) 
// CHECK: encoding: [0xd2,0x62,0x40]        
shlb %cl, 64(%rdx) 

// CHECK: shlb %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x64,0x82,0x40]        
shlb %cl, 64(%rdx,%rax,4) 

// CHECK: shlb %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x64,0x82,0xc0]        
shlb %cl, -64(%rdx,%rax,4) 

// CHECK: shlb %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd2,0x64,0x02,0x40]        
shlb %cl, 64(%rdx,%rax) 

// CHECK: shlb %cl, %r14b 
// CHECK: encoding: [0x41,0xd2,0xe6]        
shlb %cl, %r14b 

// CHECK: shlb %cl, (%rdx) 
// CHECK: encoding: [0xd2,0x22]        
shlb %cl, (%rdx) 

// CHECK: shlb %r14b 
// CHECK: encoding: [0x41,0xd0,0xe6]         
shlb %r14b 

// CHECK: shlb (%rdx) 
// CHECK: encoding: [0xd0,0x22]         
shlb (%rdx) 

// CHECK: shll 485498096 
// CHECK: encoding: [0xd1,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
shll 485498096 

// CHECK: shll 64(%rdx) 
// CHECK: encoding: [0xd1,0x62,0x40]         
shll 64(%rdx) 

// CHECK: shll 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x64,0x82,0x40]         
shll 64(%rdx,%rax,4) 

// CHECK: shll -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x64,0x82,0xc0]         
shll -64(%rdx,%rax,4) 

// CHECK: shll 64(%rdx,%rax) 
// CHECK: encoding: [0xd1,0x64,0x02,0x40]         
shll 64(%rdx,%rax) 

// CHECK: shll %cl, 485498096 
// CHECK: encoding: [0xd3,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
shll %cl, 485498096 

// CHECK: shll %cl, 64(%rdx) 
// CHECK: encoding: [0xd3,0x62,0x40]        
shll %cl, 64(%rdx) 

// CHECK: shll %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x64,0x82,0x40]        
shll %cl, 64(%rdx,%rax,4) 

// CHECK: shll %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x64,0x82,0xc0]        
shll %cl, -64(%rdx,%rax,4) 

// CHECK: shll %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd3,0x64,0x02,0x40]        
shll %cl, 64(%rdx,%rax) 

// CHECK: shll %cl, %r13d 
// CHECK: encoding: [0x41,0xd3,0xe5]        
shll %cl, %r13d 

// CHECK: shll %cl, (%rdx) 
// CHECK: encoding: [0xd3,0x22]        
shll %cl, (%rdx) 

// CHECK: shll %r13d 
// CHECK: encoding: [0x41,0xd1,0xe5]         
shll %r13d 

// CHECK: shll (%rdx) 
// CHECK: encoding: [0xd1,0x22]         
shll (%rdx) 

// CHECK: shlq 485498096 
// CHECK: encoding: [0x48,0xd1,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
shlq 485498096 

// CHECK: shlq 64(%rdx) 
// CHECK: encoding: [0x48,0xd1,0x62,0x40]         
shlq 64(%rdx) 

// CHECK: shlq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x64,0x82,0x40]         
shlq 64(%rdx,%rax,4) 

// CHECK: shlq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x64,0x82,0xc0]         
shlq -64(%rdx,%rax,4) 

// CHECK: shlq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd1,0x64,0x02,0x40]         
shlq 64(%rdx,%rax) 

// CHECK: shlq %cl, 485498096 
// CHECK: encoding: [0x48,0xd3,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
shlq %cl, 485498096 

// CHECK: shlq %cl, 64(%rdx) 
// CHECK: encoding: [0x48,0xd3,0x62,0x40]        
shlq %cl, 64(%rdx) 

// CHECK: shlq %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x64,0x82,0x40]        
shlq %cl, 64(%rdx,%rax,4) 

// CHECK: shlq %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x64,0x82,0xc0]        
shlq %cl, -64(%rdx,%rax,4) 

// CHECK: shlq %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd3,0x64,0x02,0x40]        
shlq %cl, 64(%rdx,%rax) 

// CHECK: shlq %cl, (%rdx) 
// CHECK: encoding: [0x48,0xd3,0x22]        
shlq %cl, (%rdx) 

// CHECK: shlq (%rdx) 
// CHECK: encoding: [0x48,0xd1,0x22]         
shlq (%rdx) 

// CHECK: shlw 485498096 
// CHECK: encoding: [0x66,0xd1,0x24,0x25,0xf0,0x1c,0xf0,0x1c]         
shlw 485498096 

// CHECK: shlw 64(%rdx) 
// CHECK: encoding: [0x66,0xd1,0x62,0x40]         
shlw 64(%rdx) 

// CHECK: shlw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x64,0x82,0x40]         
shlw 64(%rdx,%rax,4) 

// CHECK: shlw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x64,0x82,0xc0]         
shlw -64(%rdx,%rax,4) 

// CHECK: shlw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd1,0x64,0x02,0x40]         
shlw 64(%rdx,%rax) 

// CHECK: shlw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
shlw %cl, 485498096 

// CHECK: shlw %cl, 64(%rdx) 
// CHECK: encoding: [0x66,0xd3,0x62,0x40]        
shlw %cl, 64(%rdx) 

// CHECK: shlw %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x64,0x82,0x40]        
shlw %cl, 64(%rdx,%rax,4) 

// CHECK: shlw %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x64,0x82,0xc0]        
shlw %cl, -64(%rdx,%rax,4) 

// CHECK: shlw %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd3,0x64,0x02,0x40]        
shlw %cl, 64(%rdx,%rax) 

// CHECK: shlw %cl, %r14w 
// CHECK: encoding: [0x66,0x41,0xd3,0xe6]        
shlw %cl, %r14w 

// CHECK: shlw %cl, (%rdx) 
// CHECK: encoding: [0x66,0xd3,0x22]        
shlw %cl, (%rdx) 

// CHECK: shlw %r14w 
// CHECK: encoding: [0x66,0x41,0xd1,0xe6]         
shlw %r14w 

// CHECK: shlw (%rdx) 
// CHECK: encoding: [0x66,0xd1,0x22]         
shlw (%rdx) 

// CHECK: shrb 485498096 
// CHECK: encoding: [0xd0,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
shrb 485498096 

// CHECK: shrb 64(%rdx) 
// CHECK: encoding: [0xd0,0x6a,0x40]         
shrb 64(%rdx) 

// CHECK: shrb 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x6c,0x82,0x40]         
shrb 64(%rdx,%rax,4) 

// CHECK: shrb -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd0,0x6c,0x82,0xc0]         
shrb -64(%rdx,%rax,4) 

// CHECK: shrb 64(%rdx,%rax) 
// CHECK: encoding: [0xd0,0x6c,0x02,0x40]         
shrb 64(%rdx,%rax) 

// CHECK: shrb %cl, 485498096 
// CHECK: encoding: [0xd2,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
shrb %cl, 485498096 

// CHECK: shrb %cl, 64(%rdx) 
// CHECK: encoding: [0xd2,0x6a,0x40]        
shrb %cl, 64(%rdx) 

// CHECK: shrb %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x6c,0x82,0x40]        
shrb %cl, 64(%rdx,%rax,4) 

// CHECK: shrb %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd2,0x6c,0x82,0xc0]        
shrb %cl, -64(%rdx,%rax,4) 

// CHECK: shrb %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd2,0x6c,0x02,0x40]        
shrb %cl, 64(%rdx,%rax) 

// CHECK: shrb %cl, %r14b 
// CHECK: encoding: [0x41,0xd2,0xee]        
shrb %cl, %r14b 

// CHECK: shrb %cl, (%rdx) 
// CHECK: encoding: [0xd2,0x2a]        
shrb %cl, (%rdx) 

// CHECK: shrb %r14b 
// CHECK: encoding: [0x41,0xd0,0xee]         
shrb %r14b 

// CHECK: shrb (%rdx) 
// CHECK: encoding: [0xd0,0x2a]         
shrb (%rdx) 

// CHECK: shrl 485498096 
// CHECK: encoding: [0xd1,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
shrl 485498096 

// CHECK: shrl 64(%rdx) 
// CHECK: encoding: [0xd1,0x6a,0x40]         
shrl 64(%rdx) 

// CHECK: shrl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x6c,0x82,0x40]         
shrl 64(%rdx,%rax,4) 

// CHECK: shrl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd1,0x6c,0x82,0xc0]         
shrl -64(%rdx,%rax,4) 

// CHECK: shrl 64(%rdx,%rax) 
// CHECK: encoding: [0xd1,0x6c,0x02,0x40]         
shrl 64(%rdx,%rax) 

// CHECK: shrl %cl, 485498096 
// CHECK: encoding: [0xd3,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
shrl %cl, 485498096 

// CHECK: shrl %cl, 64(%rdx) 
// CHECK: encoding: [0xd3,0x6a,0x40]        
shrl %cl, 64(%rdx) 

// CHECK: shrl %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x6c,0x82,0x40]        
shrl %cl, 64(%rdx,%rax,4) 

// CHECK: shrl %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd3,0x6c,0x82,0xc0]        
shrl %cl, -64(%rdx,%rax,4) 

// CHECK: shrl %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0xd3,0x6c,0x02,0x40]        
shrl %cl, 64(%rdx,%rax) 

// CHECK: shrl %cl, %r13d 
// CHECK: encoding: [0x41,0xd3,0xed]        
shrl %cl, %r13d 

// CHECK: shrl %cl, (%rdx) 
// CHECK: encoding: [0xd3,0x2a]        
shrl %cl, (%rdx) 

// CHECK: shrl %r13d 
// CHECK: encoding: [0x41,0xd1,0xed]         
shrl %r13d 

// CHECK: shrl (%rdx) 
// CHECK: encoding: [0xd1,0x2a]         
shrl (%rdx) 

// CHECK: shrq 485498096 
// CHECK: encoding: [0x48,0xd1,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
shrq 485498096 

// CHECK: shrq 64(%rdx) 
// CHECK: encoding: [0x48,0xd1,0x6a,0x40]         
shrq 64(%rdx) 

// CHECK: shrq 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x6c,0x82,0x40]         
shrq 64(%rdx,%rax,4) 

// CHECK: shrq -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd1,0x6c,0x82,0xc0]         
shrq -64(%rdx,%rax,4) 

// CHECK: shrq 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd1,0x6c,0x02,0x40]         
shrq 64(%rdx,%rax) 

// CHECK: shrq %cl, 485498096 
// CHECK: encoding: [0x48,0xd3,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
shrq %cl, 485498096 

// CHECK: shrq %cl, 64(%rdx) 
// CHECK: encoding: [0x48,0xd3,0x6a,0x40]        
shrq %cl, 64(%rdx) 

// CHECK: shrq %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x6c,0x82,0x40]        
shrq %cl, 64(%rdx,%rax,4) 

// CHECK: shrq %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xd3,0x6c,0x82,0xc0]        
shrq %cl, -64(%rdx,%rax,4) 

// CHECK: shrq %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xd3,0x6c,0x02,0x40]        
shrq %cl, 64(%rdx,%rax) 

// CHECK: shrq %cl, (%rdx) 
// CHECK: encoding: [0x48,0xd3,0x2a]        
shrq %cl, (%rdx) 

// CHECK: shrq (%rdx) 
// CHECK: encoding: [0x48,0xd1,0x2a]         
shrq (%rdx) 

// CHECK: shrw 485498096 
// CHECK: encoding: [0x66,0xd1,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]         
shrw 485498096 

// CHECK: shrw 64(%rdx) 
// CHECK: encoding: [0x66,0xd1,0x6a,0x40]         
shrw 64(%rdx) 

// CHECK: shrw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x6c,0x82,0x40]         
shrw 64(%rdx,%rax,4) 

// CHECK: shrw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd1,0x6c,0x82,0xc0]         
shrw -64(%rdx,%rax,4) 

// CHECK: shrw 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd1,0x6c,0x02,0x40]         
shrw 64(%rdx,%rax) 

// CHECK: shrw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
shrw %cl, 485498096 

// CHECK: shrw %cl, 64(%rdx) 
// CHECK: encoding: [0x66,0xd3,0x6a,0x40]        
shrw %cl, 64(%rdx) 

// CHECK: shrw %cl, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x6c,0x82,0x40]        
shrw %cl, 64(%rdx,%rax,4) 

// CHECK: shrw %cl, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xd3,0x6c,0x82,0xc0]        
shrw %cl, -64(%rdx,%rax,4) 

// CHECK: shrw %cl, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xd3,0x6c,0x02,0x40]        
shrw %cl, 64(%rdx,%rax) 

// CHECK: shrw %cl, %r14w 
// CHECK: encoding: [0x66,0x41,0xd3,0xee]        
shrw %cl, %r14w 

// CHECK: shrw %cl, (%rdx) 
// CHECK: encoding: [0x66,0xd3,0x2a]        
shrw %cl, (%rdx) 

// CHECK: shrw %r14w 
// CHECK: encoding: [0x66,0x41,0xd1,0xee]         
shrw %r14w 

// CHECK: shrw (%rdx) 
// CHECK: encoding: [0x66,0xd1,0x2a]         
shrw (%rdx) 

// CHECK: stc 
// CHECK: encoding: [0xf9]          
stc 

// CHECK: std 
// CHECK: encoding: [0xfd]          
std 

// CHECK: sti 
// CHECK: encoding: [0xfb]          
sti 

// CHECK: stosb %al, %es:(%rdi) 
// CHECK: encoding: [0xaa]        
stosb %al, %es:(%rdi) 

// CHECK: stosw %ax, %es:(%rdi) 
// CHECK: encoding: [0x66,0xab]        
stosw %ax, %es:(%rdi) 

// CHECK: subb $0, 485498096 
// CHECK: encoding: [0x80,0x2c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
subb $0, 485498096 

// CHECK: subb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x6a,0x40,0x00]        
subb $0, 64(%rdx) 

// CHECK: subb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x6c,0x82,0x40,0x00]        
subb $0, 64(%rdx,%rax,4) 

// CHECK: subb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x6c,0x82,0xc0,0x00]        
subb $0, -64(%rdx,%rax,4) 

// CHECK: subb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x6c,0x02,0x40,0x00]        
subb $0, 64(%rdx,%rax) 

// CHECK: subb $0, %al 
// CHECK: encoding: [0x2c,0x00]        
subb $0, %al 

// CHECK: subb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xee,0x00]        
subb $0, %r14b 

// CHECK: subb $0, (%rdx) 
// CHECK: encoding: [0x80,0x2a,0x00]        
subb $0, (%rdx) 

// CHECK: subb 485498096, %r14b 
// CHECK: encoding: [0x44,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
subb 485498096, %r14b 

// CHECK: subb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x2a,0x72,0x40]        
subb 64(%rdx), %r14b 

// CHECK: subb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x2a,0x74,0x82,0x40]        
subb 64(%rdx,%rax,4), %r14b 

// CHECK: subb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x2a,0x74,0x82,0xc0]        
subb -64(%rdx,%rax,4), %r14b 

// CHECK: subb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x2a,0x74,0x02,0x40]        
subb 64(%rdx,%rax), %r14b 

// CHECK: subb %r14b, 485498096 
// CHECK: encoding: [0x44,0x28,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
subb %r14b, 485498096 

// CHECK: subb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x28,0x72,0x40]        
subb %r14b, 64(%rdx) 

// CHECK: subb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x28,0x74,0x82,0x40]        
subb %r14b, 64(%rdx,%rax,4) 

// CHECK: subb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x28,0x74,0x82,0xc0]        
subb %r14b, -64(%rdx,%rax,4) 

// CHECK: subb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x28,0x74,0x02,0x40]        
subb %r14b, 64(%rdx,%rax) 

// CHECK: subb %r14b, %r14b 
// CHECK: encoding: [0x45,0x28,0xf6]        
subb %r14b, %r14b 

// CHECK: subb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x28,0x32]        
subb %r14b, (%rdx) 

// CHECK: subb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x2a,0x32]        
subb (%rdx), %r14b 

// CHECK: subl $0, %eax 
// CHECK: encoding: [0x83,0xe8,0x00]        
subl $0, %eax 

// CHECK: subl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xed,0x00]        
subl $0, %r13d 

// CHECK: subl %r13d, %r13d 
// CHECK: encoding: [0x45,0x29,0xed]        
subl %r13d, %r13d 

// CHECK: testb $0, 485498096 
// CHECK: encoding: [0xf6,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
testb $0, 485498096 

// CHECK: testb $0, 64(%rdx) 
// CHECK: encoding: [0xf6,0x42,0x40,0x00]        
testb $0, 64(%rdx) 

// CHECK: testb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x44,0x82,0x40,0x00]        
testb $0, 64(%rdx,%rax,4) 

// CHECK: testb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf6,0x44,0x82,0xc0,0x00]        
testb $0, -64(%rdx,%rax,4) 

// CHECK: testb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xf6,0x44,0x02,0x40,0x00]        
testb $0, 64(%rdx,%rax) 

// CHECK: testb $0, %al 
// CHECK: encoding: [0xa8,0x00]        
testb $0, %al 

// CHECK: testb $0, %r14b 
// CHECK: encoding: [0x41,0xf6,0xc6,0x00]        
testb $0, %r14b 

// CHECK: testb $0, (%rdx) 
// CHECK: encoding: [0xf6,0x02,0x00]        
testb $0, (%rdx) 

// CHECK: testb %r14b, 485498096 
// CHECK: encoding: [0x44,0x84,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
testb %r14b, 485498096 

// CHECK: testb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x84,0x72,0x40]        
testb %r14b, 64(%rdx) 

// CHECK: testb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x84,0x74,0x82,0x40]        
testb %r14b, 64(%rdx,%rax,4) 

// CHECK: testb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x84,0x74,0x82,0xc0]        
testb %r14b, -64(%rdx,%rax,4) 

// CHECK: testb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x84,0x74,0x02,0x40]        
testb %r14b, 64(%rdx,%rax) 

// CHECK: testb %r14b, %r14b 
// CHECK: encoding: [0x45,0x84,0xf6]        
testb %r14b, %r14b 

// CHECK: testb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x84,0x32]        
testb %r14b, (%rdx) 

// CHECK: testl $0, 485498096 
// CHECK: encoding: [0xf7,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
testl $0, 485498096 

// CHECK: testl $0, 64(%rdx) 
// CHECK: encoding: [0xf7,0x42,0x40,0x00,0x00,0x00,0x00]        
testl $0, 64(%rdx) 

// CHECK: testl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x44,0x82,0x40,0x00,0x00,0x00,0x00]        
testl $0, 64(%rdx,%rax,4) 

// CHECK: testl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xf7,0x44,0x82,0xc0,0x00,0x00,0x00,0x00]        
testl $0, -64(%rdx,%rax,4) 

// CHECK: testl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0xf7,0x44,0x02,0x40,0x00,0x00,0x00,0x00]        
testl $0, 64(%rdx,%rax) 

// CHECK: testl $0, %eax 
// CHECK: encoding: [0xa9,0x00,0x00,0x00,0x00]        
testl $0, %eax 

// CHECK: testl $0, %r13d 
// CHECK: encoding: [0x41,0xf7,0xc5,0x00,0x00,0x00,0x00]        
testl $0, %r13d 

// CHECK: testl $0, (%rdx) 
// CHECK: encoding: [0xf7,0x02,0x00,0x00,0x00,0x00]        
testl $0, (%rdx) 

// CHECK: testl %r13d, %r13d 
// CHECK: encoding: [0x45,0x85,0xed]        
testl %r13d, %r13d 

// CHECK: testq $0, 485498096 
// CHECK: encoding: [0x48,0xf7,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
testq $0, 485498096 

// CHECK: testq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0xf7,0x42,0x40,0x00,0x00,0x00,0x00]        
testq $0, 64(%rdx) 

// CHECK: testq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x44,0x82,0x40,0x00,0x00,0x00,0x00]        
testq $0, 64(%rdx,%rax,4) 

// CHECK: testq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0xf7,0x44,0x82,0xc0,0x00,0x00,0x00,0x00]        
testq $0, -64(%rdx,%rax,4) 

// CHECK: testq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0xf7,0x44,0x02,0x40,0x00,0x00,0x00,0x00]        
testq $0, 64(%rdx,%rax) 

// CHECK: testq $0, (%rdx) 
// CHECK: encoding: [0x48,0xf7,0x02,0x00,0x00,0x00,0x00]        
testq $0, (%rdx) 

// CHECK: testw $0, 485498096 
// CHECK: encoding: [0x66,0xf7,0x04,0x25,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
testw $0, 485498096 

// CHECK: testw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0xf7,0x42,0x40,0x00,0x00]        
testw $0, 64(%rdx) 

// CHECK: testw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x44,0x82,0x40,0x00,0x00]        
testw $0, 64(%rdx,%rax,4) 

// CHECK: testw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0xf7,0x44,0x82,0xc0,0x00,0x00]        
testw $0, -64(%rdx,%rax,4) 

// CHECK: testw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0xf7,0x44,0x02,0x40,0x00,0x00]        
testw $0, 64(%rdx,%rax) 

// CHECK: testw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0xf7,0xc6,0x00,0x00]        
testw $0, %r14w 

// CHECK: testw $0, (%rdx) 
// CHECK: encoding: [0x66,0xf7,0x02,0x00,0x00]        
testw $0, (%rdx) 

// CHECK: testw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x85,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
testw %r14w, 485498096 

// CHECK: testw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x85,0x72,0x40]        
testw %r14w, 64(%rdx) 

// CHECK: testw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x85,0x74,0x82,0x40]        
testw %r14w, 64(%rdx,%rax,4) 

// CHECK: testw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x85,0x74,0x82,0xc0]        
testw %r14w, -64(%rdx,%rax,4) 

// CHECK: testw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x85,0x74,0x02,0x40]        
testw %r14w, 64(%rdx,%rax) 

// CHECK: testw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x85,0xf6]        
testw %r14w, %r14w 

// CHECK: testw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x85,0x32]        
testw %r14w, (%rdx) 

// CHECK: xchgb %r14b, 485498096 
// CHECK: encoding: [0x44,0x86,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xchgb %r14b, 485498096 

// CHECK: xchgb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x86,0x72,0x40]        
xchgb %r14b, 64(%rdx) 

// CHECK: xchgb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x86,0x74,0x82,0x40]        
xchgb %r14b, 64(%rdx,%rax,4) 

// CHECK: xchgb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x86,0x74,0x82,0xc0]        
xchgb %r14b, -64(%rdx,%rax,4) 

// CHECK: xchgb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x86,0x74,0x02,0x40]        
xchgb %r14b, 64(%rdx,%rax) 

// CHECK: xchgb %r14b, %r14b 
// CHECK: encoding: [0x45,0x86,0xf6]        
xchgb %r14b, %r14b 

// CHECK: xchgb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x86,0x32]        
xchgb %r14b, (%rdx) 

// CHECK: xchgl %r13d, %eax 
// CHECK: encoding: [0x41,0x95]        
xchgl %r13d, %eax 

// CHECK: xchgl %r13d, %r13d 
// CHECK: encoding: [0x45,0x87,0xed]        
xchgl %r13d, %r13d 

// CHECK: xchgl %r8d, %eax 
// CHECK: encoding: [0x41,0x90]        
xchgl %r8d, %eax 

// CHECK: xchgw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x87,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xchgw %r14w, 485498096 

// CHECK: xchgw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x87,0x72,0x40]        
xchgw %r14w, 64(%rdx) 

// CHECK: xchgw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x87,0x74,0x82,0x40]        
xchgw %r14w, 64(%rdx,%rax,4) 

// CHECK: xchgw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x87,0x74,0x82,0xc0]        
xchgw %r14w, -64(%rdx,%rax,4) 

// CHECK: xchgw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x87,0x74,0x02,0x40]        
xchgw %r14w, 64(%rdx,%rax) 

// CHECK: xchgw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x87,0xf6]        
xchgw %r14w, %r14w 

// CHECK: xchgw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x87,0x32]        
xchgw %r14w, (%rdx) 

// CHECK: xlatb 
// CHECK: encoding: [0xd7]          
xlatb 

// CHECK: xorb $0, 485498096 
// CHECK: encoding: [0x80,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
xorb $0, 485498096 

// CHECK: xorb $0, 64(%rdx) 
// CHECK: encoding: [0x80,0x72,0x40,0x00]        
xorb $0, 64(%rdx) 

// CHECK: xorb $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x74,0x82,0x40,0x00]        
xorb $0, 64(%rdx,%rax,4) 

// CHECK: xorb $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x80,0x74,0x82,0xc0,0x00]        
xorb $0, -64(%rdx,%rax,4) 

// CHECK: xorb $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x80,0x74,0x02,0x40,0x00]        
xorb $0, 64(%rdx,%rax) 

// CHECK: xorb $0, %al 
// CHECK: encoding: [0x34,0x00]        
xorb $0, %al 

// CHECK: xorb $0, %r14b 
// CHECK: encoding: [0x41,0x80,0xf6,0x00]        
xorb $0, %r14b 

// CHECK: xorb $0, (%rdx) 
// CHECK: encoding: [0x80,0x32,0x00]        
xorb $0, (%rdx) 

// CHECK: xorb 485498096, %r14b 
// CHECK: encoding: [0x44,0x32,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xorb 485498096, %r14b 

// CHECK: xorb 64(%rdx), %r14b 
// CHECK: encoding: [0x44,0x32,0x72,0x40]        
xorb 64(%rdx), %r14b 

// CHECK: xorb 64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x32,0x74,0x82,0x40]        
xorb 64(%rdx,%rax,4), %r14b 

// CHECK: xorb -64(%rdx,%rax,4), %r14b 
// CHECK: encoding: [0x44,0x32,0x74,0x82,0xc0]        
xorb -64(%rdx,%rax,4), %r14b 

// CHECK: xorb 64(%rdx,%rax), %r14b 
// CHECK: encoding: [0x44,0x32,0x74,0x02,0x40]        
xorb 64(%rdx,%rax), %r14b 

// CHECK: xorb %r14b, 485498096 
// CHECK: encoding: [0x44,0x30,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xorb %r14b, 485498096 

// CHECK: xorb %r14b, 64(%rdx) 
// CHECK: encoding: [0x44,0x30,0x72,0x40]        
xorb %r14b, 64(%rdx) 

// CHECK: xorb %r14b, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x30,0x74,0x82,0x40]        
xorb %r14b, 64(%rdx,%rax,4) 

// CHECK: xorb %r14b, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x44,0x30,0x74,0x82,0xc0]        
xorb %r14b, -64(%rdx,%rax,4) 

// CHECK: xorb %r14b, 64(%rdx,%rax) 
// CHECK: encoding: [0x44,0x30,0x74,0x02,0x40]        
xorb %r14b, 64(%rdx,%rax) 

// CHECK: xorb %r14b, %r14b 
// CHECK: encoding: [0x45,0x30,0xf6]        
xorb %r14b, %r14b 

// CHECK: xorb %r14b, (%rdx) 
// CHECK: encoding: [0x44,0x30,0x32]        
xorb %r14b, (%rdx) 

// CHECK: xorb (%rdx), %r14b 
// CHECK: encoding: [0x44,0x32,0x32]        
xorb (%rdx), %r14b 

// CHECK: xorl $0, 485498096 
// CHECK: encoding: [0x83,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
xorl $0, 485498096 

// CHECK: xorl $0, 64(%rdx) 
// CHECK: encoding: [0x83,0x72,0x40,0x00]        
xorl $0, 64(%rdx) 

// CHECK: xorl $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x74,0x82,0x40,0x00]        
xorl $0, 64(%rdx,%rax,4) 

// CHECK: xorl $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x83,0x74,0x82,0xc0,0x00]        
xorl $0, -64(%rdx,%rax,4) 

// CHECK: xorl $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x83,0x74,0x02,0x40,0x00]        
xorl $0, 64(%rdx,%rax) 

// CHECK: xorl $0, %eax 
// CHECK: encoding: [0x83,0xf0,0x00]        
xorl $0, %eax 

// CHECK: xorl $0, %r13d 
// CHECK: encoding: [0x41,0x83,0xf5,0x00]        
xorl $0, %r13d 

// CHECK: xorl $0, (%rdx) 
// CHECK: encoding: [0x83,0x32,0x00]        
xorl $0, (%rdx) 

// CHECK: xorl %r13d, %r13d 
// CHECK: encoding: [0x45,0x31,0xed]        
xorl %r13d, %r13d 

// CHECK: xorq $0, 485498096 
// CHECK: encoding: [0x48,0x83,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
xorq $0, 485498096 

// CHECK: xorq $0, 64(%rdx) 
// CHECK: encoding: [0x48,0x83,0x72,0x40,0x00]        
xorq $0, 64(%rdx) 

// CHECK: xorq $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x74,0x82,0x40,0x00]        
xorq $0, 64(%rdx,%rax,4) 

// CHECK: xorq $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x48,0x83,0x74,0x82,0xc0,0x00]        
xorq $0, -64(%rdx,%rax,4) 

// CHECK: xorq $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x48,0x83,0x74,0x02,0x40,0x00]        
xorq $0, 64(%rdx,%rax) 

// CHECK: xorq $0, (%rdx) 
// CHECK: encoding: [0x48,0x83,0x32,0x00]        
xorq $0, (%rdx) 

// CHECK: xorw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
xorw $0, 485498096 

// CHECK: xorw $0, 64(%rdx) 
// CHECK: encoding: [0x66,0x83,0x72,0x40,0x00]        
xorw $0, 64(%rdx) 

// CHECK: xorw $0, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x74,0x82,0x40,0x00]        
xorw $0, 64(%rdx,%rax,4) 

// CHECK: xorw $0, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x83,0x74,0x82,0xc0,0x00]        
xorw $0, -64(%rdx,%rax,4) 

// CHECK: xorw $0, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x83,0x74,0x02,0x40,0x00]        
xorw $0, 64(%rdx,%rax) 

// CHECK: xorw $0, %r14w 
// CHECK: encoding: [0x66,0x41,0x83,0xf6,0x00]        
xorw $0, %r14w 

// CHECK: xorw $0, (%rdx) 
// CHECK: encoding: [0x66,0x83,0x32,0x00]        
xorw $0, (%rdx) 

// CHECK: xorw 485498096, %r14w 
// CHECK: encoding: [0x66,0x44,0x33,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xorw 485498096, %r14w 

// CHECK: xorw 64(%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x33,0x72,0x40]        
xorw 64(%rdx), %r14w 

// CHECK: xorw 64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x33,0x74,0x82,0x40]        
xorw 64(%rdx,%rax,4), %r14w 

// CHECK: xorw -64(%rdx,%rax,4), %r14w 
// CHECK: encoding: [0x66,0x44,0x33,0x74,0x82,0xc0]        
xorw -64(%rdx,%rax,4), %r14w 

// CHECK: xorw 64(%rdx,%rax), %r14w 
// CHECK: encoding: [0x66,0x44,0x33,0x74,0x02,0x40]        
xorw 64(%rdx,%rax), %r14w 

// CHECK: xorw %r14w, 485498096 
// CHECK: encoding: [0x66,0x44,0x31,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
xorw %r14w, 485498096 

// CHECK: xorw %r14w, 64(%rdx) 
// CHECK: encoding: [0x66,0x44,0x31,0x72,0x40]        
xorw %r14w, 64(%rdx) 

// CHECK: xorw %r14w, 64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x31,0x74,0x82,0x40]        
xorw %r14w, 64(%rdx,%rax,4) 

// CHECK: xorw %r14w, -64(%rdx,%rax,4) 
// CHECK: encoding: [0x66,0x44,0x31,0x74,0x82,0xc0]        
xorw %r14w, -64(%rdx,%rax,4) 

// CHECK: xorw %r14w, 64(%rdx,%rax) 
// CHECK: encoding: [0x66,0x44,0x31,0x74,0x02,0x40]        
xorw %r14w, 64(%rdx,%rax) 

// CHECK: xorw %r14w, %r14w 
// CHECK: encoding: [0x66,0x45,0x31,0xf6]        
xorw %r14w, %r14w 

// CHECK: xorw %r14w, (%rdx) 
// CHECK: encoding: [0x66,0x44,0x31,0x32]        
xorw %r14w, (%rdx) 

// CHECK: xorw (%rdx), %r14w 
// CHECK: encoding: [0x66,0x44,0x33,0x32]        
xorw (%rdx), %r14w 

