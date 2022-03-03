// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: aaa 
// CHECK: encoding: [0x37]          
aaa 

// CHECK: aad $0 
// CHECK: encoding: [0xd5,0x00]         
aad $0 

// CHECK: aam $0 
// CHECK: encoding: [0xd4,0x00]         
aam $0 

// CHECK: aas 
// CHECK: encoding: [0x3f]          
aas 

// CHECK: adcb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
adcb $0, -485498096(%edx,%eax,4) 

// CHECK: adcb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
adcb $0, 485498096(%edx,%eax,4) 

// CHECK: adcb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0x92,0xf0,0x1c,0xf0,0x1c,0x00]        
adcb $0, 485498096(%edx) 

// CHECK: adcb $0, 485498096 
// CHECK: encoding: [0x80,0x15,0xf0,0x1c,0xf0,0x1c,0x00]        
adcb $0, 485498096 

// CHECK: adcb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x54,0x02,0x40,0x00]        
adcb $0, 64(%edx,%eax) 

// CHECK: adcb $0, %al 
// CHECK: encoding: [0x14,0x00]        
adcb $0, %al 

// CHECK: adcb $0, (%edx) 
// CHECK: encoding: [0x80,0x12,0x00]        
adcb $0, (%edx) 

// CHECK: adcl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
adcl $0, -485498096(%edx,%eax,4) 

// CHECK: adcl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
adcl $0, 485498096(%edx,%eax,4) 

// CHECK: adcl $0, 485498096(%edx) 
// CHECK: encoding: [0x83,0x92,0xf0,0x1c,0xf0,0x1c,0x00]        
adcl $0, 485498096(%edx) 

// CHECK: adcl $0, 485498096 
// CHECK: encoding: [0x83,0x15,0xf0,0x1c,0xf0,0x1c,0x00]        
adcl $0, 485498096 

// CHECK: adcl $0, 64(%edx,%eax) 
// CHECK: encoding: [0x83,0x54,0x02,0x40,0x00]        
adcl $0, 64(%edx,%eax) 

// CHECK: adcl $0, %eax 
// CHECK: encoding: [0x83,0xd0,0x00]        
adcl $0, %eax 

// CHECK: adcl $0, (%edx) 
// CHECK: encoding: [0x83,0x12,0x00]        
adcl $0, (%edx) 

// CHECK: adcl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x13,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
adcl 3809469200(%edx,%eax,4), %eax 

// CHECK: adcl 485498096, %eax 
// CHECK: encoding: [0x13,0x05,0xf0,0x1c,0xf0,0x1c]        
adcl 485498096, %eax 

// CHECK: adcl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x13,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
adcl 485498096(%edx,%eax,4), %eax 

// CHECK: adcl 485498096(%edx), %eax 
// CHECK: encoding: [0x13,0x82,0xf0,0x1c,0xf0,0x1c]        
adcl 485498096(%edx), %eax 

// CHECK: adcl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x13,0x44,0x02,0x40]        
adcl 64(%edx,%eax), %eax 

// CHECK: adcl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x11,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
adcl %eax, 3809469200(%edx,%eax,4) 

// CHECK: adcl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x11,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
adcl %eax, 485498096(%edx,%eax,4) 

// CHECK: adcl %eax, 485498096(%edx) 
// CHECK: encoding: [0x11,0x82,0xf0,0x1c,0xf0,0x1c]        
adcl %eax, 485498096(%edx) 

// CHECK: adcl %eax, 485498096 
// CHECK: encoding: [0x11,0x05,0xf0,0x1c,0xf0,0x1c]        
adcl %eax, 485498096 

// CHECK: adcl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x11,0x44,0x02,0x40]        
adcl %eax, 64(%edx,%eax) 

// CHECK: adcl %eax, %eax 
// CHECK: encoding: [0x11,0xc0]        
adcl %eax, %eax 

// CHECK: adcl %eax, (%edx) 
// CHECK: encoding: [0x11,0x02]        
adcl %eax, (%edx) 

// CHECK: adcl (%edx), %eax 
// CHECK: encoding: [0x13,0x02]        
adcl (%edx), %eax 

// CHECK: adcw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
adcw $0, -485498096(%edx,%eax,4) 

// CHECK: adcw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
adcw $0, 485498096(%edx,%eax,4) 

// CHECK: adcw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0x83,0x92,0xf0,0x1c,0xf0,0x1c,0x00]        
adcw $0, 485498096(%edx) 

// CHECK: adcw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x15,0xf0,0x1c,0xf0,0x1c,0x00]        
adcw $0, 485498096 

// CHECK: adcw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0x83,0x54,0x02,0x40,0x00]        
adcw $0, 64(%edx,%eax) 

// CHECK: adcw $0, (%edx) 
// CHECK: encoding: [0x66,0x83,0x12,0x00]        
adcw $0, (%edx) 

// CHECK: addb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
addb $0, -485498096(%edx,%eax,4) 

// CHECK: addb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
addb $0, 485498096(%edx,%eax,4) 

// CHECK: addb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
addb $0, 485498096(%edx) 

// CHECK: addb $0, 485498096 
// CHECK: encoding: [0x80,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
addb $0, 485498096 

// CHECK: addb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x44,0x02,0x40,0x00]        
addb $0, 64(%edx,%eax) 

// CHECK: addb $0, %al 
// CHECK: encoding: [0x04,0x00]        
addb $0, %al 

// CHECK: addb $0, (%edx) 
// CHECK: encoding: [0x80,0x02,0x00]        
addb $0, (%edx) 

// CHECK: addl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
addl $0, -485498096(%edx,%eax,4) 

// CHECK: addl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
addl $0, 485498096(%edx,%eax,4) 

// CHECK: addl $0, 485498096(%edx) 
// CHECK: encoding: [0x83,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
addl $0, 485498096(%edx) 

// CHECK: addl $0, 485498096 
// CHECK: encoding: [0x83,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
addl $0, 485498096 

// CHECK: addl $0, 64(%edx,%eax) 
// CHECK: encoding: [0x83,0x44,0x02,0x40,0x00]        
addl $0, 64(%edx,%eax) 

// CHECK: addl $0, %eax 
// CHECK: encoding: [0x83,0xc0,0x00]        
addl $0, %eax 

// CHECK: addl $0, (%edx) 
// CHECK: encoding: [0x83,0x02,0x00]        
addl $0, (%edx) 

// CHECK: addl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x03,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
addl 3809469200(%edx,%eax,4), %eax 

// CHECK: addl 485498096, %eax 
// CHECK: encoding: [0x03,0x05,0xf0,0x1c,0xf0,0x1c]        
addl 485498096, %eax 

// CHECK: addl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x03,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
addl 485498096(%edx,%eax,4), %eax 

// CHECK: addl 485498096(%edx), %eax 
// CHECK: encoding: [0x03,0x82,0xf0,0x1c,0xf0,0x1c]        
addl 485498096(%edx), %eax 

// CHECK: addl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x03,0x44,0x02,0x40]        
addl 64(%edx,%eax), %eax 

// CHECK: addl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x01,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
addl %eax, 3809469200(%edx,%eax,4) 

// CHECK: addl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x01,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
addl %eax, 485498096(%edx,%eax,4) 

// CHECK: addl %eax, 485498096(%edx) 
// CHECK: encoding: [0x01,0x82,0xf0,0x1c,0xf0,0x1c]        
addl %eax, 485498096(%edx) 

// CHECK: addl %eax, 485498096 
// CHECK: encoding: [0x01,0x05,0xf0,0x1c,0xf0,0x1c]        
addl %eax, 485498096 

// CHECK: addl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x01,0x44,0x02,0x40]        
addl %eax, 64(%edx,%eax) 

// CHECK: addl %eax, %eax 
// CHECK: encoding: [0x01,0xc0]        
addl %eax, %eax 

// CHECK: addl %eax, (%edx) 
// CHECK: encoding: [0x01,0x02]        
addl %eax, (%edx) 

// CHECK: addl (%edx), %eax 
// CHECK: encoding: [0x03,0x02]        
addl (%edx), %eax 

// CHECK: addw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
addw $0, -485498096(%edx,%eax,4) 

// CHECK: addw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
addw $0, 485498096(%edx,%eax,4) 

// CHECK: addw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0x83,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
addw $0, 485498096(%edx) 

// CHECK: addw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
addw $0, 485498096 

// CHECK: addw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0x83,0x44,0x02,0x40,0x00]        
addw $0, 64(%edx,%eax) 

// CHECK: addw $0, (%edx) 
// CHECK: encoding: [0x66,0x83,0x02,0x00]        
addw $0, (%edx) 

// CHECK: andb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
andb $0, -485498096(%edx,%eax,4) 

// CHECK: andb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
andb $0, 485498096(%edx,%eax,4) 

// CHECK: andb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]        
andb $0, 485498096(%edx) 

// CHECK: andb $0, 485498096 
// CHECK: encoding: [0x80,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
andb $0, 485498096 

// CHECK: andb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x64,0x02,0x40,0x00]        
andb $0, 64(%edx,%eax) 

// CHECK: andb $0, %al 
// CHECK: encoding: [0x24,0x00]        
andb $0, %al 

// CHECK: andb $0, (%edx) 
// CHECK: encoding: [0x80,0x22,0x00]        
andb $0, (%edx) 

// CHECK: andl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
andl $0, -485498096(%edx,%eax,4) 

// CHECK: andl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
andl $0, 485498096(%edx,%eax,4) 

// CHECK: andl $0, 485498096(%edx) 
// CHECK: encoding: [0x83,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]        
andl $0, 485498096(%edx) 

// CHECK: andl $0, 485498096 
// CHECK: encoding: [0x83,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
andl $0, 485498096 

// CHECK: andl $0, 64(%edx,%eax) 
// CHECK: encoding: [0x83,0x64,0x02,0x40,0x00]        
andl $0, 64(%edx,%eax) 

// CHECK: andl $0, %eax 
// CHECK: encoding: [0x83,0xe0,0x00]        
andl $0, %eax 

// CHECK: andl $0, (%edx) 
// CHECK: encoding: [0x83,0x22,0x00]        
andl $0, (%edx) 

// CHECK: andl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x23,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
andl 3809469200(%edx,%eax,4), %eax 

// CHECK: andl 485498096, %eax 
// CHECK: encoding: [0x23,0x05,0xf0,0x1c,0xf0,0x1c]        
andl 485498096, %eax 

// CHECK: andl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x23,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
andl 485498096(%edx,%eax,4), %eax 

// CHECK: andl 485498096(%edx), %eax 
// CHECK: encoding: [0x23,0x82,0xf0,0x1c,0xf0,0x1c]        
andl 485498096(%edx), %eax 

// CHECK: andl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x23,0x44,0x02,0x40]        
andl 64(%edx,%eax), %eax 

// CHECK: andl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x21,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
andl %eax, 3809469200(%edx,%eax,4) 

// CHECK: andl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x21,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
andl %eax, 485498096(%edx,%eax,4) 

// CHECK: andl %eax, 485498096(%edx) 
// CHECK: encoding: [0x21,0x82,0xf0,0x1c,0xf0,0x1c]        
andl %eax, 485498096(%edx) 

// CHECK: andl %eax, 485498096 
// CHECK: encoding: [0x21,0x05,0xf0,0x1c,0xf0,0x1c]        
andl %eax, 485498096 

// CHECK: andl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x21,0x44,0x02,0x40]        
andl %eax, 64(%edx,%eax) 

// CHECK: andl %eax, %eax 
// CHECK: encoding: [0x21,0xc0]        
andl %eax, %eax 

// CHECK: andl %eax, (%edx) 
// CHECK: encoding: [0x21,0x02]        
andl %eax, (%edx) 

// CHECK: andl (%edx), %eax 
// CHECK: encoding: [0x23,0x02]        
andl (%edx), %eax 

// CHECK: andw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
andw $0, -485498096(%edx,%eax,4) 

// CHECK: andw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
andw $0, 485498096(%edx,%eax,4) 

// CHECK: andw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0x83,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]        
andw $0, 485498096(%edx) 

// CHECK: andw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x25,0xf0,0x1c,0xf0,0x1c,0x00]        
andw $0, 485498096 

// CHECK: andw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0x83,0x64,0x02,0x40,0x00]        
andw $0, 64(%edx,%eax) 

// CHECK: andw $0, (%edx) 
// CHECK: encoding: [0x66,0x83,0x22,0x00]        
andw $0, (%edx) 

// CHECK: calll 64 
// CHECK: encoding: [0xe8,A,A,A,A]         
calll 64 

// CHECK: cbtw 
// CHECK: encoding: [0x66,0x98]          
cbtw 

// CHECK: cwtl 
// CHECK: encoding: [0x98]          
cwtl 

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

// CHECK: cmc 
// CHECK: encoding: [0xf5]          
cmc 

// CHECK: cmpb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xbc,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
cmpb $0, -485498096(%edx,%eax,4) 

// CHECK: cmpb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xbc,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpb $0, 485498096(%edx,%eax,4) 

// CHECK: cmpb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0xba,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpb $0, 485498096(%edx) 

// CHECK: cmpb $0, 485498096 
// CHECK: encoding: [0x80,0x3d,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpb $0, 485498096 

// CHECK: cmpb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x7c,0x02,0x40,0x00]        
cmpb $0, 64(%edx,%eax) 

// CHECK: cmpb $0, %al 
// CHECK: encoding: [0x3c,0x00]        
cmpb $0, %al 

// CHECK: cmpb $0, (%edx) 
// CHECK: encoding: [0x80,0x3a,0x00]        
cmpb $0, (%edx) 

// CHECK: cmpl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0xbc,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
cmpl $0, -485498096(%edx,%eax,4) 

// CHECK: cmpl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0xbc,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpl $0, 485498096(%edx,%eax,4) 

// CHECK: cmpl $0, 485498096(%edx) 
// CHECK: encoding: [0x83,0xba,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpl $0, 485498096(%edx) 

// CHECK: cmpl $0, 485498096 
// CHECK: encoding: [0x83,0x3d,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpl $0, 485498096 

// CHECK: cmpl $0, 64(%edx,%eax) 
// CHECK: encoding: [0x83,0x7c,0x02,0x40,0x00]        
cmpl $0, 64(%edx,%eax) 

// CHECK: cmpl $0, %eax 
// CHECK: encoding: [0x83,0xf8,0x00]        
cmpl $0, %eax 

// CHECK: cmpl $0, (%edx) 
// CHECK: encoding: [0x83,0x3a,0x00]        
cmpl $0, (%edx) 

// CHECK: cmpl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x3b,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
cmpl 3809469200(%edx,%eax,4), %eax 

// CHECK: cmpl 485498096, %eax 
// CHECK: encoding: [0x3b,0x05,0xf0,0x1c,0xf0,0x1c]        
cmpl 485498096, %eax 

// CHECK: cmpl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x3b,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
cmpl 485498096(%edx,%eax,4), %eax 

// CHECK: cmpl 485498096(%edx), %eax 
// CHECK: encoding: [0x3b,0x82,0xf0,0x1c,0xf0,0x1c]        
cmpl 485498096(%edx), %eax 

// CHECK: cmpl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x3b,0x44,0x02,0x40]        
cmpl 64(%edx,%eax), %eax 

// CHECK: cmpl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x39,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
cmpl %eax, 3809469200(%edx,%eax,4) 

// CHECK: cmpl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x39,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
cmpl %eax, 485498096(%edx,%eax,4) 

// CHECK: cmpl %eax, 485498096(%edx) 
// CHECK: encoding: [0x39,0x82,0xf0,0x1c,0xf0,0x1c]        
cmpl %eax, 485498096(%edx) 

// CHECK: cmpl %eax, 485498096 
// CHECK: encoding: [0x39,0x05,0xf0,0x1c,0xf0,0x1c]        
cmpl %eax, 485498096 

// CHECK: cmpl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x39,0x44,0x02,0x40]        
cmpl %eax, 64(%edx,%eax) 

// CHECK: cmpl %eax, %eax 
// CHECK: encoding: [0x39,0xc0]        
cmpl %eax, %eax 

// CHECK: cmpl %eax, (%edx) 
// CHECK: encoding: [0x39,0x02]        
cmpl %eax, (%edx) 

// CHECK: cmpl (%edx), %eax 
// CHECK: encoding: [0x3b,0x02]        
cmpl (%edx), %eax 

// CHECK: cmpsb %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0x26,0xa6]        
cmpsb %es:(%edi), %es:(%esi) 

// CHECK: cmpsl %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0x26,0xa7]        
cmpsl %es:(%edi), %es:(%esi) 

// CHECK: cmpsw %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0x26,0x66,0xa7]        
cmpsw %es:(%edi), %es:(%esi) 

// CHECK: cmpw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0xbc,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
cmpw $0, -485498096(%edx,%eax,4) 

// CHECK: cmpw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0xbc,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpw $0, 485498096(%edx,%eax,4) 

// CHECK: cmpw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0x83,0xba,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpw $0, 485498096(%edx) 

// CHECK: cmpw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x3d,0xf0,0x1c,0xf0,0x1c,0x00]        
cmpw $0, 485498096 

// CHECK: cmpw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0x83,0x7c,0x02,0x40,0x00]        
cmpw $0, 64(%edx,%eax) 

// CHECK: cmpw $0, (%edx) 
// CHECK: encoding: [0x66,0x83,0x3a,0x00]        
cmpw $0, (%edx) 

// CHECK: cwtd 
// CHECK: encoding: [0x66,0x99]          
cwtd 

// CHECK: daa 
// CHECK: encoding: [0x27]          
daa 

// CHECK: das 
// CHECK: encoding: [0x2f]          
das 

// CHECK: decb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xfe,0x8c,0x82,0x10,0xe3,0x0f,0xe3]         
decb -485498096(%edx,%eax,4) 

// CHECK: decb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xfe,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]         
decb 485498096(%edx,%eax,4) 

// CHECK: decb 485498096(%edx) 
// CHECK: encoding: [0xfe,0x8a,0xf0,0x1c,0xf0,0x1c]         
decb 485498096(%edx) 

// CHECK: decb 485498096 
// CHECK: encoding: [0xfe,0x0d,0xf0,0x1c,0xf0,0x1c]         
decb 485498096 

// CHECK: decb 64(%edx,%eax) 
// CHECK: encoding: [0xfe,0x4c,0x02,0x40]         
decb 64(%edx,%eax) 

// CHECK: decb (%edx) 
// CHECK: encoding: [0xfe,0x0a]         
decb (%edx) 

// CHECK: decl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xff,0x8c,0x82,0x10,0xe3,0x0f,0xe3]         
decl -485498096(%edx,%eax,4) 

// CHECK: decl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xff,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]         
decl 485498096(%edx,%eax,4) 

// CHECK: decl 485498096(%edx) 
// CHECK: encoding: [0xff,0x8a,0xf0,0x1c,0xf0,0x1c]         
decl 485498096(%edx) 

// CHECK: decl 485498096 
// CHECK: encoding: [0xff,0x0d,0xf0,0x1c,0xf0,0x1c]         
decl 485498096 

// CHECK: decl 64(%edx,%eax) 
// CHECK: encoding: [0xff,0x4c,0x02,0x40]         
decl 64(%edx,%eax) 

// CHECK: decl %eax 
// CHECK: encoding: [0x48]         
decl %eax 

// CHECK: decl (%edx) 
// CHECK: encoding: [0xff,0x0a]         
decl (%edx) 

// CHECK: decw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xff,0x8c,0x82,0x10,0xe3,0x0f,0xe3]         
decw -485498096(%edx,%eax,4) 

// CHECK: decw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xff,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]         
decw 485498096(%edx,%eax,4) 

// CHECK: decw 485498096(%edx) 
// CHECK: encoding: [0x66,0xff,0x8a,0xf0,0x1c,0xf0,0x1c]         
decw 485498096(%edx) 

// CHECK: decw 485498096 
// CHECK: encoding: [0x66,0xff,0x0d,0xf0,0x1c,0xf0,0x1c]         
decw 485498096 

// CHECK: decw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xff,0x4c,0x02,0x40]         
decw 64(%edx,%eax) 

// CHECK: decw (%edx) 
// CHECK: encoding: [0x66,0xff,0x0a]         
decw (%edx) 

// CHECK: divb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xb4,0x82,0x10,0xe3,0x0f,0xe3]         
divb -485498096(%edx,%eax,4) 

// CHECK: divb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]         
divb 485498096(%edx,%eax,4) 

// CHECK: divb 485498096(%edx) 
// CHECK: encoding: [0xf6,0xb2,0xf0,0x1c,0xf0,0x1c]         
divb 485498096(%edx) 

// CHECK: divb 485498096 
// CHECK: encoding: [0xf6,0x35,0xf0,0x1c,0xf0,0x1c]         
divb 485498096 

// CHECK: divb 64(%edx,%eax) 
// CHECK: encoding: [0xf6,0x74,0x02,0x40]         
divb 64(%edx,%eax) 

// CHECK: divb (%edx) 
// CHECK: encoding: [0xf6,0x32]         
divb (%edx) 

// CHECK: divl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xb4,0x82,0x10,0xe3,0x0f,0xe3]         
divl -485498096(%edx,%eax,4) 

// CHECK: divl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]         
divl 485498096(%edx,%eax,4) 

// CHECK: divl 485498096(%edx) 
// CHECK: encoding: [0xf7,0xb2,0xf0,0x1c,0xf0,0x1c]         
divl 485498096(%edx) 

// CHECK: divl 485498096 
// CHECK: encoding: [0xf7,0x35,0xf0,0x1c,0xf0,0x1c]         
divl 485498096 

// CHECK: divl 64(%edx,%eax) 
// CHECK: encoding: [0xf7,0x74,0x02,0x40]         
divl 64(%edx,%eax) 

// CHECK: divl %eax 
// CHECK: encoding: [0xf7,0xf0]         
divl %eax 

// CHECK: divl (%edx) 
// CHECK: encoding: [0xf7,0x32]         
divl (%edx) 

// CHECK: divw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xb4,0x82,0x10,0xe3,0x0f,0xe3]         
divw -485498096(%edx,%eax,4) 

// CHECK: divw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]         
divw 485498096(%edx,%eax,4) 

// CHECK: divw 485498096(%edx) 
// CHECK: encoding: [0x66,0xf7,0xb2,0xf0,0x1c,0xf0,0x1c]         
divw 485498096(%edx) 

// CHECK: divw 485498096 
// CHECK: encoding: [0x66,0xf7,0x35,0xf0,0x1c,0xf0,0x1c]         
divw 485498096 

// CHECK: divw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xf7,0x74,0x02,0x40]         
divw 64(%edx,%eax) 

// CHECK: divw (%edx) 
// CHECK: encoding: [0x66,0xf7,0x32]         
divw (%edx) 

// CHECK: hlt 
// CHECK: encoding: [0xf4]          
hlt 

// CHECK: idivb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
idivb -485498096(%edx,%eax,4) 

// CHECK: idivb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
idivb 485498096(%edx,%eax,4) 

// CHECK: idivb 485498096(%edx) 
// CHECK: encoding: [0xf6,0xba,0xf0,0x1c,0xf0,0x1c]         
idivb 485498096(%edx) 

// CHECK: idivb 485498096 
// CHECK: encoding: [0xf6,0x3d,0xf0,0x1c,0xf0,0x1c]         
idivb 485498096 

// CHECK: idivb 64(%edx,%eax) 
// CHECK: encoding: [0xf6,0x7c,0x02,0x40]         
idivb 64(%edx,%eax) 

// CHECK: idivb (%edx) 
// CHECK: encoding: [0xf6,0x3a]         
idivb (%edx) 

// CHECK: idivl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
idivl -485498096(%edx,%eax,4) 

// CHECK: idivl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
idivl 485498096(%edx,%eax,4) 

// CHECK: idivl 485498096(%edx) 
// CHECK: encoding: [0xf7,0xba,0xf0,0x1c,0xf0,0x1c]         
idivl 485498096(%edx) 

// CHECK: idivl 485498096 
// CHECK: encoding: [0xf7,0x3d,0xf0,0x1c,0xf0,0x1c]         
idivl 485498096 

// CHECK: idivl 64(%edx,%eax) 
// CHECK: encoding: [0xf7,0x7c,0x02,0x40]         
idivl 64(%edx,%eax) 

// CHECK: idivl %eax 
// CHECK: encoding: [0xf7,0xf8]         
idivl %eax 

// CHECK: idivl (%edx) 
// CHECK: encoding: [0xf7,0x3a]         
idivl (%edx) 

// CHECK: idivw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
idivw -485498096(%edx,%eax,4) 

// CHECK: idivw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
idivw 485498096(%edx,%eax,4) 

// CHECK: idivw 485498096(%edx) 
// CHECK: encoding: [0x66,0xf7,0xba,0xf0,0x1c,0xf0,0x1c]         
idivw 485498096(%edx) 

// CHECK: idivw 485498096 
// CHECK: encoding: [0x66,0xf7,0x3d,0xf0,0x1c,0xf0,0x1c]         
idivw 485498096 

// CHECK: idivw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xf7,0x7c,0x02,0x40]         
idivw 64(%edx,%eax) 

// CHECK: idivw (%edx) 
// CHECK: encoding: [0x66,0xf7,0x3a]         
idivw (%edx) 

// CHECK: imulb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
imulb -485498096(%edx,%eax,4) 

// CHECK: imulb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
imulb 485498096(%edx,%eax,4) 

// CHECK: imulb 485498096(%edx) 
// CHECK: encoding: [0xf6,0xaa,0xf0,0x1c,0xf0,0x1c]         
imulb 485498096(%edx) 

// CHECK: imulb 485498096 
// CHECK: encoding: [0xf6,0x2d,0xf0,0x1c,0xf0,0x1c]         
imulb 485498096 

// CHECK: imulb 64(%edx,%eax) 
// CHECK: encoding: [0xf6,0x6c,0x02,0x40]         
imulb 64(%edx,%eax) 

// CHECK: imulb (%edx) 
// CHECK: encoding: [0xf6,0x2a]         
imulb (%edx) 

// CHECK: imull -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
imull -485498096(%edx,%eax,4) 

// CHECK: imull 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
imull 485498096(%edx,%eax,4) 

// CHECK: imull 485498096(%edx) 
// CHECK: encoding: [0xf7,0xaa,0xf0,0x1c,0xf0,0x1c]         
imull 485498096(%edx) 

// CHECK: imull 485498096 
// CHECK: encoding: [0xf7,0x2d,0xf0,0x1c,0xf0,0x1c]         
imull 485498096 

// CHECK: imull 64(%edx,%eax) 
// CHECK: encoding: [0xf7,0x6c,0x02,0x40]         
imull 64(%edx,%eax) 

// CHECK: imull %eax, %eax 
// CHECK: encoding: [0x0f,0xaf,0xc0]        
imull %eax, %eax 

// CHECK: imull %eax 
// CHECK: encoding: [0xf7,0xe8]         
imull %eax 

// CHECK: imull (%edx) 
// CHECK: encoding: [0xf7,0x2a]         
imull (%edx) 

// CHECK: imulw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
imulw -485498096(%edx,%eax,4) 

// CHECK: imulw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
imulw 485498096(%edx,%eax,4) 

// CHECK: imulw 485498096(%edx) 
// CHECK: encoding: [0x66,0xf7,0xaa,0xf0,0x1c,0xf0,0x1c]         
imulw 485498096(%edx) 

// CHECK: imulw 485498096 
// CHECK: encoding: [0x66,0xf7,0x2d,0xf0,0x1c,0xf0,0x1c]         
imulw 485498096 

// CHECK: imulw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xf7,0x6c,0x02,0x40]         
imulw 64(%edx,%eax) 

// CHECK: imulw (%edx) 
// CHECK: encoding: [0x66,0xf7,0x2a]         
imulw (%edx) 

// CHECK: inb $0, %al 
// CHECK: encoding: [0xe4,0x00]        
inb $0, %al 

// CHECK: inb %dx, %al 
// CHECK: encoding: [0xec]        
inb %dx, %al 

// CHECK: incb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xfe,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
incb -485498096(%edx,%eax,4) 

// CHECK: incb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xfe,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
incb 485498096(%edx,%eax,4) 

// CHECK: incb 485498096(%edx) 
// CHECK: encoding: [0xfe,0x82,0xf0,0x1c,0xf0,0x1c]         
incb 485498096(%edx) 

// CHECK: incb 485498096 
// CHECK: encoding: [0xfe,0x05,0xf0,0x1c,0xf0,0x1c]         
incb 485498096 

// CHECK: incb 64(%edx,%eax) 
// CHECK: encoding: [0xfe,0x44,0x02,0x40]         
incb 64(%edx,%eax) 

// CHECK: incb (%edx) 
// CHECK: encoding: [0xfe,0x02]         
incb (%edx) 

// CHECK: incl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xff,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
incl -485498096(%edx,%eax,4) 

// CHECK: incl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xff,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
incl 485498096(%edx,%eax,4) 

// CHECK: incl 485498096(%edx) 
// CHECK: encoding: [0xff,0x82,0xf0,0x1c,0xf0,0x1c]         
incl 485498096(%edx) 

// CHECK: incl 485498096 
// CHECK: encoding: [0xff,0x05,0xf0,0x1c,0xf0,0x1c]         
incl 485498096 

// CHECK: incl 64(%edx,%eax) 
// CHECK: encoding: [0xff,0x44,0x02,0x40]         
incl 64(%edx,%eax) 

// CHECK: incl %eax 
// CHECK: encoding: [0x40]         
incl %eax 

// CHECK: incl (%edx) 
// CHECK: encoding: [0xff,0x02]         
incl (%edx) 

// CHECK: incw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xff,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
incw -485498096(%edx,%eax,4) 

// CHECK: incw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xff,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
incw 485498096(%edx,%eax,4) 

// CHECK: incw 485498096(%edx) 
// CHECK: encoding: [0x66,0xff,0x82,0xf0,0x1c,0xf0,0x1c]         
incw 485498096(%edx) 

// CHECK: incw 485498096 
// CHECK: encoding: [0x66,0xff,0x05,0xf0,0x1c,0xf0,0x1c]         
incw 485498096 

// CHECK: incw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xff,0x44,0x02,0x40]         
incw 64(%edx,%eax) 

// CHECK: incw (%edx) 
// CHECK: encoding: [0x66,0xff,0x02]         
incw (%edx) 

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

// CHECK: into 
// CHECK: encoding: [0xce]          
into 

// CHECK: iretl 
// CHECK: encoding: [0xcf]          
iretl 

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

// CHECK: ldsl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0xc5,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
ldsl 3809469200(%edx,%eax,4), %eax 

// CHECK: ldsl 485498096, %eax 
// CHECK: encoding: [0xc5,0x05,0xf0,0x1c,0xf0,0x1c]        
ldsl 485498096, %eax 

// CHECK: ldsl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0xc5,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
ldsl 485498096(%edx,%eax,4), %eax 

// CHECK: ldsl 485498096(%edx), %eax 
// CHECK: encoding: [0xc5,0x82,0xf0,0x1c,0xf0,0x1c]        
ldsl 485498096(%edx), %eax 

// CHECK: ldsl 64(%edx,%eax), %eax 
// CHECK: encoding: [0xc5,0x44,0x02,0x40]        
ldsl 64(%edx,%eax), %eax 

// CHECK: ldsl (%edx), %eax 
// CHECK: encoding: [0xc5,0x02]        
ldsl (%edx), %eax 

// CHECK: leal 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x8d,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
leal 3809469200(%edx,%eax,4), %eax 

// CHECK: leal 485498096, %eax 
// CHECK: encoding: [0x8d,0x05,0xf0,0x1c,0xf0,0x1c]        
leal 485498096, %eax 

// CHECK: leal 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x8d,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
leal 485498096(%edx,%eax,4), %eax 

// CHECK: leal 485498096(%edx), %eax 
// CHECK: encoding: [0x8d,0x82,0xf0,0x1c,0xf0,0x1c]        
leal 485498096(%edx), %eax 

// CHECK: leal 64(%edx,%eax), %eax 
// CHECK: encoding: [0x8d,0x44,0x02,0x40]        
leal 64(%edx,%eax), %eax 

// CHECK: leal (%edx), %eax 
// CHECK: encoding: [0x8d,0x02]        
leal (%edx), %eax 

// CHECK: lesl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0xc4,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
lesl 3809469200(%edx,%eax,4), %eax 

// CHECK: lesl 485498096, %eax 
// CHECK: encoding: [0xc4,0x05,0xf0,0x1c,0xf0,0x1c]        
lesl 485498096, %eax 

// CHECK: lesl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0xc4,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
lesl 485498096(%edx,%eax,4), %eax 

// CHECK: lesl 485498096(%edx), %eax 
// CHECK: encoding: [0xc4,0x82,0xf0,0x1c,0xf0,0x1c]        
lesl 485498096(%edx), %eax 

// CHECK: lesl 64(%edx,%eax), %eax 
// CHECK: encoding: [0xc4,0x44,0x02,0x40]        
lesl 64(%edx,%eax), %eax 

// CHECK: lesl (%edx), %eax 
// CHECK: encoding: [0xc4,0x02]        
lesl (%edx), %eax 

// CHECK: lock xchgl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0xf0,0x87,0x84,0x82,0x10,0xe3,0x0f,0xe3]       
lock xchgl %eax, 3809469200(%edx,%eax,4) 

// CHECK: lock xchgl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf0,0x87,0x84,0x82,0xf0,0x1c,0xf0,0x1c]       
lock xchgl %eax, 485498096(%edx,%eax,4) 

// CHECK: lock xchgl %eax, 485498096(%edx) 
// CHECK: encoding: [0xf0,0x87,0x82,0xf0,0x1c,0xf0,0x1c]       
lock xchgl %eax, 485498096(%edx) 

// CHECK: lock xchgl %eax, 485498096 
// CHECK: encoding: [0xf0,0x87,0x05,0xf0,0x1c,0xf0,0x1c]       
lock xchgl %eax, 485498096 

// CHECK: lock xchgl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0xf0,0x87,0x44,0x02,0x40]       
lock xchgl %eax, 64(%edx,%eax) 

// CHECK: lock xchgl %eax, (%edx) 
// CHECK: encoding: [0xf0,0x87,0x02]       
lock xchgl %eax, (%edx) 

// CHECK: lodsb %es:(%esi), %al 
// CHECK: encoding: [0x26,0xac]        
lodsb %es:(%esi), %al 

// CHECK: lodsw %es:(%esi), %ax 
// CHECK: encoding: [0x26,0x66,0xad]        
lodsw %es:(%esi), %ax 

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

// CHECK: movb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc6,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
movb $0, -485498096(%edx,%eax,4) 

// CHECK: movb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc6,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
movb $0, 485498096(%edx,%eax,4) 

// CHECK: movb $0, 485498096(%edx) 
// CHECK: encoding: [0xc6,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
movb $0, 485498096(%edx) 

// CHECK: movb $0, 485498096 
// CHECK: encoding: [0xc6,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
movb $0, 485498096 

// CHECK: movb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc6,0x44,0x02,0x40,0x00]        
movb $0, 64(%edx,%eax) 

// CHECK: movb $0, (%edx) 
// CHECK: encoding: [0xc6,0x02,0x00]        
movb $0, (%edx) 

// CHECK: movb %al, %es:485498096 
// CHECK: encoding: [0x26,0xa2,0xf0,0x1c,0xf0,0x1c]        
movb %al, %es:485498096 

// CHECK: movb %es:485498096, %al 
// CHECK: encoding: [0x26,0xa0,0xf0,0x1c,0xf0,0x1c]        
movb %es:485498096, %al 

// CHECK: movl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc7,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00,0x00,0x00,0x00]        
movl $0, -485498096(%edx,%eax,4) 

// CHECK: movl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc7,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
movl $0, 485498096(%edx,%eax,4) 

// CHECK: movl $0, 485498096(%edx) 
// CHECK: encoding: [0xc7,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
movl $0, 485498096(%edx) 

// CHECK: movl $0, 485498096 
// CHECK: encoding: [0xc7,0x05,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
movl $0, 485498096 

// CHECK: movl $0, 64(%edx,%eax) 
// CHECK: encoding: [0xc7,0x44,0x02,0x40,0x00,0x00,0x00,0x00]        
movl $0, 64(%edx,%eax) 

// CHECK: movl $0, %eax 
// CHECK: encoding: [0xb8,0x00,0x00,0x00,0x00]        
movl $0, %eax 

// CHECK: movl $0, (%edx) 
// CHECK: encoding: [0xc7,0x02,0x00,0x00,0x00,0x00]        
movl $0, (%edx) 

// CHECK: movl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x8b,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
movl 3809469200(%edx,%eax,4), %eax 

// CHECK: movl 485498096, %eax 
// CHECK: encoding: [0xa1,0xf0,0x1c,0xf0,0x1c]        
movl 485498096, %eax 

// CHECK: movl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x8b,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
movl 485498096(%edx,%eax,4), %eax 

// CHECK: movl 485498096(%edx), %eax 
// CHECK: encoding: [0x8b,0x82,0xf0,0x1c,0xf0,0x1c]        
movl 485498096(%edx), %eax 

// CHECK: movl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x8b,0x44,0x02,0x40]        
movl 64(%edx,%eax), %eax 

// CHECK: movl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x89,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
movl %eax, 3809469200(%edx,%eax,4) 

// CHECK: movl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x89,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
movl %eax, 485498096(%edx,%eax,4) 

// CHECK: movl %eax, 485498096(%edx) 
// CHECK: encoding: [0x89,0x82,0xf0,0x1c,0xf0,0x1c]        
movl %eax, 485498096(%edx) 

// CHECK: movl %eax, 485498096 
// CHECK: encoding: [0xa3,0xf0,0x1c,0xf0,0x1c]        
movl %eax, 485498096 

// CHECK: movl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x89,0x44,0x02,0x40]        
movl %eax, 64(%edx,%eax) 

// CHECK: movl %eax, %eax 
// CHECK: encoding: [0x89,0xc0]        
movl %eax, %eax 

// CHECK: movl %eax, (%edx) 
// CHECK: encoding: [0x89,0x02]        
movl %eax, (%edx) 

// CHECK: movl (%edx), %eax 
// CHECK: encoding: [0x8b,0x02]        
movl (%edx), %eax 

// CHECK: movl %es, %eax 
// CHECK: encoding: [0x8c,0xc0]        
movl %es, %eax 

// CHECK: movsb %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0x26,0xa4]        
movsb %es:(%esi), %es:(%edi) 

// CHECK: movsl %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0x26,0xa5]        
movsl %es:(%esi), %es:(%edi) 

// CHECK: movsw %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0x26,0x66,0xa5]        
movsw %es:(%esi), %es:(%edi) 

// CHECK: movw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc7,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00,0x00]        
movw $0, -485498096(%edx,%eax,4) 

// CHECK: movw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xc7,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
movw $0, 485498096(%edx,%eax,4) 

// CHECK: movw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xc7,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
movw $0, 485498096(%edx) 

// CHECK: movw $0, 485498096 
// CHECK: encoding: [0x66,0xc7,0x05,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
movw $0, 485498096 

// CHECK: movw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xc7,0x44,0x02,0x40,0x00,0x00]        
movw $0, 64(%edx,%eax) 

// CHECK: movw $0, (%edx) 
// CHECK: encoding: [0x66,0xc7,0x02,0x00,0x00]        
movw $0, (%edx) 

// CHECK: movw -485498096(%edx,%eax,4), %es 
// CHECK: encoding: [0x8e,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
movw -485498096(%edx,%eax,4), %es 

// CHECK: movw 485498096(%edx,%eax,4), %es 
// CHECK: encoding: [0x8e,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
movw 485498096(%edx,%eax,4), %es 

// CHECK: movw 485498096(%edx), %es 
// CHECK: encoding: [0x8e,0x82,0xf0,0x1c,0xf0,0x1c]        
movw 485498096(%edx), %es 

// CHECK: movw 485498096, %es 
// CHECK: encoding: [0x8e,0x05,0xf0,0x1c,0xf0,0x1c]        
movw 485498096, %es 

// CHECK: movw 64(%edx,%eax), %es 
// CHECK: encoding: [0x8e,0x44,0x02,0x40]        
movw 64(%edx,%eax), %es 

// CHECK: movw (%edx), %es 
// CHECK: encoding: [0x8e,0x02]        
movw (%edx), %es 

// CHECK: movw %es, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x8c,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
movw %es, -485498096(%edx,%eax,4) 

// CHECK: movw %es, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x8c,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
movw %es, 485498096(%edx,%eax,4) 

// CHECK: movw %es, 485498096(%edx) 
// CHECK: encoding: [0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
movw %es, 485498096(%edx) 

// CHECK: movw %es, 485498096 
// CHECK: encoding: [0x8c,0x05,0xf0,0x1c,0xf0,0x1c]        
movw %es, 485498096 

// CHECK: movw %es, 64(%edx,%eax) 
// CHECK: encoding: [0x8c,0x44,0x02,0x40]        
movw %es, 64(%edx,%eax) 

// CHECK: movw %es, (%edx) 
// CHECK: encoding: [0x8c,0x02]        
movw %es, (%edx) 

// CHECK: mulb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
mulb -485498096(%edx,%eax,4) 

// CHECK: mulb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
mulb 485498096(%edx,%eax,4) 

// CHECK: mulb 485498096(%edx) 
// CHECK: encoding: [0xf6,0xa2,0xf0,0x1c,0xf0,0x1c]         
mulb 485498096(%edx) 

// CHECK: mulb 485498096 
// CHECK: encoding: [0xf6,0x25,0xf0,0x1c,0xf0,0x1c]         
mulb 485498096 

// CHECK: mulb 64(%edx,%eax) 
// CHECK: encoding: [0xf6,0x64,0x02,0x40]         
mulb 64(%edx,%eax) 

// CHECK: mulb (%edx) 
// CHECK: encoding: [0xf6,0x22]         
mulb (%edx) 

// CHECK: mull -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
mull -485498096(%edx,%eax,4) 

// CHECK: mull 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
mull 485498096(%edx,%eax,4) 

// CHECK: mull 485498096(%edx) 
// CHECK: encoding: [0xf7,0xa2,0xf0,0x1c,0xf0,0x1c]         
mull 485498096(%edx) 

// CHECK: mull 485498096 
// CHECK: encoding: [0xf7,0x25,0xf0,0x1c,0xf0,0x1c]         
mull 485498096 

// CHECK: mull 64(%edx,%eax) 
// CHECK: encoding: [0xf7,0x64,0x02,0x40]         
mull 64(%edx,%eax) 

// CHECK: mull %eax 
// CHECK: encoding: [0xf7,0xe0]         
mull %eax 

// CHECK: mull (%edx) 
// CHECK: encoding: [0xf7,0x22]         
mull (%edx) 

// CHECK: mulw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
mulw -485498096(%edx,%eax,4) 

// CHECK: mulw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
mulw 485498096(%edx,%eax,4) 

// CHECK: mulw 485498096(%edx) 
// CHECK: encoding: [0x66,0xf7,0xa2,0xf0,0x1c,0xf0,0x1c]         
mulw 485498096(%edx) 

// CHECK: mulw 485498096 
// CHECK: encoding: [0x66,0xf7,0x25,0xf0,0x1c,0xf0,0x1c]         
mulw 485498096 

// CHECK: mulw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xf7,0x64,0x02,0x40]         
mulw 64(%edx,%eax) 

// CHECK: mulw (%edx) 
// CHECK: encoding: [0x66,0xf7,0x22]         
mulw (%edx) 

// CHECK: negb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0x9c,0x82,0x10,0xe3,0x0f,0xe3]         
negb -485498096(%edx,%eax,4) 

// CHECK: negb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
negb 485498096(%edx,%eax,4) 

// CHECK: negb 485498096(%edx) 
// CHECK: encoding: [0xf6,0x9a,0xf0,0x1c,0xf0,0x1c]         
negb 485498096(%edx) 

// CHECK: negb 485498096 
// CHECK: encoding: [0xf6,0x1d,0xf0,0x1c,0xf0,0x1c]         
negb 485498096 

// CHECK: negb 64(%edx,%eax) 
// CHECK: encoding: [0xf6,0x5c,0x02,0x40]         
negb 64(%edx,%eax) 

// CHECK: negb (%edx) 
// CHECK: encoding: [0xf6,0x1a]         
negb (%edx) 

// CHECK: negl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0x9c,0x82,0x10,0xe3,0x0f,0xe3]         
negl -485498096(%edx,%eax,4) 

// CHECK: negl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
negl 485498096(%edx,%eax,4) 

// CHECK: negl 485498096(%edx) 
// CHECK: encoding: [0xf7,0x9a,0xf0,0x1c,0xf0,0x1c]         
negl 485498096(%edx) 

// CHECK: negl 485498096 
// CHECK: encoding: [0xf7,0x1d,0xf0,0x1c,0xf0,0x1c]         
negl 485498096 

// CHECK: negl 64(%edx,%eax) 
// CHECK: encoding: [0xf7,0x5c,0x02,0x40]         
negl 64(%edx,%eax) 

// CHECK: negl %eax 
// CHECK: encoding: [0xf7,0xd8]         
negl %eax 

// CHECK: negl (%edx) 
// CHECK: encoding: [0xf7,0x1a]         
negl (%edx) 

// CHECK: negw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0x9c,0x82,0x10,0xe3,0x0f,0xe3]         
negw -485498096(%edx,%eax,4) 

// CHECK: negw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
negw 485498096(%edx,%eax,4) 

// CHECK: negw 485498096(%edx) 
// CHECK: encoding: [0x66,0xf7,0x9a,0xf0,0x1c,0xf0,0x1c]         
negw 485498096(%edx) 

// CHECK: negw 485498096 
// CHECK: encoding: [0x66,0xf7,0x1d,0xf0,0x1c,0xf0,0x1c]         
negw 485498096 

// CHECK: negw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xf7,0x5c,0x02,0x40]         
negw 64(%edx,%eax) 

// CHECK: negw (%edx) 
// CHECK: encoding: [0x66,0xf7,0x1a]         
negw (%edx) 

// CHECK: nop 
// CHECK: encoding: [0x90]          
nop 

// CHECK: notb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0x94,0x82,0x10,0xe3,0x0f,0xe3]         
notb -485498096(%edx,%eax,4) 

// CHECK: notb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0x94,0x82,0xf0,0x1c,0xf0,0x1c]         
notb 485498096(%edx,%eax,4) 

// CHECK: notb 485498096(%edx) 
// CHECK: encoding: [0xf6,0x92,0xf0,0x1c,0xf0,0x1c]         
notb 485498096(%edx) 

// CHECK: notb 485498096 
// CHECK: encoding: [0xf6,0x15,0xf0,0x1c,0xf0,0x1c]         
notb 485498096 

// CHECK: notb 64(%edx,%eax) 
// CHECK: encoding: [0xf6,0x54,0x02,0x40]         
notb 64(%edx,%eax) 

// CHECK: notb (%edx) 
// CHECK: encoding: [0xf6,0x12]         
notb (%edx) 

// CHECK: notl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0x94,0x82,0x10,0xe3,0x0f,0xe3]         
notl -485498096(%edx,%eax,4) 

// CHECK: notl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0x94,0x82,0xf0,0x1c,0xf0,0x1c]         
notl 485498096(%edx,%eax,4) 

// CHECK: notl 485498096(%edx) 
// CHECK: encoding: [0xf7,0x92,0xf0,0x1c,0xf0,0x1c]         
notl 485498096(%edx) 

// CHECK: notl 485498096 
// CHECK: encoding: [0xf7,0x15,0xf0,0x1c,0xf0,0x1c]         
notl 485498096 

// CHECK: notl 64(%edx,%eax) 
// CHECK: encoding: [0xf7,0x54,0x02,0x40]         
notl 64(%edx,%eax) 

// CHECK: notl %eax 
// CHECK: encoding: [0xf7,0xd0]         
notl %eax 

// CHECK: notl (%edx) 
// CHECK: encoding: [0xf7,0x12]         
notl (%edx) 

// CHECK: notw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0x94,0x82,0x10,0xe3,0x0f,0xe3]         
notw -485498096(%edx,%eax,4) 

// CHECK: notw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0x94,0x82,0xf0,0x1c,0xf0,0x1c]         
notw 485498096(%edx,%eax,4) 

// CHECK: notw 485498096(%edx) 
// CHECK: encoding: [0x66,0xf7,0x92,0xf0,0x1c,0xf0,0x1c]         
notw 485498096(%edx) 

// CHECK: notw 485498096 
// CHECK: encoding: [0x66,0xf7,0x15,0xf0,0x1c,0xf0,0x1c]         
notw 485498096 

// CHECK: notw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xf7,0x54,0x02,0x40]         
notw 64(%edx,%eax) 

// CHECK: notw (%edx) 
// CHECK: encoding: [0x66,0xf7,0x12]         
notw (%edx) 

// CHECK: orb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
orb $0, -485498096(%edx,%eax,4) 

// CHECK: orb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
orb $0, 485498096(%edx,%eax,4) 

// CHECK: orb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]        
orb $0, 485498096(%edx) 

// CHECK: orb $0, 485498096 
// CHECK: encoding: [0x80,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]        
orb $0, 485498096 

// CHECK: orb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x4c,0x02,0x40,0x00]        
orb $0, 64(%edx,%eax) 

// CHECK: orb $0, %al 
// CHECK: encoding: [0x0c,0x00]        
orb $0, %al 

// CHECK: orb $0, (%edx) 
// CHECK: encoding: [0x80,0x0a,0x00]        
orb $0, (%edx) 

// CHECK: orl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
orl $0, -485498096(%edx,%eax,4) 

// CHECK: orl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
orl $0, 485498096(%edx,%eax,4) 

// CHECK: orl $0, 485498096(%edx) 
// CHECK: encoding: [0x83,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]        
orl $0, 485498096(%edx) 

// CHECK: orl $0, 485498096 
// CHECK: encoding: [0x83,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]        
orl $0, 485498096 

// CHECK: orl $0, 64(%edx,%eax) 
// CHECK: encoding: [0x83,0x4c,0x02,0x40,0x00]        
orl $0, 64(%edx,%eax) 

// CHECK: orl $0, %eax 
// CHECK: encoding: [0x83,0xc8,0x00]        
orl $0, %eax 

// CHECK: orl $0, (%edx) 
// CHECK: encoding: [0x83,0x0a,0x00]        
orl $0, (%edx) 

// CHECK: orl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x0b,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
orl 3809469200(%edx,%eax,4), %eax 

// CHECK: orl 485498096, %eax 
// CHECK: encoding: [0x0b,0x05,0xf0,0x1c,0xf0,0x1c]        
orl 485498096, %eax 

// CHECK: orl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x0b,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
orl 485498096(%edx,%eax,4), %eax 

// CHECK: orl 485498096(%edx), %eax 
// CHECK: encoding: [0x0b,0x82,0xf0,0x1c,0xf0,0x1c]        
orl 485498096(%edx), %eax 

// CHECK: orl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x0b,0x44,0x02,0x40]        
orl 64(%edx,%eax), %eax 

// CHECK: orl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x09,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
orl %eax, 3809469200(%edx,%eax,4) 

// CHECK: orl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x09,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
orl %eax, 485498096(%edx,%eax,4) 

// CHECK: orl %eax, 485498096(%edx) 
// CHECK: encoding: [0x09,0x82,0xf0,0x1c,0xf0,0x1c]        
orl %eax, 485498096(%edx) 

// CHECK: orl %eax, 485498096 
// CHECK: encoding: [0x09,0x05,0xf0,0x1c,0xf0,0x1c]        
orl %eax, 485498096 

// CHECK: orl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x09,0x44,0x02,0x40]        
orl %eax, 64(%edx,%eax) 

// CHECK: orl %eax, %eax 
// CHECK: encoding: [0x09,0xc0]        
orl %eax, %eax 

// CHECK: orl %eax, (%edx) 
// CHECK: encoding: [0x09,0x02]        
orl %eax, (%edx) 

// CHECK: orl (%edx), %eax 
// CHECK: encoding: [0x0b,0x02]        
orl (%edx), %eax 

// CHECK: orw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
orw $0, -485498096(%edx,%eax,4) 

// CHECK: orw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
orw $0, 485498096(%edx,%eax,4) 

// CHECK: orw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0x83,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]        
orw $0, 485498096(%edx) 

// CHECK: orw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]        
orw $0, 485498096 

// CHECK: orw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0x83,0x4c,0x02,0x40,0x00]        
orw $0, 64(%edx,%eax) 

// CHECK: orw $0, (%edx) 
// CHECK: encoding: [0x66,0x83,0x0a,0x00]        
orw $0, (%edx) 

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

// CHECK: popfl 
// CHECK: encoding: [0x9d]          
popfl 

// CHECK: popfw 
// CHECK: encoding: [0x66,0x9d]          
popfw 

// CHECK: popl %ds 
// CHECK: encoding: [0x1f]         
popl %ds 

// CHECK: popl %eax 
// CHECK: encoding: [0x58]         
popl %eax 

// CHECK: popl %es 
// CHECK: encoding: [0x07]         
popl %es 

// CHECK: popl %fs 
// CHECK: encoding: [0x0f,0xa1]         
popl %fs 

// CHECK: popl %gs 
// CHECK: encoding: [0x0f,0xa9]         
popl %gs 

// CHECK: popl %ss 
// CHECK: encoding: [0x17]         
popl %ss 

// CHECK: popw %ds 
// CHECK: encoding: [0x66,0x1f]         
popw %ds 

// CHECK: popw %es 
// CHECK: encoding: [0x66,0x07]         
popw %es 

// CHECK: popw %fs 
// CHECK: encoding: [0x66,0x0f,0xa1]         
popw %fs 

// CHECK: popw %gs 
// CHECK: encoding: [0x66,0x0f,0xa9]         
popw %gs 

// CHECK: popw %ss 
// CHECK: encoding: [0x66,0x17]         
popw %ss 

// CHECK: pushfl 
// CHECK: encoding: [0x9c]          
pushfl 

// CHECK: pushfw 
// CHECK: encoding: [0x66,0x9c]          
pushfw 

// CHECK: pushl %cs 
// CHECK: encoding: [0x0e]         
pushl %cs 

// CHECK: pushl %ds 
// CHECK: encoding: [0x1e]         
pushl %ds 

// CHECK: pushl %eax 
// CHECK: encoding: [0x50]         
pushl %eax 

// CHECK: pushl %es 
// CHECK: encoding: [0x06]         
pushl %es 

// CHECK: pushl %fs 
// CHECK: encoding: [0x0f,0xa0]         
pushl %fs 

// CHECK: pushl %gs 
// CHECK: encoding: [0x0f,0xa8]         
pushl %gs 

// CHECK: pushl %ss 
// CHECK: encoding: [0x16]         
pushl %ss 

// CHECK: pushw %cs 
// CHECK: encoding: [0x66,0x0e]         
pushw %cs 

// CHECK: pushw %ds 
// CHECK: encoding: [0x66,0x1e]         
pushw %ds 

// CHECK: pushw %es 
// CHECK: encoding: [0x66,0x06]         
pushw %es 

// CHECK: pushw %fs 
// CHECK: encoding: [0x66,0x0f,0xa0]         
pushw %fs 

// CHECK: pushw %gs 
// CHECK: encoding: [0x66,0x0f,0xa8]         
pushw %gs 

// CHECK: pushw %ss 
// CHECK: encoding: [0x66,0x16]         
pushw %ss 

// CHECK: rclb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x94,0x82,0x10,0xe3,0x0f,0xe3]         
rclb -485498096(%edx,%eax,4) 

// CHECK: rclb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x94,0x82,0xf0,0x1c,0xf0,0x1c]         
rclb 485498096(%edx,%eax,4) 

// CHECK: rclb 485498096(%edx) 
// CHECK: encoding: [0xd0,0x92,0xf0,0x1c,0xf0,0x1c]         
rclb 485498096(%edx) 

// CHECK: rclb 485498096 
// CHECK: encoding: [0xd0,0x15,0xf0,0x1c,0xf0,0x1c]         
rclb 485498096 

// CHECK: rclb 64(%edx,%eax) 
// CHECK: encoding: [0xd0,0x54,0x02,0x40]         
rclb 64(%edx,%eax) 

// CHECK: rclb %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
rclb %cl, -485498096(%edx,%eax,4) 

// CHECK: rclb %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
rclb %cl, 485498096(%edx,%eax,4) 

// CHECK: rclb %cl, 485498096(%edx) 
// CHECK: encoding: [0xd2,0x92,0xf0,0x1c,0xf0,0x1c]        
rclb %cl, 485498096(%edx) 

// CHECK: rclb %cl, 485498096 
// CHECK: encoding: [0xd2,0x15,0xf0,0x1c,0xf0,0x1c]        
rclb %cl, 485498096 

// CHECK: rclb %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd2,0x54,0x02,0x40]        
rclb %cl, 64(%edx,%eax) 

// CHECK: rclb %cl, (%edx) 
// CHECK: encoding: [0xd2,0x12]        
rclb %cl, (%edx) 

// CHECK: rclb (%edx) 
// CHECK: encoding: [0xd0,0x12]         
rclb (%edx) 

// CHECK: rcll -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x94,0x82,0x10,0xe3,0x0f,0xe3]         
rcll -485498096(%edx,%eax,4) 

// CHECK: rcll 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x94,0x82,0xf0,0x1c,0xf0,0x1c]         
rcll 485498096(%edx,%eax,4) 

// CHECK: rcll 485498096(%edx) 
// CHECK: encoding: [0xd1,0x92,0xf0,0x1c,0xf0,0x1c]         
rcll 485498096(%edx) 

// CHECK: rcll 485498096 
// CHECK: encoding: [0xd1,0x15,0xf0,0x1c,0xf0,0x1c]         
rcll 485498096 

// CHECK: rcll 64(%edx,%eax) 
// CHECK: encoding: [0xd1,0x54,0x02,0x40]         
rcll 64(%edx,%eax) 

// CHECK: rcll %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
rcll %cl, -485498096(%edx,%eax,4) 

// CHECK: rcll %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
rcll %cl, 485498096(%edx,%eax,4) 

// CHECK: rcll %cl, 485498096(%edx) 
// CHECK: encoding: [0xd3,0x92,0xf0,0x1c,0xf0,0x1c]        
rcll %cl, 485498096(%edx) 

// CHECK: rcll %cl, 485498096 
// CHECK: encoding: [0xd3,0x15,0xf0,0x1c,0xf0,0x1c]        
rcll %cl, 485498096 

// CHECK: rcll %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd3,0x54,0x02,0x40]        
rcll %cl, 64(%edx,%eax) 

// CHECK: rcll %cl, (%edx) 
// CHECK: encoding: [0xd3,0x12]        
rcll %cl, (%edx) 

// CHECK: rcll %eax 
// CHECK: encoding: [0xd1,0xd0]         
rcll %eax 

// CHECK: rcll (%edx) 
// CHECK: encoding: [0xd1,0x12]         
rcll (%edx) 

// CHECK: rclw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x94,0x82,0x10,0xe3,0x0f,0xe3]         
rclw -485498096(%edx,%eax,4) 

// CHECK: rclw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x94,0x82,0xf0,0x1c,0xf0,0x1c]         
rclw 485498096(%edx,%eax,4) 

// CHECK: rclw 485498096(%edx) 
// CHECK: encoding: [0x66,0xd1,0x92,0xf0,0x1c,0xf0,0x1c]         
rclw 485498096(%edx) 

// CHECK: rclw 485498096 
// CHECK: encoding: [0x66,0xd1,0x15,0xf0,0x1c,0xf0,0x1c]         
rclw 485498096 

// CHECK: rclw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd1,0x54,0x02,0x40]         
rclw 64(%edx,%eax) 

// CHECK: rclw %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
rclw %cl, -485498096(%edx,%eax,4) 

// CHECK: rclw %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
rclw %cl, 485498096(%edx,%eax,4) 

// CHECK: rclw %cl, 485498096(%edx) 
// CHECK: encoding: [0x66,0xd3,0x92,0xf0,0x1c,0xf0,0x1c]        
rclw %cl, 485498096(%edx) 

// CHECK: rclw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x15,0xf0,0x1c,0xf0,0x1c]        
rclw %cl, 485498096 

// CHECK: rclw %cl, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd3,0x54,0x02,0x40]        
rclw %cl, 64(%edx,%eax) 

// CHECK: rclw %cl, (%edx) 
// CHECK: encoding: [0x66,0xd3,0x12]        
rclw %cl, (%edx) 

// CHECK: rclw (%edx) 
// CHECK: encoding: [0x66,0xd1,0x12]         
rclw (%edx) 

// CHECK: rcrb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x9c,0x82,0x10,0xe3,0x0f,0xe3]         
rcrb -485498096(%edx,%eax,4) 

// CHECK: rcrb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
rcrb 485498096(%edx,%eax,4) 

// CHECK: rcrb 485498096(%edx) 
// CHECK: encoding: [0xd0,0x9a,0xf0,0x1c,0xf0,0x1c]         
rcrb 485498096(%edx) 

// CHECK: rcrb 485498096 
// CHECK: encoding: [0xd0,0x1d,0xf0,0x1c,0xf0,0x1c]         
rcrb 485498096 

// CHECK: rcrb 64(%edx,%eax) 
// CHECK: encoding: [0xd0,0x5c,0x02,0x40]         
rcrb 64(%edx,%eax) 

// CHECK: rcrb %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
rcrb %cl, -485498096(%edx,%eax,4) 

// CHECK: rcrb %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
rcrb %cl, 485498096(%edx,%eax,4) 

// CHECK: rcrb %cl, 485498096(%edx) 
// CHECK: encoding: [0xd2,0x9a,0xf0,0x1c,0xf0,0x1c]        
rcrb %cl, 485498096(%edx) 

// CHECK: rcrb %cl, 485498096 
// CHECK: encoding: [0xd2,0x1d,0xf0,0x1c,0xf0,0x1c]        
rcrb %cl, 485498096 

// CHECK: rcrb %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd2,0x5c,0x02,0x40]        
rcrb %cl, 64(%edx,%eax) 

// CHECK: rcrb %cl, (%edx) 
// CHECK: encoding: [0xd2,0x1a]        
rcrb %cl, (%edx) 

// CHECK: rcrb (%edx) 
// CHECK: encoding: [0xd0,0x1a]         
rcrb (%edx) 

// CHECK: rcrl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x9c,0x82,0x10,0xe3,0x0f,0xe3]         
rcrl -485498096(%edx,%eax,4) 

// CHECK: rcrl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
rcrl 485498096(%edx,%eax,4) 

// CHECK: rcrl 485498096(%edx) 
// CHECK: encoding: [0xd1,0x9a,0xf0,0x1c,0xf0,0x1c]         
rcrl 485498096(%edx) 

// CHECK: rcrl 485498096 
// CHECK: encoding: [0xd1,0x1d,0xf0,0x1c,0xf0,0x1c]         
rcrl 485498096 

// CHECK: rcrl 64(%edx,%eax) 
// CHECK: encoding: [0xd1,0x5c,0x02,0x40]         
rcrl 64(%edx,%eax) 

// CHECK: rcrl %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
rcrl %cl, -485498096(%edx,%eax,4) 

// CHECK: rcrl %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
rcrl %cl, 485498096(%edx,%eax,4) 

// CHECK: rcrl %cl, 485498096(%edx) 
// CHECK: encoding: [0xd3,0x9a,0xf0,0x1c,0xf0,0x1c]        
rcrl %cl, 485498096(%edx) 

// CHECK: rcrl %cl, 485498096 
// CHECK: encoding: [0xd3,0x1d,0xf0,0x1c,0xf0,0x1c]        
rcrl %cl, 485498096 

// CHECK: rcrl %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd3,0x5c,0x02,0x40]        
rcrl %cl, 64(%edx,%eax) 

// CHECK: rcrl %cl, (%edx) 
// CHECK: encoding: [0xd3,0x1a]        
rcrl %cl, (%edx) 

// CHECK: rcrl %eax 
// CHECK: encoding: [0xd1,0xd8]         
rcrl %eax 

// CHECK: rcrl (%edx) 
// CHECK: encoding: [0xd1,0x1a]         
rcrl (%edx) 

// CHECK: rcrw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x9c,0x82,0x10,0xe3,0x0f,0xe3]         
rcrw -485498096(%edx,%eax,4) 

// CHECK: rcrw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]         
rcrw 485498096(%edx,%eax,4) 

// CHECK: rcrw 485498096(%edx) 
// CHECK: encoding: [0x66,0xd1,0x9a,0xf0,0x1c,0xf0,0x1c]         
rcrw 485498096(%edx) 

// CHECK: rcrw 485498096 
// CHECK: encoding: [0x66,0xd1,0x1d,0xf0,0x1c,0xf0,0x1c]         
rcrw 485498096 

// CHECK: rcrw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd1,0x5c,0x02,0x40]         
rcrw 64(%edx,%eax) 

// CHECK: rcrw %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
rcrw %cl, -485498096(%edx,%eax,4) 

// CHECK: rcrw %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
rcrw %cl, 485498096(%edx,%eax,4) 

// CHECK: rcrw %cl, 485498096(%edx) 
// CHECK: encoding: [0x66,0xd3,0x9a,0xf0,0x1c,0xf0,0x1c]        
rcrw %cl, 485498096(%edx) 

// CHECK: rcrw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x1d,0xf0,0x1c,0xf0,0x1c]        
rcrw %cl, 485498096 

// CHECK: rcrw %cl, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd3,0x5c,0x02,0x40]        
rcrw %cl, 64(%edx,%eax) 

// CHECK: rcrw %cl, (%edx) 
// CHECK: encoding: [0x66,0xd3,0x1a]        
rcrw %cl, (%edx) 

// CHECK: rcrw (%edx) 
// CHECK: encoding: [0x66,0xd1,0x1a]         
rcrw (%edx) 

// CHECK: rep cmpsb %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf3,0x26,0xa6]       
rep cmpsb %es:(%edi), %es:(%esi) 

// CHECK: rep cmpsl %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf3,0x26,0xa7]       
rep cmpsl %es:(%edi), %es:(%esi) 

// CHECK: rep cmpsw %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf3,0x26,0x66,0xa7]       
rep cmpsw %es:(%edi), %es:(%esi) 

// CHECK: rep lodsb %es:(%esi), %al 
// CHECK: encoding: [0xf3,0x26,0xac]       
rep lodsb %es:(%esi), %al 

// CHECK: rep lodsw %es:(%esi), %ax 
// CHECK: encoding: [0xf3,0x26,0x66,0xad]       
rep lodsw %es:(%esi), %ax 

// CHECK: rep movsb %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf3,0x26,0xa4]       
rep movsb %es:(%esi), %es:(%edi) 

// CHECK: rep movsl %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf3,0x26,0xa5]       
rep movsl %es:(%esi), %es:(%edi) 

// CHECK: rep movsw %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf3,0x26,0x66,0xa5]       
rep movsw %es:(%esi), %es:(%edi) 

// CHECK: repne cmpsb %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf2,0x26,0xa6]       
repne cmpsb %es:(%edi), %es:(%esi) 

// CHECK: repne cmpsl %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf2,0x26,0xa7]       
repne cmpsl %es:(%edi), %es:(%esi) 

// CHECK: repne cmpsw %es:(%edi), %es:(%esi) 
// CHECK: encoding: [0xf2,0x26,0x66,0xa7]       
repne cmpsw %es:(%edi), %es:(%esi) 

// CHECK: repne lodsb %es:(%esi), %al 
// CHECK: encoding: [0xf2,0x26,0xac]       
repne lodsb %es:(%esi), %al 

// CHECK: repne lodsw %es:(%esi), %ax 
// CHECK: encoding: [0xf2,0x26,0x66,0xad]       
repne lodsw %es:(%esi), %ax 

// CHECK: repne movsb %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf2,0x26,0xa4]       
repne movsb %es:(%esi), %es:(%edi) 

// CHECK: repne movsl %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf2,0x26,0xa5]       
repne movsl %es:(%esi), %es:(%edi) 

// CHECK: repne movsw %es:(%esi), %es:(%edi) 
// CHECK: encoding: [0xf2,0x26,0x66,0xa5]       
repne movsw %es:(%esi), %es:(%edi) 

// CHECK: repne scasb %es:(%edi), %al 
// CHECK: encoding: [0xf2,0xae]       
repne scasb %es:(%edi), %al 

// CHECK: repne scasw %es:(%edi), %ax 
// CHECK: encoding: [0xf2,0x66,0xaf]       
repne scasw %es:(%edi), %ax 

// CHECK: repne stosb %al, %es:(%edi) 
// CHECK: encoding: [0xf2,0xaa]       
repne stosb %al, %es:(%edi) 

// CHECK: repne stosw %ax, %es:(%edi) 
// CHECK: encoding: [0xf2,0x66,0xab]       
repne stosw %ax, %es:(%edi) 

// CHECK: rep scasb %es:(%edi), %al 
// CHECK: encoding: [0xf3,0xae]       
rep scasb %es:(%edi), %al 

// CHECK: rep scasw %es:(%edi), %ax 
// CHECK: encoding: [0xf3,0x66,0xaf]       
rep scasw %es:(%edi), %ax 

// CHECK: rep stosb %al, %es:(%edi) 
// CHECK: encoding: [0xf3,0xaa]       
rep stosb %al, %es:(%edi) 

// CHECK: rep stosw %ax, %es:(%edi) 
// CHECK: encoding: [0xf3,0x66,0xab]       
rep stosw %ax, %es:(%edi) 

// CHECK: retl $0 
// CHECK: encoding: [0xc2,0x00,0x00]         
retl $0 

// CHECK: retl 
// CHECK: encoding: [0xc3]          
retl 

// CHECK: retw $0 
// CHECK: encoding: [0x66,0xc2,0x00,0x00]         
retw $0 

// CHECK: retw 
// CHECK: encoding: [0x66,0xc3]          
retw 

// CHECK: rolb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
rolb -485498096(%edx,%eax,4) 

// CHECK: rolb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
rolb 485498096(%edx,%eax,4) 

// CHECK: rolb 485498096(%edx) 
// CHECK: encoding: [0xd0,0x82,0xf0,0x1c,0xf0,0x1c]         
rolb 485498096(%edx) 

// CHECK: rolb 485498096 
// CHECK: encoding: [0xd0,0x05,0xf0,0x1c,0xf0,0x1c]         
rolb 485498096 

// CHECK: rolb 64(%edx,%eax) 
// CHECK: encoding: [0xd0,0x44,0x02,0x40]         
rolb 64(%edx,%eax) 

// CHECK: rolb %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
rolb %cl, -485498096(%edx,%eax,4) 

// CHECK: rolb %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
rolb %cl, 485498096(%edx,%eax,4) 

// CHECK: rolb %cl, 485498096(%edx) 
// CHECK: encoding: [0xd2,0x82,0xf0,0x1c,0xf0,0x1c]        
rolb %cl, 485498096(%edx) 

// CHECK: rolb %cl, 485498096 
// CHECK: encoding: [0xd2,0x05,0xf0,0x1c,0xf0,0x1c]        
rolb %cl, 485498096 

// CHECK: rolb %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd2,0x44,0x02,0x40]        
rolb %cl, 64(%edx,%eax) 

// CHECK: rolb %cl, (%edx) 
// CHECK: encoding: [0xd2,0x02]        
rolb %cl, (%edx) 

// CHECK: rolb (%edx) 
// CHECK: encoding: [0xd0,0x02]         
rolb (%edx) 

// CHECK: roll -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
roll -485498096(%edx,%eax,4) 

// CHECK: roll 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
roll 485498096(%edx,%eax,4) 

// CHECK: roll 485498096(%edx) 
// CHECK: encoding: [0xd1,0x82,0xf0,0x1c,0xf0,0x1c]         
roll 485498096(%edx) 

// CHECK: roll 485498096 
// CHECK: encoding: [0xd1,0x05,0xf0,0x1c,0xf0,0x1c]         
roll 485498096 

// CHECK: roll 64(%edx,%eax) 
// CHECK: encoding: [0xd1,0x44,0x02,0x40]         
roll 64(%edx,%eax) 

// CHECK: roll %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
roll %cl, -485498096(%edx,%eax,4) 

// CHECK: roll %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
roll %cl, 485498096(%edx,%eax,4) 

// CHECK: roll %cl, 485498096(%edx) 
// CHECK: encoding: [0xd3,0x82,0xf0,0x1c,0xf0,0x1c]        
roll %cl, 485498096(%edx) 

// CHECK: roll %cl, 485498096 
// CHECK: encoding: [0xd3,0x05,0xf0,0x1c,0xf0,0x1c]        
roll %cl, 485498096 

// CHECK: roll %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd3,0x44,0x02,0x40]        
roll %cl, 64(%edx,%eax) 

// CHECK: roll %cl, (%edx) 
// CHECK: encoding: [0xd3,0x02]        
roll %cl, (%edx) 

// CHECK: roll %eax 
// CHECK: encoding: [0xd1,0xc0]         
roll %eax 

// CHECK: roll (%edx) 
// CHECK: encoding: [0xd1,0x02]         
roll (%edx) 

// CHECK: rolw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x84,0x82,0x10,0xe3,0x0f,0xe3]         
rolw -485498096(%edx,%eax,4) 

// CHECK: rolw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x84,0x82,0xf0,0x1c,0xf0,0x1c]         
rolw 485498096(%edx,%eax,4) 

// CHECK: rolw 485498096(%edx) 
// CHECK: encoding: [0x66,0xd1,0x82,0xf0,0x1c,0xf0,0x1c]         
rolw 485498096(%edx) 

// CHECK: rolw 485498096 
// CHECK: encoding: [0x66,0xd1,0x05,0xf0,0x1c,0xf0,0x1c]         
rolw 485498096 

// CHECK: rolw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd1,0x44,0x02,0x40]         
rolw 64(%edx,%eax) 

// CHECK: rolw %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
rolw %cl, -485498096(%edx,%eax,4) 

// CHECK: rolw %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
rolw %cl, 485498096(%edx,%eax,4) 

// CHECK: rolw %cl, 485498096(%edx) 
// CHECK: encoding: [0x66,0xd3,0x82,0xf0,0x1c,0xf0,0x1c]        
rolw %cl, 485498096(%edx) 

// CHECK: rolw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x05,0xf0,0x1c,0xf0,0x1c]        
rolw %cl, 485498096 

// CHECK: rolw %cl, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd3,0x44,0x02,0x40]        
rolw %cl, 64(%edx,%eax) 

// CHECK: rolw %cl, (%edx) 
// CHECK: encoding: [0x66,0xd3,0x02]        
rolw %cl, (%edx) 

// CHECK: rolw (%edx) 
// CHECK: encoding: [0x66,0xd1,0x02]         
rolw (%edx) 

// CHECK: rorb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x8c,0x82,0x10,0xe3,0x0f,0xe3]         
rorb -485498096(%edx,%eax,4) 

// CHECK: rorb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]         
rorb 485498096(%edx,%eax,4) 

// CHECK: rorb 485498096(%edx) 
// CHECK: encoding: [0xd0,0x8a,0xf0,0x1c,0xf0,0x1c]         
rorb 485498096(%edx) 

// CHECK: rorb 485498096 
// CHECK: encoding: [0xd0,0x0d,0xf0,0x1c,0xf0,0x1c]         
rorb 485498096 

// CHECK: rorb 64(%edx,%eax) 
// CHECK: encoding: [0xd0,0x4c,0x02,0x40]         
rorb 64(%edx,%eax) 

// CHECK: rorb %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
rorb %cl, -485498096(%edx,%eax,4) 

// CHECK: rorb %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
rorb %cl, 485498096(%edx,%eax,4) 

// CHECK: rorb %cl, 485498096(%edx) 
// CHECK: encoding: [0xd2,0x8a,0xf0,0x1c,0xf0,0x1c]        
rorb %cl, 485498096(%edx) 

// CHECK: rorb %cl, 485498096 
// CHECK: encoding: [0xd2,0x0d,0xf0,0x1c,0xf0,0x1c]        
rorb %cl, 485498096 

// CHECK: rorb %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd2,0x4c,0x02,0x40]        
rorb %cl, 64(%edx,%eax) 

// CHECK: rorb %cl, (%edx) 
// CHECK: encoding: [0xd2,0x0a]        
rorb %cl, (%edx) 

// CHECK: rorb (%edx) 
// CHECK: encoding: [0xd0,0x0a]         
rorb (%edx) 

// CHECK: rorl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x8c,0x82,0x10,0xe3,0x0f,0xe3]         
rorl -485498096(%edx,%eax,4) 

// CHECK: rorl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]         
rorl 485498096(%edx,%eax,4) 

// CHECK: rorl 485498096(%edx) 
// CHECK: encoding: [0xd1,0x8a,0xf0,0x1c,0xf0,0x1c]         
rorl 485498096(%edx) 

// CHECK: rorl 485498096 
// CHECK: encoding: [0xd1,0x0d,0xf0,0x1c,0xf0,0x1c]         
rorl 485498096 

// CHECK: rorl 64(%edx,%eax) 
// CHECK: encoding: [0xd1,0x4c,0x02,0x40]         
rorl 64(%edx,%eax) 

// CHECK: rorl %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
rorl %cl, -485498096(%edx,%eax,4) 

// CHECK: rorl %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
rorl %cl, 485498096(%edx,%eax,4) 

// CHECK: rorl %cl, 485498096(%edx) 
// CHECK: encoding: [0xd3,0x8a,0xf0,0x1c,0xf0,0x1c]        
rorl %cl, 485498096(%edx) 

// CHECK: rorl %cl, 485498096 
// CHECK: encoding: [0xd3,0x0d,0xf0,0x1c,0xf0,0x1c]        
rorl %cl, 485498096 

// CHECK: rorl %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd3,0x4c,0x02,0x40]        
rorl %cl, 64(%edx,%eax) 

// CHECK: rorl %cl, (%edx) 
// CHECK: encoding: [0xd3,0x0a]        
rorl %cl, (%edx) 

// CHECK: rorl %eax 
// CHECK: encoding: [0xd1,0xc8]         
rorl %eax 

// CHECK: rorl (%edx) 
// CHECK: encoding: [0xd1,0x0a]         
rorl (%edx) 

// CHECK: rorw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x8c,0x82,0x10,0xe3,0x0f,0xe3]         
rorw -485498096(%edx,%eax,4) 

// CHECK: rorw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]         
rorw 485498096(%edx,%eax,4) 

// CHECK: rorw 485498096(%edx) 
// CHECK: encoding: [0x66,0xd1,0x8a,0xf0,0x1c,0xf0,0x1c]         
rorw 485498096(%edx) 

// CHECK: rorw 485498096 
// CHECK: encoding: [0x66,0xd1,0x0d,0xf0,0x1c,0xf0,0x1c]         
rorw 485498096 

// CHECK: rorw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd1,0x4c,0x02,0x40]         
rorw 64(%edx,%eax) 

// CHECK: rorw %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
rorw %cl, -485498096(%edx,%eax,4) 

// CHECK: rorw %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
rorw %cl, 485498096(%edx,%eax,4) 

// CHECK: rorw %cl, 485498096(%edx) 
// CHECK: encoding: [0x66,0xd3,0x8a,0xf0,0x1c,0xf0,0x1c]        
rorw %cl, 485498096(%edx) 

// CHECK: rorw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x0d,0xf0,0x1c,0xf0,0x1c]        
rorw %cl, 485498096 

// CHECK: rorw %cl, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd3,0x4c,0x02,0x40]        
rorw %cl, 64(%edx,%eax) 

// CHECK: rorw %cl, (%edx) 
// CHECK: encoding: [0x66,0xd3,0x0a]        
rorw %cl, (%edx) 

// CHECK: rorw (%edx) 
// CHECK: encoding: [0x66,0xd1,0x0a]         
rorw (%edx) 

// CHECK: salc 
// CHECK: encoding: [0xd6]          
salc 

// CHECK: sarb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
sarb -485498096(%edx,%eax,4) 

// CHECK: sarb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
sarb 485498096(%edx,%eax,4) 

// CHECK: sarb 485498096(%edx) 
// CHECK: encoding: [0xd0,0xba,0xf0,0x1c,0xf0,0x1c]         
sarb 485498096(%edx) 

// CHECK: sarb 485498096 
// CHECK: encoding: [0xd0,0x3d,0xf0,0x1c,0xf0,0x1c]         
sarb 485498096 

// CHECK: sarb 64(%edx,%eax) 
// CHECK: encoding: [0xd0,0x7c,0x02,0x40]         
sarb 64(%edx,%eax) 

// CHECK: sarb %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
sarb %cl, -485498096(%edx,%eax,4) 

// CHECK: sarb %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
sarb %cl, 485498096(%edx,%eax,4) 

// CHECK: sarb %cl, 485498096(%edx) 
// CHECK: encoding: [0xd2,0xba,0xf0,0x1c,0xf0,0x1c]        
sarb %cl, 485498096(%edx) 

// CHECK: sarb %cl, 485498096 
// CHECK: encoding: [0xd2,0x3d,0xf0,0x1c,0xf0,0x1c]        
sarb %cl, 485498096 

// CHECK: sarb %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd2,0x7c,0x02,0x40]        
sarb %cl, 64(%edx,%eax) 

// CHECK: sarb %cl, (%edx) 
// CHECK: encoding: [0xd2,0x3a]        
sarb %cl, (%edx) 

// CHECK: sarb (%edx) 
// CHECK: encoding: [0xd0,0x3a]         
sarb (%edx) 

// CHECK: sarl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
sarl -485498096(%edx,%eax,4) 

// CHECK: sarl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
sarl 485498096(%edx,%eax,4) 

// CHECK: sarl 485498096(%edx) 
// CHECK: encoding: [0xd1,0xba,0xf0,0x1c,0xf0,0x1c]         
sarl 485498096(%edx) 

// CHECK: sarl 485498096 
// CHECK: encoding: [0xd1,0x3d,0xf0,0x1c,0xf0,0x1c]         
sarl 485498096 

// CHECK: sarl 64(%edx,%eax) 
// CHECK: encoding: [0xd1,0x7c,0x02,0x40]         
sarl 64(%edx,%eax) 

// CHECK: sarl %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
sarl %cl, -485498096(%edx,%eax,4) 

// CHECK: sarl %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
sarl %cl, 485498096(%edx,%eax,4) 

// CHECK: sarl %cl, 485498096(%edx) 
// CHECK: encoding: [0xd3,0xba,0xf0,0x1c,0xf0,0x1c]        
sarl %cl, 485498096(%edx) 

// CHECK: sarl %cl, 485498096 
// CHECK: encoding: [0xd3,0x3d,0xf0,0x1c,0xf0,0x1c]        
sarl %cl, 485498096 

// CHECK: sarl %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd3,0x7c,0x02,0x40]        
sarl %cl, 64(%edx,%eax) 

// CHECK: sarl %cl, (%edx) 
// CHECK: encoding: [0xd3,0x3a]        
sarl %cl, (%edx) 

// CHECK: sarl %eax 
// CHECK: encoding: [0xd1,0xf8]         
sarl %eax 

// CHECK: sarl (%edx) 
// CHECK: encoding: [0xd1,0x3a]         
sarl (%edx) 

// CHECK: sarw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0xbc,0x82,0x10,0xe3,0x0f,0xe3]         
sarw -485498096(%edx,%eax,4) 

// CHECK: sarw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]         
sarw 485498096(%edx,%eax,4) 

// CHECK: sarw 485498096(%edx) 
// CHECK: encoding: [0x66,0xd1,0xba,0xf0,0x1c,0xf0,0x1c]         
sarw 485498096(%edx) 

// CHECK: sarw 485498096 
// CHECK: encoding: [0x66,0xd1,0x3d,0xf0,0x1c,0xf0,0x1c]         
sarw 485498096 

// CHECK: sarw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd1,0x7c,0x02,0x40]         
sarw 64(%edx,%eax) 

// CHECK: sarw %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
sarw %cl, -485498096(%edx,%eax,4) 

// CHECK: sarw %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
sarw %cl, 485498096(%edx,%eax,4) 

// CHECK: sarw %cl, 485498096(%edx) 
// CHECK: encoding: [0x66,0xd3,0xba,0xf0,0x1c,0xf0,0x1c]        
sarw %cl, 485498096(%edx) 

// CHECK: sarw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x3d,0xf0,0x1c,0xf0,0x1c]        
sarw %cl, 485498096 

// CHECK: sarw %cl, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd3,0x7c,0x02,0x40]        
sarw %cl, 64(%edx,%eax) 

// CHECK: sarw %cl, (%edx) 
// CHECK: encoding: [0x66,0xd3,0x3a]        
sarw %cl, (%edx) 

// CHECK: sarw (%edx) 
// CHECK: encoding: [0x66,0xd1,0x3a]         
sarw (%edx) 

// CHECK: sbbb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x9c,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
sbbb $0, -485498096(%edx,%eax,4) 

// CHECK: sbbb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0x9c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
sbbb $0, 485498096(%edx,%eax,4) 

// CHECK: sbbb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0x9a,0xf0,0x1c,0xf0,0x1c,0x00]        
sbbb $0, 485498096(%edx) 

// CHECK: sbbb $0, 485498096 
// CHECK: encoding: [0x80,0x1d,0xf0,0x1c,0xf0,0x1c,0x00]        
sbbb $0, 485498096 

// CHECK: sbbb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x5c,0x02,0x40,0x00]        
sbbb $0, 64(%edx,%eax) 

// CHECK: sbbb $0, %al 
// CHECK: encoding: [0x1c,0x00]        
sbbb $0, %al 

// CHECK: sbbb $0, (%edx) 
// CHECK: encoding: [0x80,0x1a,0x00]        
sbbb $0, (%edx) 

// CHECK: sbbl $0, %eax 
// CHECK: encoding: [0x83,0xd8,0x00]        
sbbl $0, %eax 

// CHECK: sbbl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x1b,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
sbbl 3809469200(%edx,%eax,4), %eax 

// CHECK: sbbl 485498096, %eax 
// CHECK: encoding: [0x1b,0x05,0xf0,0x1c,0xf0,0x1c]        
sbbl 485498096, %eax 

// CHECK: sbbl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x1b,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
sbbl 485498096(%edx,%eax,4), %eax 

// CHECK: sbbl 485498096(%edx), %eax 
// CHECK: encoding: [0x1b,0x82,0xf0,0x1c,0xf0,0x1c]        
sbbl 485498096(%edx), %eax 

// CHECK: sbbl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x1b,0x44,0x02,0x40]        
sbbl 64(%edx,%eax), %eax 

// CHECK: sbbl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x19,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
sbbl %eax, 3809469200(%edx,%eax,4) 

// CHECK: sbbl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x19,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
sbbl %eax, 485498096(%edx,%eax,4) 

// CHECK: sbbl %eax, 485498096(%edx) 
// CHECK: encoding: [0x19,0x82,0xf0,0x1c,0xf0,0x1c]        
sbbl %eax, 485498096(%edx) 

// CHECK: sbbl %eax, 485498096 
// CHECK: encoding: [0x19,0x05,0xf0,0x1c,0xf0,0x1c]        
sbbl %eax, 485498096 

// CHECK: sbbl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x19,0x44,0x02,0x40]        
sbbl %eax, 64(%edx,%eax) 

// CHECK: sbbl %eax, %eax 
// CHECK: encoding: [0x19,0xc0]        
sbbl %eax, %eax 

// CHECK: sbbl %eax, (%edx) 
// CHECK: encoding: [0x19,0x02]        
sbbl %eax, (%edx) 

// CHECK: sbbl (%edx), %eax 
// CHECK: encoding: [0x1b,0x02]        
sbbl (%edx), %eax 

// CHECK: scasb %es:(%edi), %al 
// CHECK: encoding: [0xae]        
scasb %es:(%edi), %al 

// CHECK: scasw %es:(%edi), %ax 
// CHECK: encoding: [0x66,0xaf]        
scasw %es:(%edi), %ax 

// CHECK: shlb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
shlb -485498096(%edx,%eax,4) 

// CHECK: shlb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
shlb 485498096(%edx,%eax,4) 

// CHECK: shlb 485498096(%edx) 
// CHECK: encoding: [0xd0,0xa2,0xf0,0x1c,0xf0,0x1c]         
shlb 485498096(%edx) 

// CHECK: shlb 485498096 
// CHECK: encoding: [0xd0,0x25,0xf0,0x1c,0xf0,0x1c]         
shlb 485498096 

// CHECK: shlb 64(%edx,%eax) 
// CHECK: encoding: [0xd0,0x64,0x02,0x40]         
shlb 64(%edx,%eax) 

// CHECK: shlb %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
shlb %cl, -485498096(%edx,%eax,4) 

// CHECK: shlb %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
shlb %cl, 485498096(%edx,%eax,4) 

// CHECK: shlb %cl, 485498096(%edx) 
// CHECK: encoding: [0xd2,0xa2,0xf0,0x1c,0xf0,0x1c]        
shlb %cl, 485498096(%edx) 

// CHECK: shlb %cl, 485498096 
// CHECK: encoding: [0xd2,0x25,0xf0,0x1c,0xf0,0x1c]        
shlb %cl, 485498096 

// CHECK: shlb %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd2,0x64,0x02,0x40]        
shlb %cl, 64(%edx,%eax) 

// CHECK: shlb %cl, (%edx) 
// CHECK: encoding: [0xd2,0x22]        
shlb %cl, (%edx) 

// CHECK: shlb (%edx) 
// CHECK: encoding: [0xd0,0x22]         
shlb (%edx) 

// CHECK: shll -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
shll -485498096(%edx,%eax,4) 

// CHECK: shll 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
shll 485498096(%edx,%eax,4) 

// CHECK: shll 485498096(%edx) 
// CHECK: encoding: [0xd1,0xa2,0xf0,0x1c,0xf0,0x1c]         
shll 485498096(%edx) 

// CHECK: shll 485498096 
// CHECK: encoding: [0xd1,0x25,0xf0,0x1c,0xf0,0x1c]         
shll 485498096 

// CHECK: shll 64(%edx,%eax) 
// CHECK: encoding: [0xd1,0x64,0x02,0x40]         
shll 64(%edx,%eax) 

// CHECK: shll %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
shll %cl, -485498096(%edx,%eax,4) 

// CHECK: shll %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
shll %cl, 485498096(%edx,%eax,4) 

// CHECK: shll %cl, 485498096(%edx) 
// CHECK: encoding: [0xd3,0xa2,0xf0,0x1c,0xf0,0x1c]        
shll %cl, 485498096(%edx) 

// CHECK: shll %cl, 485498096 
// CHECK: encoding: [0xd3,0x25,0xf0,0x1c,0xf0,0x1c]        
shll %cl, 485498096 

// CHECK: shll %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd3,0x64,0x02,0x40]        
shll %cl, 64(%edx,%eax) 

// CHECK: shll %cl, (%edx) 
// CHECK: encoding: [0xd3,0x22]        
shll %cl, (%edx) 

// CHECK: shll %eax 
// CHECK: encoding: [0xd1,0xe0]         
shll %eax 

// CHECK: shll (%edx) 
// CHECK: encoding: [0xd1,0x22]         
shll (%edx) 

// CHECK: shlw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]         
shlw -485498096(%edx,%eax,4) 

// CHECK: shlw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]         
shlw 485498096(%edx,%eax,4) 

// CHECK: shlw 485498096(%edx) 
// CHECK: encoding: [0x66,0xd1,0xa2,0xf0,0x1c,0xf0,0x1c]         
shlw 485498096(%edx) 

// CHECK: shlw 485498096 
// CHECK: encoding: [0x66,0xd1,0x25,0xf0,0x1c,0xf0,0x1c]         
shlw 485498096 

// CHECK: shlw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd1,0x64,0x02,0x40]         
shlw 64(%edx,%eax) 

// CHECK: shlw %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
shlw %cl, -485498096(%edx,%eax,4) 

// CHECK: shlw %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
shlw %cl, 485498096(%edx,%eax,4) 

// CHECK: shlw %cl, 485498096(%edx) 
// CHECK: encoding: [0x66,0xd3,0xa2,0xf0,0x1c,0xf0,0x1c]        
shlw %cl, 485498096(%edx) 

// CHECK: shlw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x25,0xf0,0x1c,0xf0,0x1c]        
shlw %cl, 485498096 

// CHECK: shlw %cl, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd3,0x64,0x02,0x40]        
shlw %cl, 64(%edx,%eax) 

// CHECK: shlw %cl, (%edx) 
// CHECK: encoding: [0x66,0xd3,0x22]        
shlw %cl, (%edx) 

// CHECK: shlw (%edx) 
// CHECK: encoding: [0x66,0xd1,0x22]         
shlw (%edx) 

// CHECK: shrb -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
shrb -485498096(%edx,%eax,4) 

// CHECK: shrb 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd0,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
shrb 485498096(%edx,%eax,4) 

// CHECK: shrb 485498096(%edx) 
// CHECK: encoding: [0xd0,0xaa,0xf0,0x1c,0xf0,0x1c]         
shrb 485498096(%edx) 

// CHECK: shrb 485498096 
// CHECK: encoding: [0xd0,0x2d,0xf0,0x1c,0xf0,0x1c]         
shrb 485498096 

// CHECK: shrb 64(%edx,%eax) 
// CHECK: encoding: [0xd0,0x6c,0x02,0x40]         
shrb 64(%edx,%eax) 

// CHECK: shrb %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
shrb %cl, -485498096(%edx,%eax,4) 

// CHECK: shrb %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd2,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
shrb %cl, 485498096(%edx,%eax,4) 

// CHECK: shrb %cl, 485498096(%edx) 
// CHECK: encoding: [0xd2,0xaa,0xf0,0x1c,0xf0,0x1c]        
shrb %cl, 485498096(%edx) 

// CHECK: shrb %cl, 485498096 
// CHECK: encoding: [0xd2,0x2d,0xf0,0x1c,0xf0,0x1c]        
shrb %cl, 485498096 

// CHECK: shrb %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd2,0x6c,0x02,0x40]        
shrb %cl, 64(%edx,%eax) 

// CHECK: shrb %cl, (%edx) 
// CHECK: encoding: [0xd2,0x2a]        
shrb %cl, (%edx) 

// CHECK: shrb (%edx) 
// CHECK: encoding: [0xd0,0x2a]         
shrb (%edx) 

// CHECK: shrl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
shrl -485498096(%edx,%eax,4) 

// CHECK: shrl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd1,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
shrl 485498096(%edx,%eax,4) 

// CHECK: shrl 485498096(%edx) 
// CHECK: encoding: [0xd1,0xaa,0xf0,0x1c,0xf0,0x1c]         
shrl 485498096(%edx) 

// CHECK: shrl 485498096 
// CHECK: encoding: [0xd1,0x2d,0xf0,0x1c,0xf0,0x1c]         
shrl 485498096 

// CHECK: shrl 64(%edx,%eax) 
// CHECK: encoding: [0xd1,0x6c,0x02,0x40]         
shrl 64(%edx,%eax) 

// CHECK: shrl %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
shrl %cl, -485498096(%edx,%eax,4) 

// CHECK: shrl %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd3,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
shrl %cl, 485498096(%edx,%eax,4) 

// CHECK: shrl %cl, 485498096(%edx) 
// CHECK: encoding: [0xd3,0xaa,0xf0,0x1c,0xf0,0x1c]        
shrl %cl, 485498096(%edx) 

// CHECK: shrl %cl, 485498096 
// CHECK: encoding: [0xd3,0x2d,0xf0,0x1c,0xf0,0x1c]        
shrl %cl, 485498096 

// CHECK: shrl %cl, 64(%edx,%eax) 
// CHECK: encoding: [0xd3,0x6c,0x02,0x40]        
shrl %cl, 64(%edx,%eax) 

// CHECK: shrl %cl, (%edx) 
// CHECK: encoding: [0xd3,0x2a]        
shrl %cl, (%edx) 

// CHECK: shrl %eax 
// CHECK: encoding: [0xd1,0xe8]         
shrl %eax 

// CHECK: shrl (%edx) 
// CHECK: encoding: [0xd1,0x2a]         
shrl (%edx) 

// CHECK: shrw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0xac,0x82,0x10,0xe3,0x0f,0xe3]         
shrw -485498096(%edx,%eax,4) 

// CHECK: shrw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd1,0xac,0x82,0xf0,0x1c,0xf0,0x1c]         
shrw 485498096(%edx,%eax,4) 

// CHECK: shrw 485498096(%edx) 
// CHECK: encoding: [0x66,0xd1,0xaa,0xf0,0x1c,0xf0,0x1c]         
shrw 485498096(%edx) 

// CHECK: shrw 485498096 
// CHECK: encoding: [0x66,0xd1,0x2d,0xf0,0x1c,0xf0,0x1c]         
shrw 485498096 

// CHECK: shrw 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd1,0x6c,0x02,0x40]         
shrw 64(%edx,%eax) 

// CHECK: shrw %cl, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
shrw %cl, -485498096(%edx,%eax,4) 

// CHECK: shrw %cl, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xd3,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
shrw %cl, 485498096(%edx,%eax,4) 

// CHECK: shrw %cl, 485498096(%edx) 
// CHECK: encoding: [0x66,0xd3,0xaa,0xf0,0x1c,0xf0,0x1c]        
shrw %cl, 485498096(%edx) 

// CHECK: shrw %cl, 485498096 
// CHECK: encoding: [0x66,0xd3,0x2d,0xf0,0x1c,0xf0,0x1c]        
shrw %cl, 485498096 

// CHECK: shrw %cl, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xd3,0x6c,0x02,0x40]        
shrw %cl, 64(%edx,%eax) 

// CHECK: shrw %cl, (%edx) 
// CHECK: encoding: [0x66,0xd3,0x2a]        
shrw %cl, (%edx) 

// CHECK: shrw (%edx) 
// CHECK: encoding: [0x66,0xd1,0x2a]         
shrw (%edx) 

// CHECK: stc 
// CHECK: encoding: [0xf9]          
stc 

// CHECK: std 
// CHECK: encoding: [0xfd]          
std 

// CHECK: sti 
// CHECK: encoding: [0xfb]          
sti 

// CHECK: stosb %al, %es:(%edi) 
// CHECK: encoding: [0xaa]        
stosb %al, %es:(%edi) 

// CHECK: stosw %ax, %es:(%edi) 
// CHECK: encoding: [0x66,0xab]        
stosw %ax, %es:(%edi) 

// CHECK: subb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xac,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
subb $0, -485498096(%edx,%eax,4) 

// CHECK: subb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xac,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
subb $0, 485498096(%edx,%eax,4) 

// CHECK: subb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0xaa,0xf0,0x1c,0xf0,0x1c,0x00]        
subb $0, 485498096(%edx) 

// CHECK: subb $0, 485498096 
// CHECK: encoding: [0x80,0x2d,0xf0,0x1c,0xf0,0x1c,0x00]        
subb $0, 485498096 

// CHECK: subb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x6c,0x02,0x40,0x00]        
subb $0, 64(%edx,%eax) 

// CHECK: subb $0, %al 
// CHECK: encoding: [0x2c,0x00]        
subb $0, %al 

// CHECK: subb $0, (%edx) 
// CHECK: encoding: [0x80,0x2a,0x00]        
subb $0, (%edx) 

// CHECK: subl $0, %eax 
// CHECK: encoding: [0x83,0xe8,0x00]        
subl $0, %eax 

// CHECK: subl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x2b,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
subl 3809469200(%edx,%eax,4), %eax 

// CHECK: subl 485498096, %eax 
// CHECK: encoding: [0x2b,0x05,0xf0,0x1c,0xf0,0x1c]        
subl 485498096, %eax 

// CHECK: subl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x2b,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
subl 485498096(%edx,%eax,4), %eax 

// CHECK: subl 485498096(%edx), %eax 
// CHECK: encoding: [0x2b,0x82,0xf0,0x1c,0xf0,0x1c]        
subl 485498096(%edx), %eax 

// CHECK: subl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x2b,0x44,0x02,0x40]        
subl 64(%edx,%eax), %eax 

// CHECK: subl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x29,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
subl %eax, 3809469200(%edx,%eax,4) 

// CHECK: subl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x29,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
subl %eax, 485498096(%edx,%eax,4) 

// CHECK: subl %eax, 485498096(%edx) 
// CHECK: encoding: [0x29,0x82,0xf0,0x1c,0xf0,0x1c]        
subl %eax, 485498096(%edx) 

// CHECK: subl %eax, 485498096 
// CHECK: encoding: [0x29,0x05,0xf0,0x1c,0xf0,0x1c]        
subl %eax, 485498096 

// CHECK: subl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x29,0x44,0x02,0x40]        
subl %eax, 64(%edx,%eax) 

// CHECK: subl %eax, %eax 
// CHECK: encoding: [0x29,0xc0]        
subl %eax, %eax 

// CHECK: subl %eax, (%edx) 
// CHECK: encoding: [0x29,0x02]        
subl %eax, (%edx) 

// CHECK: subl (%edx), %eax 
// CHECK: encoding: [0x2b,0x02]        
subl (%edx), %eax 

// CHECK: testb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
testb $0, -485498096(%edx,%eax,4) 

// CHECK: testb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf6,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
testb $0, 485498096(%edx,%eax,4) 

// CHECK: testb $0, 485498096(%edx) 
// CHECK: encoding: [0xf6,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
testb $0, 485498096(%edx) 

// CHECK: testb $0, 485498096 
// CHECK: encoding: [0xf6,0x05,0xf0,0x1c,0xf0,0x1c,0x00]        
testb $0, 485498096 

// CHECK: testb $0, 64(%edx,%eax) 
// CHECK: encoding: [0xf6,0x44,0x02,0x40,0x00]        
testb $0, 64(%edx,%eax) 

// CHECK: testb $0, %al 
// CHECK: encoding: [0xa8,0x00]        
testb $0, %al 

// CHECK: testb $0, (%edx) 
// CHECK: encoding: [0xf6,0x02,0x00]        
testb $0, (%edx) 

// CHECK: testl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00,0x00,0x00,0x00]        
testl $0, -485498096(%edx,%eax,4) 

// CHECK: testl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xf7,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
testl $0, 485498096(%edx,%eax,4) 

// CHECK: testl $0, 485498096(%edx) 
// CHECK: encoding: [0xf7,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
testl $0, 485498096(%edx) 

// CHECK: testl $0, 485498096 
// CHECK: encoding: [0xf7,0x05,0xf0,0x1c,0xf0,0x1c,0x00,0x00,0x00,0x00]        
testl $0, 485498096 

// CHECK: testl $0, 64(%edx,%eax) 
// CHECK: encoding: [0xf7,0x44,0x02,0x40,0x00,0x00,0x00,0x00]        
testl $0, 64(%edx,%eax) 

// CHECK: testl $0, %eax 
// CHECK: encoding: [0xa9,0x00,0x00,0x00,0x00]        
testl $0, %eax 

// CHECK: testl $0, (%edx) 
// CHECK: encoding: [0xf7,0x02,0x00,0x00,0x00,0x00]        
testl $0, (%edx) 

// CHECK: testl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x85,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
testl %eax, 3809469200(%edx,%eax,4) 

// CHECK: testl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x85,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
testl %eax, 485498096(%edx,%eax,4) 

// CHECK: testl %eax, 485498096(%edx) 
// CHECK: encoding: [0x85,0x82,0xf0,0x1c,0xf0,0x1c]        
testl %eax, 485498096(%edx) 

// CHECK: testl %eax, 485498096 
// CHECK: encoding: [0x85,0x05,0xf0,0x1c,0xf0,0x1c]        
testl %eax, 485498096 

// CHECK: testl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x85,0x44,0x02,0x40]        
testl %eax, 64(%edx,%eax) 

// CHECK: testl %eax, %eax 
// CHECK: encoding: [0x85,0xc0]        
testl %eax, %eax 

// CHECK: testl %eax, (%edx) 
// CHECK: encoding: [0x85,0x02]        
testl %eax, (%edx) 

// CHECK: testw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0x84,0x82,0x10,0xe3,0x0f,0xe3,0x00,0x00]        
testw $0, -485498096(%edx,%eax,4) 

// CHECK: testw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0xf7,0x84,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
testw $0, 485498096(%edx,%eax,4) 

// CHECK: testw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0xf7,0x82,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
testw $0, 485498096(%edx) 

// CHECK: testw $0, 485498096 
// CHECK: encoding: [0x66,0xf7,0x05,0xf0,0x1c,0xf0,0x1c,0x00,0x00]        
testw $0, 485498096 

// CHECK: testw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0xf7,0x44,0x02,0x40,0x00,0x00]        
testw $0, 64(%edx,%eax) 

// CHECK: testw $0, (%edx) 
// CHECK: encoding: [0x66,0xf7,0x02,0x00,0x00]        
testw $0, (%edx) 

// CHECK: xchgl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x87,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
xchgl %eax, 3809469200(%edx,%eax,4) 

// CHECK: xchgl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x87,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
xchgl %eax, 485498096(%edx,%eax,4) 

// CHECK: xchgl %eax, 485498096(%edx) 
// CHECK: encoding: [0x87,0x82,0xf0,0x1c,0xf0,0x1c]        
xchgl %eax, 485498096(%edx) 

// CHECK: xchgl %eax, 485498096 
// CHECK: encoding: [0x87,0x05,0xf0,0x1c,0xf0,0x1c]        
xchgl %eax, 485498096 

// CHECK: xchgl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x87,0x44,0x02,0x40]        
xchgl %eax, 64(%edx,%eax) 

// CHECK: xchgl %eax, %eax 
// CHECK: encoding: [0x90]        
xchgl %eax, %eax 

// CHECK: xchgl %eax, (%edx) 
// CHECK: encoding: [0x87,0x02]        
xchgl %eax, (%edx) 

// CHECK: xlatb 
// CHECK: encoding: [0xd7]          
xlatb 

// CHECK: xorb $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xb4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
xorb $0, -485498096(%edx,%eax,4) 

// CHECK: xorb $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x80,0xb4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
xorb $0, 485498096(%edx,%eax,4) 

// CHECK: xorb $0, 485498096(%edx) 
// CHECK: encoding: [0x80,0xb2,0xf0,0x1c,0xf0,0x1c,0x00]        
xorb $0, 485498096(%edx) 

// CHECK: xorb $0, 485498096 
// CHECK: encoding: [0x80,0x35,0xf0,0x1c,0xf0,0x1c,0x00]        
xorb $0, 485498096 

// CHECK: xorb $0, 64(%edx,%eax) 
// CHECK: encoding: [0x80,0x74,0x02,0x40,0x00]        
xorb $0, 64(%edx,%eax) 

// CHECK: xorb $0, %al 
// CHECK: encoding: [0x34,0x00]        
xorb $0, %al 

// CHECK: xorb $0, (%edx) 
// CHECK: encoding: [0x80,0x32,0x00]        
xorb $0, (%edx) 

// CHECK: xorl $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0xb4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
xorl $0, -485498096(%edx,%eax,4) 

// CHECK: xorl $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x83,0xb4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
xorl $0, 485498096(%edx,%eax,4) 

// CHECK: xorl $0, 485498096(%edx) 
// CHECK: encoding: [0x83,0xb2,0xf0,0x1c,0xf0,0x1c,0x00]        
xorl $0, 485498096(%edx) 

// CHECK: xorl $0, 485498096 
// CHECK: encoding: [0x83,0x35,0xf0,0x1c,0xf0,0x1c,0x00]        
xorl $0, 485498096 

// CHECK: xorl $0, 64(%edx,%eax) 
// CHECK: encoding: [0x83,0x74,0x02,0x40,0x00]        
xorl $0, 64(%edx,%eax) 

// CHECK: xorl $0, %eax 
// CHECK: encoding: [0x83,0xf0,0x00]        
xorl $0, %eax 

// CHECK: xorl $0, (%edx) 
// CHECK: encoding: [0x83,0x32,0x00]        
xorl $0, (%edx) 

// CHECK: xorl 3809469200(%edx,%eax,4), %eax 
// CHECK: encoding: [0x33,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
xorl 3809469200(%edx,%eax,4), %eax 

// CHECK: xorl 485498096, %eax 
// CHECK: encoding: [0x33,0x05,0xf0,0x1c,0xf0,0x1c]        
xorl 485498096, %eax 

// CHECK: xorl 485498096(%edx,%eax,4), %eax 
// CHECK: encoding: [0x33,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
xorl 485498096(%edx,%eax,4), %eax 

// CHECK: xorl 485498096(%edx), %eax 
// CHECK: encoding: [0x33,0x82,0xf0,0x1c,0xf0,0x1c]        
xorl 485498096(%edx), %eax 

// CHECK: xorl 64(%edx,%eax), %eax 
// CHECK: encoding: [0x33,0x44,0x02,0x40]        
xorl 64(%edx,%eax), %eax 

// CHECK: xorl %eax, 3809469200(%edx,%eax,4) 
// CHECK: encoding: [0x31,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
xorl %eax, 3809469200(%edx,%eax,4) 

// CHECK: xorl %eax, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x31,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
xorl %eax, 485498096(%edx,%eax,4) 

// CHECK: xorl %eax, 485498096(%edx) 
// CHECK: encoding: [0x31,0x82,0xf0,0x1c,0xf0,0x1c]        
xorl %eax, 485498096(%edx) 

// CHECK: xorl %eax, 485498096 
// CHECK: encoding: [0x31,0x05,0xf0,0x1c,0xf0,0x1c]        
xorl %eax, 485498096 

// CHECK: xorl %eax, 64(%edx,%eax) 
// CHECK: encoding: [0x31,0x44,0x02,0x40]        
xorl %eax, 64(%edx,%eax) 

// CHECK: xorl %eax, %eax 
// CHECK: encoding: [0x31,0xc0]        
xorl %eax, %eax 

// CHECK: xorl %eax, (%edx) 
// CHECK: encoding: [0x31,0x02]        
xorl %eax, (%edx) 

// CHECK: xorl (%edx), %eax 
// CHECK: encoding: [0x33,0x02]        
xorl (%edx), %eax 

// CHECK: xorw $0, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0xb4,0x82,0x10,0xe3,0x0f,0xe3,0x00]        
xorw $0, -485498096(%edx,%eax,4) 

// CHECK: xorw $0, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0x66,0x83,0xb4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]        
xorw $0, 485498096(%edx,%eax,4) 

// CHECK: xorw $0, 485498096(%edx) 
// CHECK: encoding: [0x66,0x83,0xb2,0xf0,0x1c,0xf0,0x1c,0x00]        
xorw $0, 485498096(%edx) 

// CHECK: xorw $0, 485498096 
// CHECK: encoding: [0x66,0x83,0x35,0xf0,0x1c,0xf0,0x1c,0x00]        
xorw $0, 485498096 

// CHECK: xorw $0, 64(%edx,%eax) 
// CHECK: encoding: [0x66,0x83,0x74,0x02,0x40,0x00]        
xorw $0, 64(%edx,%eax) 

// CHECK: xorw $0, (%edx) 
// CHECK: encoding: [0x66,0x83,0x32,0x00]        
xorw $0, (%edx) 

