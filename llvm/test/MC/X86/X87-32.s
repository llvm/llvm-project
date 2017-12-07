// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s
// CHECK: f2xm1 
// CHECK: encoding: [0xd9,0xf0]         
f2xm1 

// CHECK: fabs 
// CHECK: encoding: [0xd9,0xe1]         
fabs 

// CHECK: faddl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
faddl -485498096(%edx,%eax,4) 

// CHECK: faddl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
faddl 485498096(%edx,%eax,4) 

// CHECK: faddl 485498096(%edx) 
// CHECK: encoding: [0xdc,0x82,0xf0,0x1c,0xf0,0x1c]        
faddl 485498096(%edx) 

// CHECK: faddl 485498096 
// CHECK: encoding: [0xdc,0x05,0xf0,0x1c,0xf0,0x1c]        
faddl 485498096 

// CHECK: faddl 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x44,0x02,0x40]        
faddl 64(%edx,%eax) 

// CHECK: faddl (%edx) 
// CHECK: encoding: [0xdc,0x02]        
faddl (%edx) 

// CHECK: faddp %st(4) 
// CHECK: encoding: [0xde,0xc4]        
faddp %st(4) 

// CHECK: fadds -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
fadds -485498096(%edx,%eax,4) 

// CHECK: fadds 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
fadds 485498096(%edx,%eax,4) 

// CHECK: fadds 485498096(%edx) 
// CHECK: encoding: [0xd8,0x82,0xf0,0x1c,0xf0,0x1c]        
fadds 485498096(%edx) 

// CHECK: fadds 485498096 
// CHECK: encoding: [0xd8,0x05,0xf0,0x1c,0xf0,0x1c]        
fadds 485498096 

// CHECK: fadds 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x44,0x02,0x40]        
fadds 64(%edx,%eax) 

// CHECK: fadds (%edx) 
// CHECK: encoding: [0xd8,0x02]        
fadds (%edx) 

// CHECK: fadd %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xc4]       
fadd %st(0), %st(4) 

// CHECK: fadd %st(4) 
// CHECK: encoding: [0xd8,0xc4]        
fadd %st(4) 

// CHECK: fbld -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
fbld -485498096(%edx,%eax,4) 

// CHECK: fbld 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
fbld 485498096(%edx,%eax,4) 

// CHECK: fbld 485498096(%edx) 
// CHECK: encoding: [0xdf,0xa2,0xf0,0x1c,0xf0,0x1c]        
fbld 485498096(%edx) 

// CHECK: fbld 485498096 
// CHECK: encoding: [0xdf,0x25,0xf0,0x1c,0xf0,0x1c]        
fbld 485498096 

// CHECK: fbld 64(%edx,%eax) 
// CHECK: encoding: [0xdf,0x64,0x02,0x40]        
fbld 64(%edx,%eax) 

// CHECK: fbld (%edx) 
// CHECK: encoding: [0xdf,0x22]        
fbld (%edx) 

// CHECK: fbstp -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xb4,0x82,0x10,0xe3,0x0f,0xe3]        
fbstp -485498096(%edx,%eax,4) 

// CHECK: fbstp 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]        
fbstp 485498096(%edx,%eax,4) 

// CHECK: fbstp 485498096(%edx) 
// CHECK: encoding: [0xdf,0xb2,0xf0,0x1c,0xf0,0x1c]        
fbstp 485498096(%edx) 

// CHECK: fbstp 485498096 
// CHECK: encoding: [0xdf,0x35,0xf0,0x1c,0xf0,0x1c]        
fbstp 485498096 

// CHECK: fbstp 64(%edx,%eax) 
// CHECK: encoding: [0xdf,0x74,0x02,0x40]        
fbstp 64(%edx,%eax) 

// CHECK: fbstp (%edx) 
// CHECK: encoding: [0xdf,0x32]        
fbstp (%edx) 

// CHECK: fchs 
// CHECK: encoding: [0xd9,0xe0]         
fchs 

// CHECK: fcoml -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
fcoml -485498096(%edx,%eax,4) 

// CHECK: fcoml 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
fcoml 485498096(%edx,%eax,4) 

// CHECK: fcoml 485498096(%edx) 
// CHECK: encoding: [0xdc,0x92,0xf0,0x1c,0xf0,0x1c]        
fcoml 485498096(%edx) 

// CHECK: fcoml 485498096 
// CHECK: encoding: [0xdc,0x15,0xf0,0x1c,0xf0,0x1c]        
fcoml 485498096 

// CHECK: fcoml 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x54,0x02,0x40]        
fcoml 64(%edx,%eax) 

// CHECK: fcoml (%edx) 
// CHECK: encoding: [0xdc,0x12]        
fcoml (%edx) 

// CHECK: fcompl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
fcompl -485498096(%edx,%eax,4) 

// CHECK: fcompl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
fcompl 485498096(%edx,%eax,4) 

// CHECK: fcompl 485498096(%edx) 
// CHECK: encoding: [0xdc,0x9a,0xf0,0x1c,0xf0,0x1c]        
fcompl 485498096(%edx) 

// CHECK: fcompl 485498096 
// CHECK: encoding: [0xdc,0x1d,0xf0,0x1c,0xf0,0x1c]        
fcompl 485498096 

// CHECK: fcompl 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x5c,0x02,0x40]        
fcompl 64(%edx,%eax) 

// CHECK: fcompl (%edx) 
// CHECK: encoding: [0xdc,0x1a]        
fcompl (%edx) 

// CHECK: fcompp 
// CHECK: encoding: [0xde,0xd9]         
fcompp 

// CHECK: fcomps -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
fcomps -485498096(%edx,%eax,4) 

// CHECK: fcomps 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
fcomps 485498096(%edx,%eax,4) 

// CHECK: fcomps 485498096(%edx) 
// CHECK: encoding: [0xd8,0x9a,0xf0,0x1c,0xf0,0x1c]        
fcomps 485498096(%edx) 

// CHECK: fcomps 485498096 
// CHECK: encoding: [0xd8,0x1d,0xf0,0x1c,0xf0,0x1c]        
fcomps 485498096 

// CHECK: fcomps 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x5c,0x02,0x40]        
fcomps 64(%edx,%eax) 

// CHECK: fcomps (%edx) 
// CHECK: encoding: [0xd8,0x1a]        
fcomps (%edx) 

// CHECK: fcomp %st(4) 
// CHECK: encoding: [0xd8,0xdc]        
fcomp %st(4) 

// CHECK: fcoms -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
fcoms -485498096(%edx,%eax,4) 

// CHECK: fcoms 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
fcoms 485498096(%edx,%eax,4) 

// CHECK: fcoms 485498096(%edx) 
// CHECK: encoding: [0xd8,0x92,0xf0,0x1c,0xf0,0x1c]        
fcoms 485498096(%edx) 

// CHECK: fcoms 485498096 
// CHECK: encoding: [0xd8,0x15,0xf0,0x1c,0xf0,0x1c]        
fcoms 485498096 

// CHECK: fcoms 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x54,0x02,0x40]        
fcoms 64(%edx,%eax) 

// CHECK: fcoms (%edx) 
// CHECK: encoding: [0xd8,0x12]        
fcoms (%edx) 

// CHECK: fcom %st(4) 
// CHECK: encoding: [0xd8,0xd4]        
fcom %st(4) 

// CHECK: fcos 
// CHECK: encoding: [0xd9,0xff]         
fcos 

// CHECK: fdecstp 
// CHECK: encoding: [0xd9,0xf6]         
fdecstp 

// CHECK: fdivl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xb4,0x82,0x10,0xe3,0x0f,0xe3]        
fdivl -485498096(%edx,%eax,4) 

// CHECK: fdivl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]        
fdivl 485498096(%edx,%eax,4) 

// CHECK: fdivl 485498096(%edx) 
// CHECK: encoding: [0xdc,0xb2,0xf0,0x1c,0xf0,0x1c]        
fdivl 485498096(%edx) 

// CHECK: fdivl 485498096 
// CHECK: encoding: [0xdc,0x35,0xf0,0x1c,0xf0,0x1c]        
fdivl 485498096 

// CHECK: fdivl 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x74,0x02,0x40]        
fdivl 64(%edx,%eax) 

// CHECK: fdivl (%edx) 
// CHECK: encoding: [0xdc,0x32]        
fdivl (%edx) 

// CHECK: fdivp %st(4) 
// CHECK: encoding: [0xde,0xf4]        
fdivp %st(4) 

// CHECK: fdivrl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fdivrl -485498096(%edx,%eax,4) 

// CHECK: fdivrl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fdivrl 485498096(%edx,%eax,4) 

// CHECK: fdivrl 485498096(%edx) 
// CHECK: encoding: [0xdc,0xba,0xf0,0x1c,0xf0,0x1c]        
fdivrl 485498096(%edx) 

// CHECK: fdivrl 485498096 
// CHECK: encoding: [0xdc,0x3d,0xf0,0x1c,0xf0,0x1c]        
fdivrl 485498096 

// CHECK: fdivrl 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x7c,0x02,0x40]        
fdivrl 64(%edx,%eax) 

// CHECK: fdivrl (%edx) 
// CHECK: encoding: [0xdc,0x3a]        
fdivrl (%edx) 

// CHECK: fdivrp %st(4) 
// CHECK: encoding: [0xde,0xfc]        
fdivrp %st(4) 

// CHECK: fdivrs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fdivrs -485498096(%edx,%eax,4) 

// CHECK: fdivrs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fdivrs 485498096(%edx,%eax,4) 

// CHECK: fdivrs 485498096(%edx) 
// CHECK: encoding: [0xd8,0xba,0xf0,0x1c,0xf0,0x1c]        
fdivrs 485498096(%edx) 

// CHECK: fdivrs 485498096 
// CHECK: encoding: [0xd8,0x3d,0xf0,0x1c,0xf0,0x1c]        
fdivrs 485498096 

// CHECK: fdivrs 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x7c,0x02,0x40]        
fdivrs 64(%edx,%eax) 

// CHECK: fdivrs (%edx) 
// CHECK: encoding: [0xd8,0x3a]        
fdivrs (%edx) 

// CHECK: fdivr %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xfc]       
fdivr %st(0), %st(4) 

// CHECK: fdivr %st(4) 
// CHECK: encoding: [0xd8,0xfc]        
fdivr %st(4) 

// CHECK: fdivs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xb4,0x82,0x10,0xe3,0x0f,0xe3]        
fdivs -485498096(%edx,%eax,4) 

// CHECK: fdivs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]        
fdivs 485498096(%edx,%eax,4) 

// CHECK: fdivs 485498096(%edx) 
// CHECK: encoding: [0xd8,0xb2,0xf0,0x1c,0xf0,0x1c]        
fdivs 485498096(%edx) 

// CHECK: fdivs 485498096 
// CHECK: encoding: [0xd8,0x35,0xf0,0x1c,0xf0,0x1c]        
fdivs 485498096 

// CHECK: fdivs 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x74,0x02,0x40]        
fdivs 64(%edx,%eax) 

// CHECK: fdivs (%edx) 
// CHECK: encoding: [0xd8,0x32]        
fdivs (%edx) 

// CHECK: fdiv %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xf4]       
fdiv %st(0), %st(4) 

// CHECK: fdiv %st(4) 
// CHECK: encoding: [0xd8,0xf4]        
fdiv %st(4) 

// CHECK: ffreep %st(4) 
// CHECK: encoding: [0xdf,0xc4]        
ffreep %st(4) 

// CHECK: ffree %st(4) 
// CHECK: encoding: [0xdd,0xc4]        
ffree %st(4) 

// CHECK: fiaddl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
fiaddl -485498096(%edx,%eax,4) 

// CHECK: fiaddl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
fiaddl 485498096(%edx,%eax,4) 

// CHECK: fiaddl 485498096(%edx) 
// CHECK: encoding: [0xda,0x82,0xf0,0x1c,0xf0,0x1c]        
fiaddl 485498096(%edx) 

// CHECK: fiaddl 485498096 
// CHECK: encoding: [0xda,0x05,0xf0,0x1c,0xf0,0x1c]        
fiaddl 485498096 

// CHECK: fiaddl 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x44,0x02,0x40]        
fiaddl 64(%edx,%eax) 

// CHECK: fiaddl (%edx) 
// CHECK: encoding: [0xda,0x02]        
fiaddl (%edx) 

// CHECK: fiadds -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
fiadds -485498096(%edx,%eax,4) 

// CHECK: fiadds 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
fiadds 485498096(%edx,%eax,4) 

// CHECK: fiadds 485498096(%edx) 
// CHECK: encoding: [0xde,0x82,0xf0,0x1c,0xf0,0x1c]        
fiadds 485498096(%edx) 

// CHECK: fiadds 485498096 
// CHECK: encoding: [0xde,0x05,0xf0,0x1c,0xf0,0x1c]        
fiadds 485498096 

// CHECK: fiadds 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x44,0x02,0x40]        
fiadds 64(%edx,%eax) 

// CHECK: fiadds (%edx) 
// CHECK: encoding: [0xde,0x02]        
fiadds (%edx) 

// CHECK: ficoml -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
ficoml -485498096(%edx,%eax,4) 

// CHECK: ficoml 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
ficoml 485498096(%edx,%eax,4) 

// CHECK: ficoml 485498096(%edx) 
// CHECK: encoding: [0xda,0x92,0xf0,0x1c,0xf0,0x1c]        
ficoml 485498096(%edx) 

// CHECK: ficoml 485498096 
// CHECK: encoding: [0xda,0x15,0xf0,0x1c,0xf0,0x1c]        
ficoml 485498096 

// CHECK: ficoml 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x54,0x02,0x40]        
ficoml 64(%edx,%eax) 

// CHECK: ficoml (%edx) 
// CHECK: encoding: [0xda,0x12]        
ficoml (%edx) 

// CHECK: ficompl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
ficompl -485498096(%edx,%eax,4) 

// CHECK: ficompl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
ficompl 485498096(%edx,%eax,4) 

// CHECK: ficompl 485498096(%edx) 
// CHECK: encoding: [0xda,0x9a,0xf0,0x1c,0xf0,0x1c]        
ficompl 485498096(%edx) 

// CHECK: ficompl 485498096 
// CHECK: encoding: [0xda,0x1d,0xf0,0x1c,0xf0,0x1c]        
ficompl 485498096 

// CHECK: ficompl 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x5c,0x02,0x40]        
ficompl 64(%edx,%eax) 

// CHECK: ficompl (%edx) 
// CHECK: encoding: [0xda,0x1a]        
ficompl (%edx) 

// CHECK: ficomps -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
ficomps -485498096(%edx,%eax,4) 

// CHECK: ficomps 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
ficomps 485498096(%edx,%eax,4) 

// CHECK: ficomps 485498096(%edx) 
// CHECK: encoding: [0xde,0x9a,0xf0,0x1c,0xf0,0x1c]        
ficomps 485498096(%edx) 

// CHECK: ficomps 485498096 
// CHECK: encoding: [0xde,0x1d,0xf0,0x1c,0xf0,0x1c]        
ficomps 485498096 

// CHECK: ficomps 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x5c,0x02,0x40]        
ficomps 64(%edx,%eax) 

// CHECK: ficomps (%edx) 
// CHECK: encoding: [0xde,0x1a]        
ficomps (%edx) 

// CHECK: ficoms -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
ficoms -485498096(%edx,%eax,4) 

// CHECK: ficoms 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
ficoms 485498096(%edx,%eax,4) 

// CHECK: ficoms 485498096(%edx) 
// CHECK: encoding: [0xde,0x92,0xf0,0x1c,0xf0,0x1c]        
ficoms 485498096(%edx) 

// CHECK: ficoms 485498096 
// CHECK: encoding: [0xde,0x15,0xf0,0x1c,0xf0,0x1c]        
ficoms 485498096 

// CHECK: ficoms 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x54,0x02,0x40]        
ficoms 64(%edx,%eax) 

// CHECK: ficoms (%edx) 
// CHECK: encoding: [0xde,0x12]        
ficoms (%edx) 

// CHECK: fidivl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xb4,0x82,0x10,0xe3,0x0f,0xe3]        
fidivl -485498096(%edx,%eax,4) 

// CHECK: fidivl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]        
fidivl 485498096(%edx,%eax,4) 

// CHECK: fidivl 485498096(%edx) 
// CHECK: encoding: [0xda,0xb2,0xf0,0x1c,0xf0,0x1c]        
fidivl 485498096(%edx) 

// CHECK: fidivl 485498096 
// CHECK: encoding: [0xda,0x35,0xf0,0x1c,0xf0,0x1c]        
fidivl 485498096 

// CHECK: fidivl 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x74,0x02,0x40]        
fidivl 64(%edx,%eax) 

// CHECK: fidivl (%edx) 
// CHECK: encoding: [0xda,0x32]        
fidivl (%edx) 

// CHECK: fidivrl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fidivrl -485498096(%edx,%eax,4) 

// CHECK: fidivrl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fidivrl 485498096(%edx,%eax,4) 

// CHECK: fidivrl 485498096(%edx) 
// CHECK: encoding: [0xda,0xba,0xf0,0x1c,0xf0,0x1c]        
fidivrl 485498096(%edx) 

// CHECK: fidivrl 485498096 
// CHECK: encoding: [0xda,0x3d,0xf0,0x1c,0xf0,0x1c]        
fidivrl 485498096 

// CHECK: fidivrl 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x7c,0x02,0x40]        
fidivrl 64(%edx,%eax) 

// CHECK: fidivrl (%edx) 
// CHECK: encoding: [0xda,0x3a]        
fidivrl (%edx) 

// CHECK: fidivrs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fidivrs -485498096(%edx,%eax,4) 

// CHECK: fidivrs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fidivrs 485498096(%edx,%eax,4) 

// CHECK: fidivrs 485498096(%edx) 
// CHECK: encoding: [0xde,0xba,0xf0,0x1c,0xf0,0x1c]        
fidivrs 485498096(%edx) 

// CHECK: fidivrs 485498096 
// CHECK: encoding: [0xde,0x3d,0xf0,0x1c,0xf0,0x1c]        
fidivrs 485498096 

// CHECK: fidivrs 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x7c,0x02,0x40]        
fidivrs 64(%edx,%eax) 

// CHECK: fidivrs (%edx) 
// CHECK: encoding: [0xde,0x3a]        
fidivrs (%edx) 

// CHECK: fidivs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xb4,0x82,0x10,0xe3,0x0f,0xe3]        
fidivs -485498096(%edx,%eax,4) 

// CHECK: fidivs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]        
fidivs 485498096(%edx,%eax,4) 

// CHECK: fidivs 485498096(%edx) 
// CHECK: encoding: [0xde,0xb2,0xf0,0x1c,0xf0,0x1c]        
fidivs 485498096(%edx) 

// CHECK: fidivs 485498096 
// CHECK: encoding: [0xde,0x35,0xf0,0x1c,0xf0,0x1c]        
fidivs 485498096 

// CHECK: fidivs 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x74,0x02,0x40]        
fidivs 64(%edx,%eax) 

// CHECK: fidivs (%edx) 
// CHECK: encoding: [0xde,0x32]        
fidivs (%edx) 

// CHECK: fildl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
fildl -485498096(%edx,%eax,4) 

// CHECK: fildl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
fildl 485498096(%edx,%eax,4) 

// CHECK: fildl 485498096(%edx) 
// CHECK: encoding: [0xdb,0x82,0xf0,0x1c,0xf0,0x1c]        
fildl 485498096(%edx) 

// CHECK: fildl 485498096 
// CHECK: encoding: [0xdb,0x05,0xf0,0x1c,0xf0,0x1c]        
fildl 485498096 

// CHECK: fildl 64(%edx,%eax) 
// CHECK: encoding: [0xdb,0x44,0x02,0x40]        
fildl 64(%edx,%eax) 

// CHECK: fildl (%edx) 
// CHECK: encoding: [0xdb,0x02]        
fildl (%edx) 

// CHECK: fildll -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
fildll -485498096(%edx,%eax,4) 

// CHECK: fildll 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
fildll 485498096(%edx,%eax,4) 

// CHECK: fildll 485498096(%edx) 
// CHECK: encoding: [0xdf,0xaa,0xf0,0x1c,0xf0,0x1c]        
fildll 485498096(%edx) 

// CHECK: fildll 485498096 
// CHECK: encoding: [0xdf,0x2d,0xf0,0x1c,0xf0,0x1c]        
fildll 485498096 

// CHECK: fildll 64(%edx,%eax) 
// CHECK: encoding: [0xdf,0x6c,0x02,0x40]        
fildll 64(%edx,%eax) 

// CHECK: fildll (%edx) 
// CHECK: encoding: [0xdf,0x2a]        
fildll (%edx) 

// CHECK: filds -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
filds -485498096(%edx,%eax,4) 

// CHECK: filds 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
filds 485498096(%edx,%eax,4) 

// CHECK: filds 485498096(%edx) 
// CHECK: encoding: [0xdf,0x82,0xf0,0x1c,0xf0,0x1c]        
filds 485498096(%edx) 

// CHECK: filds 485498096 
// CHECK: encoding: [0xdf,0x05,0xf0,0x1c,0xf0,0x1c]        
filds 485498096 

// CHECK: filds 64(%edx,%eax) 
// CHECK: encoding: [0xdf,0x44,0x02,0x40]        
filds 64(%edx,%eax) 

// CHECK: filds (%edx) 
// CHECK: encoding: [0xdf,0x02]        
filds (%edx) 

// CHECK: fimull -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
fimull -485498096(%edx,%eax,4) 

// CHECK: fimull 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
fimull 485498096(%edx,%eax,4) 

// CHECK: fimull 485498096(%edx) 
// CHECK: encoding: [0xda,0x8a,0xf0,0x1c,0xf0,0x1c]        
fimull 485498096(%edx) 

// CHECK: fimull 485498096 
// CHECK: encoding: [0xda,0x0d,0xf0,0x1c,0xf0,0x1c]        
fimull 485498096 

// CHECK: fimull 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x4c,0x02,0x40]        
fimull 64(%edx,%eax) 

// CHECK: fimull (%edx) 
// CHECK: encoding: [0xda,0x0a]        
fimull (%edx) 

// CHECK: fimuls -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
fimuls -485498096(%edx,%eax,4) 

// CHECK: fimuls 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
fimuls 485498096(%edx,%eax,4) 

// CHECK: fimuls 485498096(%edx) 
// CHECK: encoding: [0xde,0x8a,0xf0,0x1c,0xf0,0x1c]        
fimuls 485498096(%edx) 

// CHECK: fimuls 485498096 
// CHECK: encoding: [0xde,0x0d,0xf0,0x1c,0xf0,0x1c]        
fimuls 485498096 

// CHECK: fimuls 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x4c,0x02,0x40]        
fimuls 64(%edx,%eax) 

// CHECK: fimuls (%edx) 
// CHECK: encoding: [0xde,0x0a]        
fimuls (%edx) 

// CHECK: fincstp 
// CHECK: encoding: [0xd9,0xf7]         
fincstp 

// CHECK: fistl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
fistl -485498096(%edx,%eax,4) 

// CHECK: fistl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
fistl 485498096(%edx,%eax,4) 

// CHECK: fistl 485498096(%edx) 
// CHECK: encoding: [0xdb,0x92,0xf0,0x1c,0xf0,0x1c]        
fistl 485498096(%edx) 

// CHECK: fistl 485498096 
// CHECK: encoding: [0xdb,0x15,0xf0,0x1c,0xf0,0x1c]        
fistl 485498096 

// CHECK: fistl 64(%edx,%eax) 
// CHECK: encoding: [0xdb,0x54,0x02,0x40]        
fistl 64(%edx,%eax) 

// CHECK: fistl (%edx) 
// CHECK: encoding: [0xdb,0x12]        
fistl (%edx) 

// CHECK: fistpl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
fistpl -485498096(%edx,%eax,4) 

// CHECK: fistpl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
fistpl 485498096(%edx,%eax,4) 

// CHECK: fistpl 485498096(%edx) 
// CHECK: encoding: [0xdb,0x9a,0xf0,0x1c,0xf0,0x1c]        
fistpl 485498096(%edx) 

// CHECK: fistpl 485498096 
// CHECK: encoding: [0xdb,0x1d,0xf0,0x1c,0xf0,0x1c]        
fistpl 485498096 

// CHECK: fistpl 64(%edx,%eax) 
// CHECK: encoding: [0xdb,0x5c,0x02,0x40]        
fistpl 64(%edx,%eax) 

// CHECK: fistpl (%edx) 
// CHECK: encoding: [0xdb,0x1a]        
fistpl (%edx) 

// CHECK: fistpll -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fistpll -485498096(%edx,%eax,4) 

// CHECK: fistpll 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fistpll 485498096(%edx,%eax,4) 

// CHECK: fistpll 485498096(%edx) 
// CHECK: encoding: [0xdf,0xba,0xf0,0x1c,0xf0,0x1c]        
fistpll 485498096(%edx) 

// CHECK: fistpll 485498096 
// CHECK: encoding: [0xdf,0x3d,0xf0,0x1c,0xf0,0x1c]        
fistpll 485498096 

// CHECK: fistpll 64(%edx,%eax) 
// CHECK: encoding: [0xdf,0x7c,0x02,0x40]        
fistpll 64(%edx,%eax) 

// CHECK: fistpll (%edx) 
// CHECK: encoding: [0xdf,0x3a]        
fistpll (%edx) 

// CHECK: fistps -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
fistps -485498096(%edx,%eax,4) 

// CHECK: fistps 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
fistps 485498096(%edx,%eax,4) 

// CHECK: fistps 485498096(%edx) 
// CHECK: encoding: [0xdf,0x9a,0xf0,0x1c,0xf0,0x1c]        
fistps 485498096(%edx) 

// CHECK: fistps 485498096 
// CHECK: encoding: [0xdf,0x1d,0xf0,0x1c,0xf0,0x1c]        
fistps 485498096 

// CHECK: fistps 64(%edx,%eax) 
// CHECK: encoding: [0xdf,0x5c,0x02,0x40]        
fistps 64(%edx,%eax) 

// CHECK: fistps (%edx) 
// CHECK: encoding: [0xdf,0x1a]        
fistps (%edx) 

// CHECK: fists -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
fists -485498096(%edx,%eax,4) 

// CHECK: fists 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdf,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
fists 485498096(%edx,%eax,4) 

// CHECK: fists 485498096(%edx) 
// CHECK: encoding: [0xdf,0x92,0xf0,0x1c,0xf0,0x1c]        
fists 485498096(%edx) 

// CHECK: fists 485498096 
// CHECK: encoding: [0xdf,0x15,0xf0,0x1c,0xf0,0x1c]        
fists 485498096 

// CHECK: fists 64(%edx,%eax) 
// CHECK: encoding: [0xdf,0x54,0x02,0x40]        
fists 64(%edx,%eax) 

// CHECK: fists (%edx) 
// CHECK: encoding: [0xdf,0x12]        
fists (%edx) 

// CHECK: fisubl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
fisubl -485498096(%edx,%eax,4) 

// CHECK: fisubl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
fisubl 485498096(%edx,%eax,4) 

// CHECK: fisubl 485498096(%edx) 
// CHECK: encoding: [0xda,0xa2,0xf0,0x1c,0xf0,0x1c]        
fisubl 485498096(%edx) 

// CHECK: fisubl 485498096 
// CHECK: encoding: [0xda,0x25,0xf0,0x1c,0xf0,0x1c]        
fisubl 485498096 

// CHECK: fisubl 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x64,0x02,0x40]        
fisubl 64(%edx,%eax) 

// CHECK: fisubl (%edx) 
// CHECK: encoding: [0xda,0x22]        
fisubl (%edx) 

// CHECK: fisubrl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
fisubrl -485498096(%edx,%eax,4) 

// CHECK: fisubrl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xda,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
fisubrl 485498096(%edx,%eax,4) 

// CHECK: fisubrl 485498096(%edx) 
// CHECK: encoding: [0xda,0xaa,0xf0,0x1c,0xf0,0x1c]        
fisubrl 485498096(%edx) 

// CHECK: fisubrl 485498096 
// CHECK: encoding: [0xda,0x2d,0xf0,0x1c,0xf0,0x1c]        
fisubrl 485498096 

// CHECK: fisubrl 64(%edx,%eax) 
// CHECK: encoding: [0xda,0x6c,0x02,0x40]        
fisubrl 64(%edx,%eax) 

// CHECK: fisubrl (%edx) 
// CHECK: encoding: [0xda,0x2a]        
fisubrl (%edx) 

// CHECK: fisubrs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
fisubrs -485498096(%edx,%eax,4) 

// CHECK: fisubrs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
fisubrs 485498096(%edx,%eax,4) 

// CHECK: fisubrs 485498096(%edx) 
// CHECK: encoding: [0xde,0xaa,0xf0,0x1c,0xf0,0x1c]        
fisubrs 485498096(%edx) 

// CHECK: fisubrs 485498096 
// CHECK: encoding: [0xde,0x2d,0xf0,0x1c,0xf0,0x1c]        
fisubrs 485498096 

// CHECK: fisubrs 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x6c,0x02,0x40]        
fisubrs 64(%edx,%eax) 

// CHECK: fisubrs (%edx) 
// CHECK: encoding: [0xde,0x2a]        
fisubrs (%edx) 

// CHECK: fisubs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
fisubs -485498096(%edx,%eax,4) 

// CHECK: fisubs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xde,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
fisubs 485498096(%edx,%eax,4) 

// CHECK: fisubs 485498096(%edx) 
// CHECK: encoding: [0xde,0xa2,0xf0,0x1c,0xf0,0x1c]        
fisubs 485498096(%edx) 

// CHECK: fisubs 485498096 
// CHECK: encoding: [0xde,0x25,0xf0,0x1c,0xf0,0x1c]        
fisubs 485498096 

// CHECK: fisubs 64(%edx,%eax) 
// CHECK: encoding: [0xde,0x64,0x02,0x40]        
fisubs 64(%edx,%eax) 

// CHECK: fisubs (%edx) 
// CHECK: encoding: [0xde,0x22]        
fisubs (%edx) 

// CHECK: fld1 
// CHECK: encoding: [0xd9,0xe8]         
fld1 

// CHECK: fldcw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
fldcw -485498096(%edx,%eax,4) 

// CHECK: fldcw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
fldcw 485498096(%edx,%eax,4) 

// CHECK: fldcw 485498096(%edx) 
// CHECK: encoding: [0xd9,0xaa,0xf0,0x1c,0xf0,0x1c]        
fldcw 485498096(%edx) 

// CHECK: fldcw 485498096 
// CHECK: encoding: [0xd9,0x2d,0xf0,0x1c,0xf0,0x1c]        
fldcw 485498096 

// CHECK: fldcw 64(%edx,%eax) 
// CHECK: encoding: [0xd9,0x6c,0x02,0x40]        
fldcw 64(%edx,%eax) 

// CHECK: fldcw (%edx) 
// CHECK: encoding: [0xd9,0x2a]        
fldcw (%edx) 

// CHECK: fldenv -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
fldenv -485498096(%edx,%eax,4) 

// CHECK: fldenv 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
fldenv 485498096(%edx,%eax,4) 

// CHECK: fldenv 485498096(%edx) 
// CHECK: encoding: [0xd9,0xa2,0xf0,0x1c,0xf0,0x1c]        
fldenv 485498096(%edx) 

// CHECK: fldenv 485498096 
// CHECK: encoding: [0xd9,0x25,0xf0,0x1c,0xf0,0x1c]        
fldenv 485498096 

// CHECK: fldenv 64(%edx,%eax) 
// CHECK: encoding: [0xd9,0x64,0x02,0x40]        
fldenv 64(%edx,%eax) 

// CHECK: fldenv (%edx) 
// CHECK: encoding: [0xd9,0x22]        
fldenv (%edx) 

// CHECK: fldl2e 
// CHECK: encoding: [0xd9,0xea]         
fldl2e 

// CHECK: fldl2t 
// CHECK: encoding: [0xd9,0xe9]         
fldl2t 

// CHECK: fldl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
fldl -485498096(%edx,%eax,4) 

// CHECK: fldl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
fldl 485498096(%edx,%eax,4) 

// CHECK: fldl 485498096(%edx) 
// CHECK: encoding: [0xdd,0x82,0xf0,0x1c,0xf0,0x1c]        
fldl 485498096(%edx) 

// CHECK: fldl 485498096 
// CHECK: encoding: [0xdd,0x05,0xf0,0x1c,0xf0,0x1c]        
fldl 485498096 

// CHECK: fldl 64(%edx,%eax) 
// CHECK: encoding: [0xdd,0x44,0x02,0x40]        
fldl 64(%edx,%eax) 

// CHECK: fldl (%edx) 
// CHECK: encoding: [0xdd,0x02]        
fldl (%edx) 

// CHECK: fldlg2 
// CHECK: encoding: [0xd9,0xec]         
fldlg2 

// CHECK: fldln2 
// CHECK: encoding: [0xd9,0xed]         
fldln2 

// CHECK: fldpi 
// CHECK: encoding: [0xd9,0xeb]         
fldpi 

// CHECK: flds -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0x84,0x82,0x10,0xe3,0x0f,0xe3]        
flds -485498096(%edx,%eax,4) 

// CHECK: flds 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0x84,0x82,0xf0,0x1c,0xf0,0x1c]        
flds 485498096(%edx,%eax,4) 

// CHECK: flds 485498096(%edx) 
// CHECK: encoding: [0xd9,0x82,0xf0,0x1c,0xf0,0x1c]        
flds 485498096(%edx) 

// CHECK: flds 485498096 
// CHECK: encoding: [0xd9,0x05,0xf0,0x1c,0xf0,0x1c]        
flds 485498096 

// CHECK: flds 64(%edx,%eax) 
// CHECK: encoding: [0xd9,0x44,0x02,0x40]        
flds 64(%edx,%eax) 

// CHECK: flds (%edx) 
// CHECK: encoding: [0xd9,0x02]        
flds (%edx) 

// CHECK: fld %st(4) 
// CHECK: encoding: [0xd9,0xc4]        
fld %st(4) 

// CHECK: fldt -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
fldt -485498096(%edx,%eax,4) 

// CHECK: fldt 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
fldt 485498096(%edx,%eax,4) 

// CHECK: fldt 485498096(%edx) 
// CHECK: encoding: [0xdb,0xaa,0xf0,0x1c,0xf0,0x1c]        
fldt 485498096(%edx) 

// CHECK: fldt 485498096 
// CHECK: encoding: [0xdb,0x2d,0xf0,0x1c,0xf0,0x1c]        
fldt 485498096 

// CHECK: fldt 64(%edx,%eax) 
// CHECK: encoding: [0xdb,0x6c,0x02,0x40]        
fldt 64(%edx,%eax) 

// CHECK: fldt (%edx) 
// CHECK: encoding: [0xdb,0x2a]        
fldt (%edx) 

// CHECK: fldz 
// CHECK: encoding: [0xd9,0xee]         
fldz 

// CHECK: fmull -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
fmull -485498096(%edx,%eax,4) 

// CHECK: fmull 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
fmull 485498096(%edx,%eax,4) 

// CHECK: fmull 485498096(%edx) 
// CHECK: encoding: [0xdc,0x8a,0xf0,0x1c,0xf0,0x1c]        
fmull 485498096(%edx) 

// CHECK: fmull 485498096 
// CHECK: encoding: [0xdc,0x0d,0xf0,0x1c,0xf0,0x1c]        
fmull 485498096 

// CHECK: fmull 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x4c,0x02,0x40]        
fmull 64(%edx,%eax) 

// CHECK: fmull (%edx) 
// CHECK: encoding: [0xdc,0x0a]        
fmull (%edx) 

// CHECK: fmulp %st(4) 
// CHECK: encoding: [0xde,0xcc]        
fmulp %st(4) 

// CHECK: fmuls -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
fmuls -485498096(%edx,%eax,4) 

// CHECK: fmuls 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
fmuls 485498096(%edx,%eax,4) 

// CHECK: fmuls 485498096(%edx) 
// CHECK: encoding: [0xd8,0x8a,0xf0,0x1c,0xf0,0x1c]        
fmuls 485498096(%edx) 

// CHECK: fmuls 485498096 
// CHECK: encoding: [0xd8,0x0d,0xf0,0x1c,0xf0,0x1c]        
fmuls 485498096 

// CHECK: fmuls 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x4c,0x02,0x40]        
fmuls 64(%edx,%eax) 

// CHECK: fmuls (%edx) 
// CHECK: encoding: [0xd8,0x0a]        
fmuls (%edx) 

// CHECK: fmul %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xcc]       
fmul %st(0), %st(4) 

// CHECK: fmul %st(4) 
// CHECK: encoding: [0xd8,0xcc]        
fmul %st(4) 

// CHECK: fnclex 
// CHECK: encoding: [0xdb,0xe2]         
fnclex 

// CHECK: fninit 
// CHECK: encoding: [0xdb,0xe3]         
fninit 

// CHECK: fnop 
// CHECK: encoding: [0xd9,0xd0]         
fnop 

// CHECK: fnsave -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0xb4,0x82,0x10,0xe3,0x0f,0xe3]        
fnsave -485498096(%edx,%eax,4) 

// CHECK: fnsave 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]        
fnsave 485498096(%edx,%eax,4) 

// CHECK: fnsave 485498096(%edx) 
// CHECK: encoding: [0xdd,0xb2,0xf0,0x1c,0xf0,0x1c]        
fnsave 485498096(%edx) 

// CHECK: fnsave 485498096 
// CHECK: encoding: [0xdd,0x35,0xf0,0x1c,0xf0,0x1c]        
fnsave 485498096 

// CHECK: fnsave 64(%edx,%eax) 
// CHECK: encoding: [0xdd,0x74,0x02,0x40]        
fnsave 64(%edx,%eax) 

// CHECK: fnsave (%edx) 
// CHECK: encoding: [0xdd,0x32]        
fnsave (%edx) 

// CHECK: fnstcw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fnstcw -485498096(%edx,%eax,4) 

// CHECK: fnstcw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fnstcw 485498096(%edx,%eax,4) 

// CHECK: fnstcw 485498096(%edx) 
// CHECK: encoding: [0xd9,0xba,0xf0,0x1c,0xf0,0x1c]        
fnstcw 485498096(%edx) 

// CHECK: fnstcw 485498096 
// CHECK: encoding: [0xd9,0x3d,0xf0,0x1c,0xf0,0x1c]        
fnstcw 485498096 

// CHECK: fnstcw 64(%edx,%eax) 
// CHECK: encoding: [0xd9,0x7c,0x02,0x40]        
fnstcw 64(%edx,%eax) 

// CHECK: fnstcw (%edx) 
// CHECK: encoding: [0xd9,0x3a]        
fnstcw (%edx) 

// CHECK: fnstenv -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xb4,0x82,0x10,0xe3,0x0f,0xe3]        
fnstenv -485498096(%edx,%eax,4) 

// CHECK: fnstenv 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0xb4,0x82,0xf0,0x1c,0xf0,0x1c]        
fnstenv 485498096(%edx,%eax,4) 

// CHECK: fnstenv 485498096(%edx) 
// CHECK: encoding: [0xd9,0xb2,0xf0,0x1c,0xf0,0x1c]        
fnstenv 485498096(%edx) 

// CHECK: fnstenv 485498096 
// CHECK: encoding: [0xd9,0x35,0xf0,0x1c,0xf0,0x1c]        
fnstenv 485498096 

// CHECK: fnstenv 64(%edx,%eax) 
// CHECK: encoding: [0xd9,0x74,0x02,0x40]        
fnstenv 64(%edx,%eax) 

// CHECK: fnstenv (%edx) 
// CHECK: encoding: [0xd9,0x32]        
fnstenv (%edx) 

// CHECK: fnstsw -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fnstsw -485498096(%edx,%eax,4) 

// CHECK: fnstsw 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fnstsw 485498096(%edx,%eax,4) 

// CHECK: fnstsw 485498096(%edx) 
// CHECK: encoding: [0xdd,0xba,0xf0,0x1c,0xf0,0x1c]        
fnstsw 485498096(%edx) 

// CHECK: fnstsw 485498096 
// CHECK: encoding: [0xdd,0x3d,0xf0,0x1c,0xf0,0x1c]        
fnstsw 485498096 

// CHECK: fnstsw 64(%edx,%eax) 
// CHECK: encoding: [0xdd,0x7c,0x02,0x40]        
fnstsw 64(%edx,%eax) 

// CHECK: fnstsw %ax 
// CHECK: encoding: [0xdf,0xe0]        
fnstsw %ax 

// CHECK: fnstsw (%edx) 
// CHECK: encoding: [0xdd,0x3a]        
fnstsw (%edx) 

// CHECK: fpatan 
// CHECK: encoding: [0xd9,0xf3]         
fpatan 

// CHECK: fprem1 
// CHECK: encoding: [0xd9,0xf5]         
fprem1 

// CHECK: fprem 
// CHECK: encoding: [0xd9,0xf8]         
fprem 

// CHECK: fptan 
// CHECK: encoding: [0xd9,0xf2]         
fptan 

// CHECK: frndint 
// CHECK: encoding: [0xd9,0xfc]         
frndint 

// CHECK: frstor -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
frstor -485498096(%edx,%eax,4) 

// CHECK: frstor 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
frstor 485498096(%edx,%eax,4) 

// CHECK: frstor 485498096(%edx) 
// CHECK: encoding: [0xdd,0xa2,0xf0,0x1c,0xf0,0x1c]        
frstor 485498096(%edx) 

// CHECK: frstor 485498096 
// CHECK: encoding: [0xdd,0x25,0xf0,0x1c,0xf0,0x1c]        
frstor 485498096 

// CHECK: frstor 64(%edx,%eax) 
// CHECK: encoding: [0xdd,0x64,0x02,0x40]        
frstor 64(%edx,%eax) 

// CHECK: frstor (%edx) 
// CHECK: encoding: [0xdd,0x22]        
frstor (%edx) 

// CHECK: fscale 
// CHECK: encoding: [0xd9,0xfd]         
fscale 

// CHECK: fsincos 
// CHECK: encoding: [0xd9,0xfb]         
fsincos 

// CHECK: fsin 
// CHECK: encoding: [0xd9,0xfe]         
fsin 

// CHECK: fsqrt 
// CHECK: encoding: [0xd9,0xfa]         
fsqrt 

// CHECK: fstl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
fstl -485498096(%edx,%eax,4) 

// CHECK: fstl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
fstl 485498096(%edx,%eax,4) 

// CHECK: fstl 485498096(%edx) 
// CHECK: encoding: [0xdd,0x92,0xf0,0x1c,0xf0,0x1c]        
fstl 485498096(%edx) 

// CHECK: fstl 485498096 
// CHECK: encoding: [0xdd,0x15,0xf0,0x1c,0xf0,0x1c]        
fstl 485498096 

// CHECK: fstl 64(%edx,%eax) 
// CHECK: encoding: [0xdd,0x54,0x02,0x40]        
fstl 64(%edx,%eax) 

// CHECK: fstl (%edx) 
// CHECK: encoding: [0xdd,0x12]        
fstl (%edx) 

// CHECK: fstpl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
fstpl -485498096(%edx,%eax,4) 

// CHECK: fstpl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdd,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
fstpl 485498096(%edx,%eax,4) 

// CHECK: fstpl 485498096(%edx) 
// CHECK: encoding: [0xdd,0x9a,0xf0,0x1c,0xf0,0x1c]        
fstpl 485498096(%edx) 

// CHECK: fstpl 485498096 
// CHECK: encoding: [0xdd,0x1d,0xf0,0x1c,0xf0,0x1c]        
fstpl 485498096 

// CHECK: fstpl 64(%edx,%eax) 
// CHECK: encoding: [0xdd,0x5c,0x02,0x40]        
fstpl 64(%edx,%eax) 

// CHECK: fstpl (%edx) 
// CHECK: encoding: [0xdd,0x1a]        
fstpl (%edx) 

// CHECK: fstps -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0x9c,0x82,0x10,0xe3,0x0f,0xe3]        
fstps -485498096(%edx,%eax,4) 

// CHECK: fstps 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0x9c,0x82,0xf0,0x1c,0xf0,0x1c]        
fstps 485498096(%edx,%eax,4) 

// CHECK: fstps 485498096(%edx) 
// CHECK: encoding: [0xd9,0x9a,0xf0,0x1c,0xf0,0x1c]        
fstps 485498096(%edx) 

// CHECK: fstps 485498096 
// CHECK: encoding: [0xd9,0x1d,0xf0,0x1c,0xf0,0x1c]        
fstps 485498096 

// CHECK: fstps 64(%edx,%eax) 
// CHECK: encoding: [0xd9,0x5c,0x02,0x40]        
fstps 64(%edx,%eax) 

// CHECK: fstps (%edx) 
// CHECK: encoding: [0xd9,0x1a]        
fstps (%edx) 

// CHECK: fstp %st(4) 
// CHECK: encoding: [0xdd,0xdc]        
fstp %st(4) 

// CHECK: fstpt -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0xbc,0x82,0x10,0xe3,0x0f,0xe3]        
fstpt -485498096(%edx,%eax,4) 

// CHECK: fstpt 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdb,0xbc,0x82,0xf0,0x1c,0xf0,0x1c]        
fstpt 485498096(%edx,%eax,4) 

// CHECK: fstpt 485498096(%edx) 
// CHECK: encoding: [0xdb,0xba,0xf0,0x1c,0xf0,0x1c]        
fstpt 485498096(%edx) 

// CHECK: fstpt 485498096 
// CHECK: encoding: [0xdb,0x3d,0xf0,0x1c,0xf0,0x1c]        
fstpt 485498096 

// CHECK: fstpt 64(%edx,%eax) 
// CHECK: encoding: [0xdb,0x7c,0x02,0x40]        
fstpt 64(%edx,%eax) 

// CHECK: fstpt (%edx) 
// CHECK: encoding: [0xdb,0x3a]        
fstpt (%edx) 

// CHECK: fsts -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0x94,0x82,0x10,0xe3,0x0f,0xe3]        
fsts -485498096(%edx,%eax,4) 

// CHECK: fsts 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd9,0x94,0x82,0xf0,0x1c,0xf0,0x1c]        
fsts 485498096(%edx,%eax,4) 

// CHECK: fsts 485498096(%edx) 
// CHECK: encoding: [0xd9,0x92,0xf0,0x1c,0xf0,0x1c]        
fsts 485498096(%edx) 

// CHECK: fsts 485498096 
// CHECK: encoding: [0xd9,0x15,0xf0,0x1c,0xf0,0x1c]        
fsts 485498096 

// CHECK: fsts 64(%edx,%eax) 
// CHECK: encoding: [0xd9,0x54,0x02,0x40]        
fsts 64(%edx,%eax) 

// CHECK: fsts (%edx) 
// CHECK: encoding: [0xd9,0x12]        
fsts (%edx) 

// CHECK: fst %st(4) 
// CHECK: encoding: [0xdd,0xd4]        
fst %st(4) 

// CHECK: fsubl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
fsubl -485498096(%edx,%eax,4) 

// CHECK: fsubl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
fsubl 485498096(%edx,%eax,4) 

// CHECK: fsubl 485498096(%edx) 
// CHECK: encoding: [0xdc,0xa2,0xf0,0x1c,0xf0,0x1c]        
fsubl 485498096(%edx) 

// CHECK: fsubl 485498096 
// CHECK: encoding: [0xdc,0x25,0xf0,0x1c,0xf0,0x1c]        
fsubl 485498096 

// CHECK: fsubl 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x64,0x02,0x40]        
fsubl 64(%edx,%eax) 

// CHECK: fsubl (%edx) 
// CHECK: encoding: [0xdc,0x22]        
fsubl (%edx) 

// CHECK: fsubp %st(4) 
// CHECK: encoding: [0xde,0xe4]        
fsubp %st(4) 

// CHECK: fsubrl -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
fsubrl -485498096(%edx,%eax,4) 

// CHECK: fsubrl 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xdc,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
fsubrl 485498096(%edx,%eax,4) 

// CHECK: fsubrl 485498096(%edx) 
// CHECK: encoding: [0xdc,0xaa,0xf0,0x1c,0xf0,0x1c]        
fsubrl 485498096(%edx) 

// CHECK: fsubrl 485498096 
// CHECK: encoding: [0xdc,0x2d,0xf0,0x1c,0xf0,0x1c]        
fsubrl 485498096 

// CHECK: fsubrl 64(%edx,%eax) 
// CHECK: encoding: [0xdc,0x6c,0x02,0x40]        
fsubrl 64(%edx,%eax) 

// CHECK: fsubrl (%edx) 
// CHECK: encoding: [0xdc,0x2a]        
fsubrl (%edx) 

// CHECK: fsubrp %st(4) 
// CHECK: encoding: [0xde,0xec]        
fsubrp %st(4) 

// CHECK: fsubrs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xac,0x82,0x10,0xe3,0x0f,0xe3]        
fsubrs -485498096(%edx,%eax,4) 

// CHECK: fsubrs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xac,0x82,0xf0,0x1c,0xf0,0x1c]        
fsubrs 485498096(%edx,%eax,4) 

// CHECK: fsubrs 485498096(%edx) 
// CHECK: encoding: [0xd8,0xaa,0xf0,0x1c,0xf0,0x1c]        
fsubrs 485498096(%edx) 

// CHECK: fsubrs 485498096 
// CHECK: encoding: [0xd8,0x2d,0xf0,0x1c,0xf0,0x1c]        
fsubrs 485498096 

// CHECK: fsubrs 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x6c,0x02,0x40]        
fsubrs 64(%edx,%eax) 

// CHECK: fsubrs (%edx) 
// CHECK: encoding: [0xd8,0x2a]        
fsubrs (%edx) 

// CHECK: fsubr %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xec]       
fsubr %st(0), %st(4) 

// CHECK: fsubr %st(4) 
// CHECK: encoding: [0xd8,0xec]        
fsubr %st(4) 

// CHECK: fsubs -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]        
fsubs -485498096(%edx,%eax,4) 

// CHECK: fsubs 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xd8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]        
fsubs 485498096(%edx,%eax,4) 

// CHECK: fsubs 485498096(%edx) 
// CHECK: encoding: [0xd8,0xa2,0xf0,0x1c,0xf0,0x1c]        
fsubs 485498096(%edx) 

// CHECK: fsubs 485498096 
// CHECK: encoding: [0xd8,0x25,0xf0,0x1c,0xf0,0x1c]        
fsubs 485498096 

// CHECK: fsubs 64(%edx,%eax) 
// CHECK: encoding: [0xd8,0x64,0x02,0x40]        
fsubs 64(%edx,%eax) 

// CHECK: fsubs (%edx) 
// CHECK: encoding: [0xd8,0x22]        
fsubs (%edx) 

// CHECK: fsub %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xe4]       
fsub %st(0), %st(4) 

// CHECK: fsub %st(4) 
// CHECK: encoding: [0xd8,0xe4]        
fsub %st(4) 

// CHECK: ftst 
// CHECK: encoding: [0xd9,0xe4]         
ftst 

// CHECK: fucompp 
// CHECK: encoding: [0xda,0xe9]         
fucompp 

// CHECK: fucomp %st(4) 
// CHECK: encoding: [0xdd,0xec]        
fucomp %st(4) 

// CHECK: fucom %st(4) 
// CHECK: encoding: [0xdd,0xe4]        
fucom %st(4) 

// CHECK: fxam 
// CHECK: encoding: [0xd9,0xe5]         
fxam 

// CHECK: fxch %st(4) 
// CHECK: encoding: [0xd9,0xcc]        
fxch %st(4) 

// CHECK: fxtract 
// CHECK: encoding: [0xd9,0xf4]         
fxtract 

// CHECK: fyl2x 
// CHECK: encoding: [0xd9,0xf1]         
fyl2x 

// CHECK: fyl2xp1 
// CHECK: encoding: [0xd9,0xf9]         
fyl2xp1 


// CHECK: wait 
// CHECK: encoding: [0x9b]         
wait 

