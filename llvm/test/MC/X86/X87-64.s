// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s
// CHECK: f2xm1 
// CHECK: encoding: [0xd9,0xf0]         
f2xm1 

// CHECK: fabs 
// CHECK: encoding: [0xd9,0xe1]         
fabs 

// CHECK: faddl 485498096 
// CHECK: encoding: [0xdc,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
faddl 485498096 

// CHECK: faddl 64(%rdx) 
// CHECK: encoding: [0xdc,0x42,0x40]        
faddl 64(%rdx) 

// CHECK: faddl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x44,0x82,0xc0]        
faddl -64(%rdx,%rax,4) 

// CHECK: faddl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x44,0x82,0x40]        
faddl 64(%rdx,%rax,4) 

// CHECK: faddl 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x44,0x02,0x40]        
faddl 64(%rdx,%rax) 

// CHECK: faddl (%rdx) 
// CHECK: encoding: [0xdc,0x02]        
faddl (%rdx) 

// CHECK: faddp %st(4) 
// CHECK: encoding: [0xde,0xc4]        
faddp %st(4) 

// CHECK: fadds 485498096 
// CHECK: encoding: [0xd8,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
fadds 485498096 

// CHECK: fadds 64(%rdx) 
// CHECK: encoding: [0xd8,0x42,0x40]        
fadds 64(%rdx) 

// CHECK: fadds -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x44,0x82,0xc0]        
fadds -64(%rdx,%rax,4) 

// CHECK: fadds 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x44,0x82,0x40]        
fadds 64(%rdx,%rax,4) 

// CHECK: fadds 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x44,0x02,0x40]        
fadds 64(%rdx,%rax) 

// CHECK: fadds (%rdx) 
// CHECK: encoding: [0xd8,0x02]        
fadds (%rdx) 

// CHECK: fadd %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xc4]       
fadd %st(0), %st(4) 

// CHECK: fadd %st(4) 
// CHECK: encoding: [0xd8,0xc4]        
fadd %st(4) 

// CHECK: fbld 485498096 
// CHECK: encoding: [0xdf,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
fbld 485498096 

// CHECK: fbld 64(%rdx) 
// CHECK: encoding: [0xdf,0x62,0x40]        
fbld 64(%rdx) 

// CHECK: fbld -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x64,0x82,0xc0]        
fbld -64(%rdx,%rax,4) 

// CHECK: fbld 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x64,0x82,0x40]        
fbld 64(%rdx,%rax,4) 

// CHECK: fbld 64(%rdx,%rax) 
// CHECK: encoding: [0xdf,0x64,0x02,0x40]        
fbld 64(%rdx,%rax) 

// CHECK: fbld (%rdx) 
// CHECK: encoding: [0xdf,0x22]        
fbld (%rdx) 

// CHECK: fbstp 485498096 
// CHECK: encoding: [0xdf,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
fbstp 485498096 

// CHECK: fbstp 64(%rdx) 
// CHECK: encoding: [0xdf,0x72,0x40]        
fbstp 64(%rdx) 

// CHECK: fbstp -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x74,0x82,0xc0]        
fbstp -64(%rdx,%rax,4) 

// CHECK: fbstp 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x74,0x82,0x40]        
fbstp 64(%rdx,%rax,4) 

// CHECK: fbstp 64(%rdx,%rax) 
// CHECK: encoding: [0xdf,0x74,0x02,0x40]        
fbstp 64(%rdx,%rax) 

// CHECK: fbstp (%rdx) 
// CHECK: encoding: [0xdf,0x32]        
fbstp (%rdx) 

// CHECK: fchs 
// CHECK: encoding: [0xd9,0xe0]         
fchs 

// CHECK: fcoml 485498096 
// CHECK: encoding: [0xdc,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
fcoml 485498096 

// CHECK: fcoml 64(%rdx) 
// CHECK: encoding: [0xdc,0x52,0x40]        
fcoml 64(%rdx) 

// CHECK: fcoml -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x54,0x82,0xc0]        
fcoml -64(%rdx,%rax,4) 

// CHECK: fcoml 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x54,0x82,0x40]        
fcoml 64(%rdx,%rax,4) 

// CHECK: fcoml 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x54,0x02,0x40]        
fcoml 64(%rdx,%rax) 

// CHECK: fcoml (%rdx) 
// CHECK: encoding: [0xdc,0x12]        
fcoml (%rdx) 

// CHECK: fcompl 485498096 
// CHECK: encoding: [0xdc,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
fcompl 485498096 

// CHECK: fcompl 64(%rdx) 
// CHECK: encoding: [0xdc,0x5a,0x40]        
fcompl 64(%rdx) 

// CHECK: fcompl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x5c,0x82,0xc0]        
fcompl -64(%rdx,%rax,4) 

// CHECK: fcompl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x5c,0x82,0x40]        
fcompl 64(%rdx,%rax,4) 

// CHECK: fcompl 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x5c,0x02,0x40]        
fcompl 64(%rdx,%rax) 

// CHECK: fcompl (%rdx) 
// CHECK: encoding: [0xdc,0x1a]        
fcompl (%rdx) 

// CHECK: fcompp 
// CHECK: encoding: [0xde,0xd9]         
fcompp 

// CHECK: fcomps 485498096 
// CHECK: encoding: [0xd8,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
fcomps 485498096 

// CHECK: fcomps 64(%rdx) 
// CHECK: encoding: [0xd8,0x5a,0x40]        
fcomps 64(%rdx) 

// CHECK: fcomps -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x5c,0x82,0xc0]        
fcomps -64(%rdx,%rax,4) 

// CHECK: fcomps 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x5c,0x82,0x40]        
fcomps 64(%rdx,%rax,4) 

// CHECK: fcomps 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x5c,0x02,0x40]        
fcomps 64(%rdx,%rax) 

// CHECK: fcomps (%rdx) 
// CHECK: encoding: [0xd8,0x1a]        
fcomps (%rdx) 

// CHECK: fcomp %st(4) 
// CHECK: encoding: [0xd8,0xdc]        
fcomp %st(4) 

// CHECK: fcoms 485498096 
// CHECK: encoding: [0xd8,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
fcoms 485498096 

// CHECK: fcoms 64(%rdx) 
// CHECK: encoding: [0xd8,0x52,0x40]        
fcoms 64(%rdx) 

// CHECK: fcoms -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x54,0x82,0xc0]        
fcoms -64(%rdx,%rax,4) 

// CHECK: fcoms 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x54,0x82,0x40]        
fcoms 64(%rdx,%rax,4) 

// CHECK: fcoms 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x54,0x02,0x40]        
fcoms 64(%rdx,%rax) 

// CHECK: fcoms (%rdx) 
// CHECK: encoding: [0xd8,0x12]        
fcoms (%rdx) 

// CHECK: fcom %st(4) 
// CHECK: encoding: [0xd8,0xd4]        
fcom %st(4) 

// CHECK: fcos 
// CHECK: encoding: [0xd9,0xff]         
fcos 

// CHECK: fdecstp 
// CHECK: encoding: [0xd9,0xf6]         
fdecstp 

// CHECK: fdivl 485498096 
// CHECK: encoding: [0xdc,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
fdivl 485498096 

// CHECK: fdivl 64(%rdx) 
// CHECK: encoding: [0xdc,0x72,0x40]        
fdivl 64(%rdx) 

// CHECK: fdivl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x74,0x82,0xc0]        
fdivl -64(%rdx,%rax,4) 

// CHECK: fdivl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x74,0x82,0x40]        
fdivl 64(%rdx,%rax,4) 

// CHECK: fdivl 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x74,0x02,0x40]        
fdivl 64(%rdx,%rax) 

// CHECK: fdivl (%rdx) 
// CHECK: encoding: [0xdc,0x32]        
fdivl (%rdx) 

// CHECK: fdivp %st(4) 
// CHECK: encoding: [0xde,0xf4]        
fdivp %st(4) 

// CHECK: fdivrl 485498096 
// CHECK: encoding: [0xdc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fdivrl 485498096 

// CHECK: fdivrl 64(%rdx) 
// CHECK: encoding: [0xdc,0x7a,0x40]        
fdivrl 64(%rdx) 

// CHECK: fdivrl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x7c,0x82,0xc0]        
fdivrl -64(%rdx,%rax,4) 

// CHECK: fdivrl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x7c,0x82,0x40]        
fdivrl 64(%rdx,%rax,4) 

// CHECK: fdivrl 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x7c,0x02,0x40]        
fdivrl 64(%rdx,%rax) 

// CHECK: fdivrl (%rdx) 
// CHECK: encoding: [0xdc,0x3a]        
fdivrl (%rdx) 

// CHECK: fdivrp %st(4) 
// CHECK: encoding: [0xde,0xfc]        
fdivrp %st(4) 

// CHECK: fdivrs 485498096 
// CHECK: encoding: [0xd8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fdivrs 485498096 

// CHECK: fdivrs 64(%rdx) 
// CHECK: encoding: [0xd8,0x7a,0x40]        
fdivrs 64(%rdx) 

// CHECK: fdivrs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x7c,0x82,0xc0]        
fdivrs -64(%rdx,%rax,4) 

// CHECK: fdivrs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x7c,0x82,0x40]        
fdivrs 64(%rdx,%rax,4) 

// CHECK: fdivrs 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x7c,0x02,0x40]        
fdivrs 64(%rdx,%rax) 

// CHECK: fdivrs (%rdx) 
// CHECK: encoding: [0xd8,0x3a]        
fdivrs (%rdx) 

// CHECK: fdivr %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xfc]       
fdivr %st(0), %st(4) 

// CHECK: fdivr %st(4) 
// CHECK: encoding: [0xd8,0xfc]        
fdivr %st(4) 

// CHECK: fdivs 485498096 
// CHECK: encoding: [0xd8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
fdivs 485498096 

// CHECK: fdivs 64(%rdx) 
// CHECK: encoding: [0xd8,0x72,0x40]        
fdivs 64(%rdx) 

// CHECK: fdivs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x74,0x82,0xc0]        
fdivs -64(%rdx,%rax,4) 

// CHECK: fdivs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x74,0x82,0x40]        
fdivs 64(%rdx,%rax,4) 

// CHECK: fdivs 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x74,0x02,0x40]        
fdivs 64(%rdx,%rax) 

// CHECK: fdivs (%rdx) 
// CHECK: encoding: [0xd8,0x32]        
fdivs (%rdx) 

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

// CHECK: fiaddl 485498096 
// CHECK: encoding: [0xda,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
fiaddl 485498096 

// CHECK: fiaddl 64(%rdx) 
// CHECK: encoding: [0xda,0x42,0x40]        
fiaddl 64(%rdx) 

// CHECK: fiaddl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x44,0x82,0xc0]        
fiaddl -64(%rdx,%rax,4) 

// CHECK: fiaddl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x44,0x82,0x40]        
fiaddl 64(%rdx,%rax,4) 

// CHECK: fiaddl 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x44,0x02,0x40]        
fiaddl 64(%rdx,%rax) 

// CHECK: fiaddl (%rdx) 
// CHECK: encoding: [0xda,0x02]        
fiaddl (%rdx) 

// CHECK: fiadds 485498096 
// CHECK: encoding: [0xde,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
fiadds 485498096 

// CHECK: fiadds 64(%rdx) 
// CHECK: encoding: [0xde,0x42,0x40]        
fiadds 64(%rdx) 

// CHECK: fiadds -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x44,0x82,0xc0]        
fiadds -64(%rdx,%rax,4) 

// CHECK: fiadds 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x44,0x82,0x40]        
fiadds 64(%rdx,%rax,4) 

// CHECK: fiadds 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x44,0x02,0x40]        
fiadds 64(%rdx,%rax) 

// CHECK: fiadds (%rdx) 
// CHECK: encoding: [0xde,0x02]        
fiadds (%rdx) 

// CHECK: ficoml 485498096 
// CHECK: encoding: [0xda,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
ficoml 485498096 

// CHECK: ficoml 64(%rdx) 
// CHECK: encoding: [0xda,0x52,0x40]        
ficoml 64(%rdx) 

// CHECK: ficoml -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x54,0x82,0xc0]        
ficoml -64(%rdx,%rax,4) 

// CHECK: ficoml 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x54,0x82,0x40]        
ficoml 64(%rdx,%rax,4) 

// CHECK: ficoml 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x54,0x02,0x40]        
ficoml 64(%rdx,%rax) 

// CHECK: ficoml (%rdx) 
// CHECK: encoding: [0xda,0x12]        
ficoml (%rdx) 

// CHECK: ficompl 485498096 
// CHECK: encoding: [0xda,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
ficompl 485498096 

// CHECK: ficompl 64(%rdx) 
// CHECK: encoding: [0xda,0x5a,0x40]        
ficompl 64(%rdx) 

// CHECK: ficompl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x5c,0x82,0xc0]        
ficompl -64(%rdx,%rax,4) 

// CHECK: ficompl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x5c,0x82,0x40]        
ficompl 64(%rdx,%rax,4) 

// CHECK: ficompl 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x5c,0x02,0x40]        
ficompl 64(%rdx,%rax) 

// CHECK: ficompl (%rdx) 
// CHECK: encoding: [0xda,0x1a]        
ficompl (%rdx) 

// CHECK: ficomps 485498096 
// CHECK: encoding: [0xde,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
ficomps 485498096 

// CHECK: ficomps 64(%rdx) 
// CHECK: encoding: [0xde,0x5a,0x40]        
ficomps 64(%rdx) 

// CHECK: ficomps -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x5c,0x82,0xc0]        
ficomps -64(%rdx,%rax,4) 

// CHECK: ficomps 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x5c,0x82,0x40]        
ficomps 64(%rdx,%rax,4) 

// CHECK: ficomps 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x5c,0x02,0x40]        
ficomps 64(%rdx,%rax) 

// CHECK: ficomps (%rdx) 
// CHECK: encoding: [0xde,0x1a]        
ficomps (%rdx) 

// CHECK: ficoms 485498096 
// CHECK: encoding: [0xde,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
ficoms 485498096 

// CHECK: ficoms 64(%rdx) 
// CHECK: encoding: [0xde,0x52,0x40]        
ficoms 64(%rdx) 

// CHECK: ficoms -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x54,0x82,0xc0]        
ficoms -64(%rdx,%rax,4) 

// CHECK: ficoms 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x54,0x82,0x40]        
ficoms 64(%rdx,%rax,4) 

// CHECK: ficoms 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x54,0x02,0x40]        
ficoms 64(%rdx,%rax) 

// CHECK: ficoms (%rdx) 
// CHECK: encoding: [0xde,0x12]        
ficoms (%rdx) 

// CHECK: fidivl 485498096 
// CHECK: encoding: [0xda,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
fidivl 485498096 

// CHECK: fidivl 64(%rdx) 
// CHECK: encoding: [0xda,0x72,0x40]        
fidivl 64(%rdx) 

// CHECK: fidivl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x74,0x82,0xc0]        
fidivl -64(%rdx,%rax,4) 

// CHECK: fidivl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x74,0x82,0x40]        
fidivl 64(%rdx,%rax,4) 

// CHECK: fidivl 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x74,0x02,0x40]        
fidivl 64(%rdx,%rax) 

// CHECK: fidivl (%rdx) 
// CHECK: encoding: [0xda,0x32]        
fidivl (%rdx) 

// CHECK: fidivrl 485498096 
// CHECK: encoding: [0xda,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fidivrl 485498096 

// CHECK: fidivrl 64(%rdx) 
// CHECK: encoding: [0xda,0x7a,0x40]        
fidivrl 64(%rdx) 

// CHECK: fidivrl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x7c,0x82,0xc0]        
fidivrl -64(%rdx,%rax,4) 

// CHECK: fidivrl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x7c,0x82,0x40]        
fidivrl 64(%rdx,%rax,4) 

// CHECK: fidivrl 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x7c,0x02,0x40]        
fidivrl 64(%rdx,%rax) 

// CHECK: fidivrl (%rdx) 
// CHECK: encoding: [0xda,0x3a]        
fidivrl (%rdx) 

// CHECK: fidivrs 485498096 
// CHECK: encoding: [0xde,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fidivrs 485498096 

// CHECK: fidivrs 64(%rdx) 
// CHECK: encoding: [0xde,0x7a,0x40]        
fidivrs 64(%rdx) 

// CHECK: fidivrs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x7c,0x82,0xc0]        
fidivrs -64(%rdx,%rax,4) 

// CHECK: fidivrs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x7c,0x82,0x40]        
fidivrs 64(%rdx,%rax,4) 

// CHECK: fidivrs 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x7c,0x02,0x40]        
fidivrs 64(%rdx,%rax) 

// CHECK: fidivrs (%rdx) 
// CHECK: encoding: [0xde,0x3a]        
fidivrs (%rdx) 

// CHECK: fidivs 485498096 
// CHECK: encoding: [0xde,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
fidivs 485498096 

// CHECK: fidivs 64(%rdx) 
// CHECK: encoding: [0xde,0x72,0x40]        
fidivs 64(%rdx) 

// CHECK: fidivs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x74,0x82,0xc0]        
fidivs -64(%rdx,%rax,4) 

// CHECK: fidivs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x74,0x82,0x40]        
fidivs 64(%rdx,%rax,4) 

// CHECK: fidivs 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x74,0x02,0x40]        
fidivs 64(%rdx,%rax) 

// CHECK: fidivs (%rdx) 
// CHECK: encoding: [0xde,0x32]        
fidivs (%rdx) 

// CHECK: fildl 485498096 
// CHECK: encoding: [0xdb,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
fildl 485498096 

// CHECK: fildl 64(%rdx) 
// CHECK: encoding: [0xdb,0x42,0x40]        
fildl 64(%rdx) 

// CHECK: fildl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x44,0x82,0xc0]        
fildl -64(%rdx,%rax,4) 

// CHECK: fildl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x44,0x82,0x40]        
fildl 64(%rdx,%rax,4) 

// CHECK: fildl 64(%rdx,%rax) 
// CHECK: encoding: [0xdb,0x44,0x02,0x40]        
fildl 64(%rdx,%rax) 

// CHECK: fildll 485498096 
// CHECK: encoding: [0xdf,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
fildll 485498096 

// CHECK: fildll 64(%rdx) 
// CHECK: encoding: [0xdf,0x6a,0x40]        
fildll 64(%rdx) 

// CHECK: fildll -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x6c,0x82,0xc0]        
fildll -64(%rdx,%rax,4) 

// CHECK: fildll 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x6c,0x82,0x40]        
fildll 64(%rdx,%rax,4) 

// CHECK: fildll 64(%rdx,%rax) 
// CHECK: encoding: [0xdf,0x6c,0x02,0x40]        
fildll 64(%rdx,%rax) 

// CHECK: fildll (%rdx) 
// CHECK: encoding: [0xdf,0x2a]        
fildll (%rdx) 

// CHECK: fildl (%rdx) 
// CHECK: encoding: [0xdb,0x02]        
fildl (%rdx) 

// CHECK: filds 485498096 
// CHECK: encoding: [0xdf,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
filds 485498096 

// CHECK: filds 64(%rdx) 
// CHECK: encoding: [0xdf,0x42,0x40]        
filds 64(%rdx) 

// CHECK: filds -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x44,0x82,0xc0]        
filds -64(%rdx,%rax,4) 

// CHECK: filds 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x44,0x82,0x40]        
filds 64(%rdx,%rax,4) 

// CHECK: filds 64(%rdx,%rax) 
// CHECK: encoding: [0xdf,0x44,0x02,0x40]        
filds 64(%rdx,%rax) 

// CHECK: filds (%rdx) 
// CHECK: encoding: [0xdf,0x02]        
filds (%rdx) 

// CHECK: fimull 485498096 
// CHECK: encoding: [0xda,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
fimull 485498096 

// CHECK: fimull 64(%rdx) 
// CHECK: encoding: [0xda,0x4a,0x40]        
fimull 64(%rdx) 

// CHECK: fimull -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x4c,0x82,0xc0]        
fimull -64(%rdx,%rax,4) 

// CHECK: fimull 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x4c,0x82,0x40]        
fimull 64(%rdx,%rax,4) 

// CHECK: fimull 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x4c,0x02,0x40]        
fimull 64(%rdx,%rax) 

// CHECK: fimull (%rdx) 
// CHECK: encoding: [0xda,0x0a]        
fimull (%rdx) 

// CHECK: fimuls 485498096 
// CHECK: encoding: [0xde,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
fimuls 485498096 

// CHECK: fimuls 64(%rdx) 
// CHECK: encoding: [0xde,0x4a,0x40]        
fimuls 64(%rdx) 

// CHECK: fimuls -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x4c,0x82,0xc0]        
fimuls -64(%rdx,%rax,4) 

// CHECK: fimuls 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x4c,0x82,0x40]        
fimuls 64(%rdx,%rax,4) 

// CHECK: fimuls 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x4c,0x02,0x40]        
fimuls 64(%rdx,%rax) 

// CHECK: fimuls (%rdx) 
// CHECK: encoding: [0xde,0x0a]        
fimuls (%rdx) 

// CHECK: fincstp 
// CHECK: encoding: [0xd9,0xf7]         
fincstp 

// CHECK: fistl 485498096 
// CHECK: encoding: [0xdb,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
fistl 485498096 

// CHECK: fistl 64(%rdx) 
// CHECK: encoding: [0xdb,0x52,0x40]        
fistl 64(%rdx) 

// CHECK: fistl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x54,0x82,0xc0]        
fistl -64(%rdx,%rax,4) 

// CHECK: fistl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x54,0x82,0x40]        
fistl 64(%rdx,%rax,4) 

// CHECK: fistl 64(%rdx,%rax) 
// CHECK: encoding: [0xdb,0x54,0x02,0x40]        
fistl 64(%rdx,%rax) 

// CHECK: fistl (%rdx) 
// CHECK: encoding: [0xdb,0x12]        
fistl (%rdx) 

// CHECK: fistpl 485498096 
// CHECK: encoding: [0xdb,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
fistpl 485498096 

// CHECK: fistpl 64(%rdx) 
// CHECK: encoding: [0xdb,0x5a,0x40]        
fistpl 64(%rdx) 

// CHECK: fistpl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x5c,0x82,0xc0]        
fistpl -64(%rdx,%rax,4) 

// CHECK: fistpl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x5c,0x82,0x40]        
fistpl 64(%rdx,%rax,4) 

// CHECK: fistpl 64(%rdx,%rax) 
// CHECK: encoding: [0xdb,0x5c,0x02,0x40]        
fistpl 64(%rdx,%rax) 

// CHECK: fistpll 485498096 
// CHECK: encoding: [0xdf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fistpll 485498096 

// CHECK: fistpll 64(%rdx) 
// CHECK: encoding: [0xdf,0x7a,0x40]        
fistpll 64(%rdx) 

// CHECK: fistpll -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x7c,0x82,0xc0]        
fistpll -64(%rdx,%rax,4) 

// CHECK: fistpll 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x7c,0x82,0x40]        
fistpll 64(%rdx,%rax,4) 

// CHECK: fistpll 64(%rdx,%rax) 
// CHECK: encoding: [0xdf,0x7c,0x02,0x40]        
fistpll 64(%rdx,%rax) 

// CHECK: fistpll (%rdx) 
// CHECK: encoding: [0xdf,0x3a]        
fistpll (%rdx) 

// CHECK: fistpl (%rdx) 
// CHECK: encoding: [0xdb,0x1a]        
fistpl (%rdx) 

// CHECK: fistps 485498096 
// CHECK: encoding: [0xdf,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
fistps 485498096 

// CHECK: fistps 64(%rdx) 
// CHECK: encoding: [0xdf,0x5a,0x40]        
fistps 64(%rdx) 

// CHECK: fistps -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x5c,0x82,0xc0]        
fistps -64(%rdx,%rax,4) 

// CHECK: fistps 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x5c,0x82,0x40]        
fistps 64(%rdx,%rax,4) 

// CHECK: fistps 64(%rdx,%rax) 
// CHECK: encoding: [0xdf,0x5c,0x02,0x40]        
fistps 64(%rdx,%rax) 

// CHECK: fistps (%rdx) 
// CHECK: encoding: [0xdf,0x1a]        
fistps (%rdx) 

// CHECK: fists 485498096 
// CHECK: encoding: [0xdf,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
fists 485498096 

// CHECK: fists 64(%rdx) 
// CHECK: encoding: [0xdf,0x52,0x40]        
fists 64(%rdx) 

// CHECK: fists -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x54,0x82,0xc0]        
fists -64(%rdx,%rax,4) 

// CHECK: fists 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdf,0x54,0x82,0x40]        
fists 64(%rdx,%rax,4) 

// CHECK: fists 64(%rdx,%rax) 
// CHECK: encoding: [0xdf,0x54,0x02,0x40]        
fists 64(%rdx,%rax) 

// CHECK: fists (%rdx) 
// CHECK: encoding: [0xdf,0x12]        
fists (%rdx) 

// CHECK: fisubl 485498096 
// CHECK: encoding: [0xda,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
fisubl 485498096 

// CHECK: fisubl 64(%rdx) 
// CHECK: encoding: [0xda,0x62,0x40]        
fisubl 64(%rdx) 

// CHECK: fisubl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x64,0x82,0xc0]        
fisubl -64(%rdx,%rax,4) 

// CHECK: fisubl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x64,0x82,0x40]        
fisubl 64(%rdx,%rax,4) 

// CHECK: fisubl 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x64,0x02,0x40]        
fisubl 64(%rdx,%rax) 

// CHECK: fisubl (%rdx) 
// CHECK: encoding: [0xda,0x22]        
fisubl (%rdx) 

// CHECK: fisubrl 485498096 
// CHECK: encoding: [0xda,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
fisubrl 485498096 

// CHECK: fisubrl 64(%rdx) 
// CHECK: encoding: [0xda,0x6a,0x40]        
fisubrl 64(%rdx) 

// CHECK: fisubrl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x6c,0x82,0xc0]        
fisubrl -64(%rdx,%rax,4) 

// CHECK: fisubrl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xda,0x6c,0x82,0x40]        
fisubrl 64(%rdx,%rax,4) 

// CHECK: fisubrl 64(%rdx,%rax) 
// CHECK: encoding: [0xda,0x6c,0x02,0x40]        
fisubrl 64(%rdx,%rax) 

// CHECK: fisubrl (%rdx) 
// CHECK: encoding: [0xda,0x2a]        
fisubrl (%rdx) 

// CHECK: fisubrs 485498096 
// CHECK: encoding: [0xde,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
fisubrs 485498096 

// CHECK: fisubrs 64(%rdx) 
// CHECK: encoding: [0xde,0x6a,0x40]        
fisubrs 64(%rdx) 

// CHECK: fisubrs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x6c,0x82,0xc0]        
fisubrs -64(%rdx,%rax,4) 

// CHECK: fisubrs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x6c,0x82,0x40]        
fisubrs 64(%rdx,%rax,4) 

// CHECK: fisubrs 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x6c,0x02,0x40]        
fisubrs 64(%rdx,%rax) 

// CHECK: fisubrs (%rdx) 
// CHECK: encoding: [0xde,0x2a]        
fisubrs (%rdx) 

// CHECK: fisubs 485498096 
// CHECK: encoding: [0xde,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
fisubs 485498096 

// CHECK: fisubs 64(%rdx) 
// CHECK: encoding: [0xde,0x62,0x40]        
fisubs 64(%rdx) 

// CHECK: fisubs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x64,0x82,0xc0]        
fisubs -64(%rdx,%rax,4) 

// CHECK: fisubs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xde,0x64,0x82,0x40]        
fisubs 64(%rdx,%rax,4) 

// CHECK: fisubs 64(%rdx,%rax) 
// CHECK: encoding: [0xde,0x64,0x02,0x40]        
fisubs 64(%rdx,%rax) 

// CHECK: fisubs (%rdx) 
// CHECK: encoding: [0xde,0x22]        
fisubs (%rdx) 

// CHECK: fld1 
// CHECK: encoding: [0xd9,0xe8]         
fld1 

// CHECK: fldcw 485498096 
// CHECK: encoding: [0xd9,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
fldcw 485498096 

// CHECK: fldcw 64(%rdx) 
// CHECK: encoding: [0xd9,0x6a,0x40]        
fldcw 64(%rdx) 

// CHECK: fldcw -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x6c,0x82,0xc0]        
fldcw -64(%rdx,%rax,4) 

// CHECK: fldcw 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x6c,0x82,0x40]        
fldcw 64(%rdx,%rax,4) 

// CHECK: fldcw 64(%rdx,%rax) 
// CHECK: encoding: [0xd9,0x6c,0x02,0x40]        
fldcw 64(%rdx,%rax) 

// CHECK: fldcw (%rdx) 
// CHECK: encoding: [0xd9,0x2a]        
fldcw (%rdx) 

// CHECK: fldenv 485498096 
// CHECK: encoding: [0xd9,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
fldenv 485498096 

// CHECK: fldenv 64(%rdx) 
// CHECK: encoding: [0xd9,0x62,0x40]        
fldenv 64(%rdx) 

// CHECK: fldenv -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x64,0x82,0xc0]        
fldenv -64(%rdx,%rax,4) 

// CHECK: fldenv 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x64,0x82,0x40]        
fldenv 64(%rdx,%rax,4) 

// CHECK: fldenv 64(%rdx,%rax) 
// CHECK: encoding: [0xd9,0x64,0x02,0x40]        
fldenv 64(%rdx,%rax) 

// CHECK: fldenv (%rdx) 
// CHECK: encoding: [0xd9,0x22]        
fldenv (%rdx) 

// CHECK: fldl2e 
// CHECK: encoding: [0xd9,0xea]         
fldl2e 

// CHECK: fldl2t 
// CHECK: encoding: [0xd9,0xe9]         
fldl2t 

// CHECK: fldl 485498096 
// CHECK: encoding: [0xdd,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
fldl 485498096 

// CHECK: fldl 64(%rdx) 
// CHECK: encoding: [0xdd,0x42,0x40]        
fldl 64(%rdx) 

// CHECK: fldl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x44,0x82,0xc0]        
fldl -64(%rdx,%rax,4) 

// CHECK: fldl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x44,0x82,0x40]        
fldl 64(%rdx,%rax,4) 

// CHECK: fldl 64(%rdx,%rax) 
// CHECK: encoding: [0xdd,0x44,0x02,0x40]        
fldl 64(%rdx,%rax) 

// CHECK: fldlg2 
// CHECK: encoding: [0xd9,0xec]         
fldlg2 

// CHECK: fldln2 
// CHECK: encoding: [0xd9,0xed]         
fldln2 

// CHECK: fldl (%rdx) 
// CHECK: encoding: [0xdd,0x02]        
fldl (%rdx) 

// CHECK: fldpi 
// CHECK: encoding: [0xd9,0xeb]         
fldpi 

// CHECK: flds 485498096 
// CHECK: encoding: [0xd9,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
flds 485498096 

// CHECK: flds 64(%rdx) 
// CHECK: encoding: [0xd9,0x42,0x40]        
flds 64(%rdx) 

// CHECK: flds -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x44,0x82,0xc0]        
flds -64(%rdx,%rax,4) 

// CHECK: flds 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x44,0x82,0x40]        
flds 64(%rdx,%rax,4) 

// CHECK: flds 64(%rdx,%rax) 
// CHECK: encoding: [0xd9,0x44,0x02,0x40]        
flds 64(%rdx,%rax) 

// CHECK: flds (%rdx) 
// CHECK: encoding: [0xd9,0x02]        
flds (%rdx) 

// CHECK: fld %st(4) 
// CHECK: encoding: [0xd9,0xc4]        
fld %st(4) 

// CHECK: fldt 485498096 
// CHECK: encoding: [0xdb,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
fldt 485498096 

// CHECK: fldt 64(%rdx) 
// CHECK: encoding: [0xdb,0x6a,0x40]        
fldt 64(%rdx) 

// CHECK: fldt -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x6c,0x82,0xc0]        
fldt -64(%rdx,%rax,4) 

// CHECK: fldt 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x6c,0x82,0x40]        
fldt 64(%rdx,%rax,4) 

// CHECK: fldt 64(%rdx,%rax) 
// CHECK: encoding: [0xdb,0x6c,0x02,0x40]        
fldt 64(%rdx,%rax) 

// CHECK: fldt (%rdx) 
// CHECK: encoding: [0xdb,0x2a]        
fldt (%rdx) 

// CHECK: fldz 
// CHECK: encoding: [0xd9,0xee]         
fldz 

// CHECK: fmull 485498096 
// CHECK: encoding: [0xdc,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
fmull 485498096 

// CHECK: fmull 64(%rdx) 
// CHECK: encoding: [0xdc,0x4a,0x40]        
fmull 64(%rdx) 

// CHECK: fmull -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x4c,0x82,0xc0]        
fmull -64(%rdx,%rax,4) 

// CHECK: fmull 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x4c,0x82,0x40]        
fmull 64(%rdx,%rax,4) 

// CHECK: fmull 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x4c,0x02,0x40]        
fmull 64(%rdx,%rax) 

// CHECK: fmull (%rdx) 
// CHECK: encoding: [0xdc,0x0a]        
fmull (%rdx) 

// CHECK: fmulp %st(4) 
// CHECK: encoding: [0xde,0xcc]        
fmulp %st(4) 

// CHECK: fmuls 485498096 
// CHECK: encoding: [0xd8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
fmuls 485498096 

// CHECK: fmuls 64(%rdx) 
// CHECK: encoding: [0xd8,0x4a,0x40]        
fmuls 64(%rdx) 

// CHECK: fmuls -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x4c,0x82,0xc0]        
fmuls -64(%rdx,%rax,4) 

// CHECK: fmuls 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x4c,0x82,0x40]        
fmuls 64(%rdx,%rax,4) 

// CHECK: fmuls 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x4c,0x02,0x40]        
fmuls 64(%rdx,%rax) 

// CHECK: fmuls (%rdx) 
// CHECK: encoding: [0xd8,0x0a]        
fmuls (%rdx) 

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

// CHECK: fnsave 485498096 
// CHECK: encoding: [0xdd,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
fnsave 485498096 

// CHECK: fnsave 64(%rdx) 
// CHECK: encoding: [0xdd,0x72,0x40]        
fnsave 64(%rdx) 

// CHECK: fnsave -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x74,0x82,0xc0]        
fnsave -64(%rdx,%rax,4) 

// CHECK: fnsave 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x74,0x82,0x40]        
fnsave 64(%rdx,%rax,4) 

// CHECK: fnsave 64(%rdx,%rax) 
// CHECK: encoding: [0xdd,0x74,0x02,0x40]        
fnsave 64(%rdx,%rax) 

// CHECK: fnsave (%rdx) 
// CHECK: encoding: [0xdd,0x32]        
fnsave (%rdx) 

// CHECK: fnstcw 485498096 
// CHECK: encoding: [0xd9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fnstcw 485498096 

// CHECK: fnstcw 64(%rdx) 
// CHECK: encoding: [0xd9,0x7a,0x40]        
fnstcw 64(%rdx) 

// CHECK: fnstcw -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x7c,0x82,0xc0]        
fnstcw -64(%rdx,%rax,4) 

// CHECK: fnstcw 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x7c,0x82,0x40]        
fnstcw 64(%rdx,%rax,4) 

// CHECK: fnstcw 64(%rdx,%rax) 
// CHECK: encoding: [0xd9,0x7c,0x02,0x40]        
fnstcw 64(%rdx,%rax) 

// CHECK: fnstcw (%rdx) 
// CHECK: encoding: [0xd9,0x3a]        
fnstcw (%rdx) 

// CHECK: fnstenv 485498096 
// CHECK: encoding: [0xd9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
fnstenv 485498096 

// CHECK: fnstenv 64(%rdx) 
// CHECK: encoding: [0xd9,0x72,0x40]        
fnstenv 64(%rdx) 

// CHECK: fnstenv -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x74,0x82,0xc0]        
fnstenv -64(%rdx,%rax,4) 

// CHECK: fnstenv 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x74,0x82,0x40]        
fnstenv 64(%rdx,%rax,4) 

// CHECK: fnstenv 64(%rdx,%rax) 
// CHECK: encoding: [0xd9,0x74,0x02,0x40]        
fnstenv 64(%rdx,%rax) 

// CHECK: fnstenv (%rdx) 
// CHECK: encoding: [0xd9,0x32]        
fnstenv (%rdx) 

// CHECK: fnstsw 485498096 
// CHECK: encoding: [0xdd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fnstsw 485498096 

// CHECK: fnstsw 64(%rdx) 
// CHECK: encoding: [0xdd,0x7a,0x40]        
fnstsw 64(%rdx) 

// CHECK: fnstsw -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x7c,0x82,0xc0]        
fnstsw -64(%rdx,%rax,4) 

// CHECK: fnstsw 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x7c,0x82,0x40]        
fnstsw 64(%rdx,%rax,4) 

// CHECK: fnstsw 64(%rdx,%rax) 
// CHECK: encoding: [0xdd,0x7c,0x02,0x40]        
fnstsw 64(%rdx,%rax) 

// CHECK: fnstsw %ax 
// CHECK: encoding: [0xdf,0xe0]        
fnstsw %ax 

// CHECK: fnstsw (%rdx) 
// CHECK: encoding: [0xdd,0x3a]        
fnstsw (%rdx) 

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

// CHECK: frstor 485498096 
// CHECK: encoding: [0xdd,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
frstor 485498096 

// CHECK: frstor 64(%rdx) 
// CHECK: encoding: [0xdd,0x62,0x40]        
frstor 64(%rdx) 

// CHECK: frstor -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x64,0x82,0xc0]        
frstor -64(%rdx,%rax,4) 

// CHECK: frstor 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x64,0x82,0x40]        
frstor 64(%rdx,%rax,4) 

// CHECK: frstor 64(%rdx,%rax) 
// CHECK: encoding: [0xdd,0x64,0x02,0x40]        
frstor 64(%rdx,%rax) 

// CHECK: frstor (%rdx) 
// CHECK: encoding: [0xdd,0x22]        
frstor (%rdx) 

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

// CHECK: fstl 485498096 
// CHECK: encoding: [0xdd,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
fstl 485498096 

// CHECK: fstl 64(%rdx) 
// CHECK: encoding: [0xdd,0x52,0x40]        
fstl 64(%rdx) 

// CHECK: fstl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x54,0x82,0xc0]        
fstl -64(%rdx,%rax,4) 

// CHECK: fstl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x54,0x82,0x40]        
fstl 64(%rdx,%rax,4) 

// CHECK: fstl 64(%rdx,%rax) 
// CHECK: encoding: [0xdd,0x54,0x02,0x40]        
fstl 64(%rdx,%rax) 

// CHECK: fstl (%rdx) 
// CHECK: encoding: [0xdd,0x12]        
fstl (%rdx) 

// CHECK: fstpl 485498096 
// CHECK: encoding: [0xdd,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
fstpl 485498096 

// CHECK: fstpl 64(%rdx) 
// CHECK: encoding: [0xdd,0x5a,0x40]        
fstpl 64(%rdx) 

// CHECK: fstpl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x5c,0x82,0xc0]        
fstpl -64(%rdx,%rax,4) 

// CHECK: fstpl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdd,0x5c,0x82,0x40]        
fstpl 64(%rdx,%rax,4) 

// CHECK: fstpl 64(%rdx,%rax) 
// CHECK: encoding: [0xdd,0x5c,0x02,0x40]        
fstpl 64(%rdx,%rax) 

// CHECK: fstpl (%rdx) 
// CHECK: encoding: [0xdd,0x1a]        
fstpl (%rdx) 

// CHECK: fstps 485498096 
// CHECK: encoding: [0xd9,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
fstps 485498096 

// CHECK: fstps 64(%rdx) 
// CHECK: encoding: [0xd9,0x5a,0x40]        
fstps 64(%rdx) 

// CHECK: fstps -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x5c,0x82,0xc0]        
fstps -64(%rdx,%rax,4) 

// CHECK: fstps 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x5c,0x82,0x40]        
fstps 64(%rdx,%rax,4) 

// CHECK: fstps 64(%rdx,%rax) 
// CHECK: encoding: [0xd9,0x5c,0x02,0x40]        
fstps 64(%rdx,%rax) 

// CHECK: fstps (%rdx) 
// CHECK: encoding: [0xd9,0x1a]        
fstps (%rdx) 

// CHECK: fstp %st(4) 
// CHECK: encoding: [0xdd,0xdc]        
fstp %st(4) 

// CHECK: fstpt 485498096 
// CHECK: encoding: [0xdb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
fstpt 485498096 

// CHECK: fstpt 64(%rdx) 
// CHECK: encoding: [0xdb,0x7a,0x40]        
fstpt 64(%rdx) 

// CHECK: fstpt -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x7c,0x82,0xc0]        
fstpt -64(%rdx,%rax,4) 

// CHECK: fstpt 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdb,0x7c,0x82,0x40]        
fstpt 64(%rdx,%rax,4) 

// CHECK: fstpt 64(%rdx,%rax) 
// CHECK: encoding: [0xdb,0x7c,0x02,0x40]        
fstpt 64(%rdx,%rax) 

// CHECK: fstpt (%rdx) 
// CHECK: encoding: [0xdb,0x3a]        
fstpt (%rdx) 

// CHECK: fsts 485498096 
// CHECK: encoding: [0xd9,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
fsts 485498096 

// CHECK: fsts 64(%rdx) 
// CHECK: encoding: [0xd9,0x52,0x40]        
fsts 64(%rdx) 

// CHECK: fsts -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x54,0x82,0xc0]        
fsts -64(%rdx,%rax,4) 

// CHECK: fsts 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd9,0x54,0x82,0x40]        
fsts 64(%rdx,%rax,4) 

// CHECK: fsts 64(%rdx,%rax) 
// CHECK: encoding: [0xd9,0x54,0x02,0x40]        
fsts 64(%rdx,%rax) 

// CHECK: fsts (%rdx) 
// CHECK: encoding: [0xd9,0x12]        
fsts (%rdx) 

// CHECK: fst %st(4) 
// CHECK: encoding: [0xdd,0xd4]        
fst %st(4) 

// CHECK: fsubl 485498096 
// CHECK: encoding: [0xdc,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
fsubl 485498096 

// CHECK: fsubl 64(%rdx) 
// CHECK: encoding: [0xdc,0x62,0x40]        
fsubl 64(%rdx) 

// CHECK: fsubl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x64,0x82,0xc0]        
fsubl -64(%rdx,%rax,4) 

// CHECK: fsubl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x64,0x82,0x40]        
fsubl 64(%rdx,%rax,4) 

// CHECK: fsubl 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x64,0x02,0x40]        
fsubl 64(%rdx,%rax) 

// CHECK: fsubl (%rdx) 
// CHECK: encoding: [0xdc,0x22]        
fsubl (%rdx) 

// CHECK: fsubp %st(4) 
// CHECK: encoding: [0xde,0xe4]        
fsubp %st(4) 

// CHECK: fsubrl 485498096 
// CHECK: encoding: [0xdc,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
fsubrl 485498096 

// CHECK: fsubrl 64(%rdx) 
// CHECK: encoding: [0xdc,0x6a,0x40]        
fsubrl 64(%rdx) 

// CHECK: fsubrl -64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x6c,0x82,0xc0]        
fsubrl -64(%rdx,%rax,4) 

// CHECK: fsubrl 64(%rdx,%rax,4) 
// CHECK: encoding: [0xdc,0x6c,0x82,0x40]        
fsubrl 64(%rdx,%rax,4) 

// CHECK: fsubrl 64(%rdx,%rax) 
// CHECK: encoding: [0xdc,0x6c,0x02,0x40]        
fsubrl 64(%rdx,%rax) 

// CHECK: fsubrl (%rdx) 
// CHECK: encoding: [0xdc,0x2a]        
fsubrl (%rdx) 

// CHECK: fsubrp %st(4) 
// CHECK: encoding: [0xde,0xec]        
fsubrp %st(4) 

// CHECK: fsubrs 485498096 
// CHECK: encoding: [0xd8,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]        
fsubrs 485498096 

// CHECK: fsubrs 64(%rdx) 
// CHECK: encoding: [0xd8,0x6a,0x40]        
fsubrs 64(%rdx) 

// CHECK: fsubrs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x6c,0x82,0xc0]        
fsubrs -64(%rdx,%rax,4) 

// CHECK: fsubrs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x6c,0x82,0x40]        
fsubrs 64(%rdx,%rax,4) 

// CHECK: fsubrs 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x6c,0x02,0x40]        
fsubrs 64(%rdx,%rax) 

// CHECK: fsubrs (%rdx) 
// CHECK: encoding: [0xd8,0x2a]        
fsubrs (%rdx) 

// CHECK: fsubr %st(0), %st(4) 
// CHECK: encoding: [0xdc,0xec]       
fsubr %st(0), %st(4) 

// CHECK: fsubr %st(4) 
// CHECK: encoding: [0xd8,0xec]        
fsubr %st(4) 

// CHECK: fsubs 485498096 
// CHECK: encoding: [0xd8,0x24,0x25,0xf0,0x1c,0xf0,0x1c]        
fsubs 485498096 

// CHECK: fsubs 64(%rdx) 
// CHECK: encoding: [0xd8,0x62,0x40]        
fsubs 64(%rdx) 

// CHECK: fsubs -64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x64,0x82,0xc0]        
fsubs -64(%rdx,%rax,4) 

// CHECK: fsubs 64(%rdx,%rax,4) 
// CHECK: encoding: [0xd8,0x64,0x82,0x40]        
fsubs 64(%rdx,%rax,4) 

// CHECK: fsubs 64(%rdx,%rax) 
// CHECK: encoding: [0xd8,0x64,0x02,0x40]        
fsubs 64(%rdx,%rax) 

// CHECK: fsubs (%rdx) 
// CHECK: encoding: [0xd8,0x22]        
fsubs (%rdx) 

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

