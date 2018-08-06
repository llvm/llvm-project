// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: cmovael %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x43,0xed]        
cmovael %r13d, %r13d 

// CHECK: cmoval %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x47,0xed]        
cmoval %r13d, %r13d 

// CHECK: cmovbel %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x46,0xed]        
cmovbel %r13d, %r13d 

// CHECK: cmovbl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x42,0xed]        
cmovbl %r13d, %r13d 

// CHECK: cmovel %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x44,0xed]        
cmovel %r13d, %r13d 

// CHECK: cmovgel %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x4d,0xed]        
cmovgel %r13d, %r13d 

// CHECK: cmovgl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x4f,0xed]        
cmovgl %r13d, %r13d 

// CHECK: cmovlel %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x4e,0xed]        
cmovlel %r13d, %r13d 

// CHECK: cmovll %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x4c,0xed]        
cmovll %r13d, %r13d 

// CHECK: cmovnel %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x45,0xed]        
cmovnel %r13d, %r13d 

// CHECK: cmovnol %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x41,0xed]        
cmovnol %r13d, %r13d 

// CHECK: cmovnpl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x4b,0xed]        
cmovnpl %r13d, %r13d 

// CHECK: cmovnsl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x49,0xed]        
cmovnsl %r13d, %r13d 

// CHECK: cmovol %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x40,0xed]        
cmovol %r13d, %r13d 

// CHECK: cmovpl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x4a,0xed]        
cmovpl %r13d, %r13d 

// CHECK: cmovsl %r13d, %r13d 
// CHECK: encoding: [0x45,0x0f,0x48,0xed]        
cmovsl %r13d, %r13d 

// CHECK: fcmovbe %st(4), %st(0) 
// CHECK: encoding: [0xda,0xd4]        
fcmovbe %st(4), %st(0) 

// CHECK: fcmovb %st(4), %st(0) 
// CHECK: encoding: [0xda,0xc4]        
fcmovb %st(4), %st(0) 

// CHECK: fcmove %st(4), %st(0) 
// CHECK: encoding: [0xda,0xcc]        
fcmove %st(4), %st(0) 

// CHECK: fcmovnbe %st(4), %st(0) 
// CHECK: encoding: [0xdb,0xd4]        
fcmovnbe %st(4), %st(0) 

// CHECK: fcmovnb %st(4), %st(0) 
// CHECK: encoding: [0xdb,0xc4]        
fcmovnb %st(4), %st(0) 

// CHECK: fcmovne %st(4), %st(0) 
// CHECK: encoding: [0xdb,0xcc]        
fcmovne %st(4), %st(0) 

// CHECK: fcmovnu %st(4), %st(0) 
// CHECK: encoding: [0xdb,0xdc]        
fcmovnu %st(4), %st(0) 

// CHECK: fcmovu %st(4), %st(0) 
// CHECK: encoding: [0xda,0xdc]        
fcmovu %st(4), %st(0) 

// CHECK: fcomi %st(4) 
// CHECK: encoding: [0xdb,0xf4]         
fcomi %st(4) 

// CHECK: fcompi %st(4) 
// CHECK: encoding: [0xdf,0xf4]         
fcompi %st(4) 

// CHECK: fucomi %st(4) 
// CHECK: encoding: [0xdb,0xec]         
fucomi %st(4) 

// CHECK: fucompi %st(4) 
// CHECK: encoding: [0xdf,0xec]         
fucompi %st(4) 

// CHECK: sysenter 
// CHECK: encoding: [0x0f,0x34]          
sysenter 

// CHECK: sysexitl 
// CHECK: encoding: [0x0f,0x35]          
sysexitl 

// CHECK: sysexitq 
// CHECK: encoding: [0x48,0x0f,0x35]          
sysexitq 

// CHECK: ud2 
// CHECK: encoding: [0x0f,0x0b]          
ud2 

