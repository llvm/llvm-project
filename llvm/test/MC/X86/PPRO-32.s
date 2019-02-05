// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: cmovael %eax, %eax 
// CHECK: encoding: [0x0f,0x43,0xc0]        
cmovael %eax, %eax 

// CHECK: cmoval %eax, %eax 
// CHECK: encoding: [0x0f,0x47,0xc0]        
cmoval %eax, %eax 

// CHECK: cmovbel %eax, %eax 
// CHECK: encoding: [0x0f,0x46,0xc0]        
cmovbel %eax, %eax 

// CHECK: cmovbl %eax, %eax 
// CHECK: encoding: [0x0f,0x42,0xc0]        
cmovbl %eax, %eax 

// CHECK: cmovel %eax, %eax 
// CHECK: encoding: [0x0f,0x44,0xc0]        
cmovel %eax, %eax 

// CHECK: cmovgel %eax, %eax 
// CHECK: encoding: [0x0f,0x4d,0xc0]        
cmovgel %eax, %eax 

// CHECK: cmovgl %eax, %eax 
// CHECK: encoding: [0x0f,0x4f,0xc0]        
cmovgl %eax, %eax 

// CHECK: cmovlel %eax, %eax 
// CHECK: encoding: [0x0f,0x4e,0xc0]        
cmovlel %eax, %eax 

// CHECK: cmovll %eax, %eax 
// CHECK: encoding: [0x0f,0x4c,0xc0]        
cmovll %eax, %eax 

// CHECK: cmovnel %eax, %eax 
// CHECK: encoding: [0x0f,0x45,0xc0]        
cmovnel %eax, %eax 

// CHECK: cmovnol %eax, %eax 
// CHECK: encoding: [0x0f,0x41,0xc0]        
cmovnol %eax, %eax 

// CHECK: cmovnpl %eax, %eax 
// CHECK: encoding: [0x0f,0x4b,0xc0]        
cmovnpl %eax, %eax 

// CHECK: cmovnsl %eax, %eax 
// CHECK: encoding: [0x0f,0x49,0xc0]        
cmovnsl %eax, %eax 

// CHECK: cmovol %eax, %eax 
// CHECK: encoding: [0x0f,0x40,0xc0]        
cmovol %eax, %eax 

// CHECK: cmovpl %eax, %eax 
// CHECK: encoding: [0x0f,0x4a,0xc0]        
cmovpl %eax, %eax 

// CHECK: cmovsl %eax, %eax 
// CHECK: encoding: [0x0f,0x48,0xc0]        
cmovsl %eax, %eax 

// CHECK: fcmovbe %st(4), %st 
// CHECK: encoding: [0xda,0xd4]        
fcmovbe %st(4), %st 

// CHECK: fcmovb %st(4), %st 
// CHECK: encoding: [0xda,0xc4]        
fcmovb %st(4), %st 

// CHECK: fcmove %st(4), %st 
// CHECK: encoding: [0xda,0xcc]        
fcmove %st(4), %st 

// CHECK: fcmovnbe %st(4), %st 
// CHECK: encoding: [0xdb,0xd4]        
fcmovnbe %st(4), %st 

// CHECK: fcmovnb %st(4), %st 
// CHECK: encoding: [0xdb,0xc4]        
fcmovnb %st(4), %st 

// CHECK: fcmovne %st(4), %st 
// CHECK: encoding: [0xdb,0xcc]        
fcmovne %st(4), %st 

// CHECK: fcmovnu %st(4), %st 
// CHECK: encoding: [0xdb,0xdc]        
fcmovnu %st(4), %st 

// CHECK: fcmovu %st(4), %st 
// CHECK: encoding: [0xda,0xdc]        
fcmovu %st(4), %st 

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

// CHECK: ud2 
// CHECK: encoding: [0x0f,0x0b]          
ud2 

