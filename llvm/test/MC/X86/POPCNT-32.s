// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: popcntl %eax, %eax 
// CHECK: encoding: [0xf3,0x0f,0xb8,0xc0]        
popcntl %eax, %eax 

// CHECK: popcntl 4096(%eax), %eax
// CHECK: encoding: [0xf3,0x0f,0xb8,0x80,0x00,0x10,0x00,0x00]
popcntl 4096(%eax), %eax

// CHECK: popcntl 64(%edx,%eax), %ecx     
// CHECK: encoding: [0xf3,0x0f,0xb8,0x4c,0x02,0x40]
popcntl 64(%edx,%eax), %ecx     

// CHECK: popcntl 64(%edx,%eax,4), %ecx   
// CHECK: encoding: [0xf3,0x0f,0xb8,0x4c,0x82,0x40]
popcntl 64(%edx,%eax,4), %ecx   

// CHECK: popcntw %ax, %ax 
// CHECK: encoding: [0x66,0xf3,0x0f,0xb8,0xc0]        
popcntw %ax, %ax 

// CHECK: popcntw 4096(%eax), %ax
// CHECK: encoding: [0x66,0xf3,0x0f,0xb8,0x80,0x00,0x10,0x00,0x00]
popcntw 4096(%eax), %ax

// CHECK: popcntw 64(%edx,%eax), %cx      
// CHECK: encoding: [0x66,0xf3,0x0f,0xb8,0x4c,0x02,0x40]
popcntw 64(%edx,%eax), %cx      

// CHECK: popcntw 64(%edx,%eax,4), %cx    
// CHECK: encoding: [0x66,0xf3,0x0f,0xb8,0x4c,0x82,0x40]
popcntw 64(%edx,%eax,4), %cx    

