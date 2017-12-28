// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: prefetch 485498096 
// CHECK: encoding: [0x0f,0x0d,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
prefetch 485498096 

// CHECK: prefetch 64(%rdx) 
// CHECK: encoding: [0x0f,0x0d,0x42,0x40]        
prefetch 64(%rdx) 

// CHECK: prefetch 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x0d,0x44,0x82,0x40]        
prefetch 64(%rdx,%rax,4) 

// CHECK: prefetch -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x0d,0x44,0x82,0xc0]        
prefetch -64(%rdx,%rax,4) 

// CHECK: prefetch 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x0d,0x44,0x02,0x40]        
prefetch 64(%rdx,%rax) 

// CHECK: prefetchnta 485498096 
// CHECK: encoding: [0x0f,0x18,0x04,0x25,0xf0,0x1c,0xf0,0x1c]        
prefetchnta 485498096 

// CHECK: prefetchnta 64(%rdx) 
// CHECK: encoding: [0x0f,0x18,0x42,0x40]        
prefetchnta 64(%rdx) 

// CHECK: prefetchnta 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x44,0x82,0x40]        
prefetchnta 64(%rdx,%rax,4) 

// CHECK: prefetchnta -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x44,0x82,0xc0]        
prefetchnta -64(%rdx,%rax,4) 

// CHECK: prefetchnta 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x18,0x44,0x02,0x40]        
prefetchnta 64(%rdx,%rax) 

// CHECK: prefetchnta (%rdx) 
// CHECK: encoding: [0x0f,0x18,0x02]        
prefetchnta (%rdx) 

// CHECK: prefetch (%rdx) 
// CHECK: encoding: [0x0f,0x0d,0x02]        
prefetch (%rdx) 

// CHECK: prefetcht0 485498096 
// CHECK: encoding: [0x0f,0x18,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
prefetcht0 485498096 

// CHECK: prefetcht0 64(%rdx) 
// CHECK: encoding: [0x0f,0x18,0x4a,0x40]        
prefetcht0 64(%rdx) 

// CHECK: prefetcht0 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x4c,0x82,0x40]        
prefetcht0 64(%rdx,%rax,4) 

// CHECK: prefetcht0 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x4c,0x82,0xc0]        
prefetcht0 -64(%rdx,%rax,4) 

// CHECK: prefetcht0 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x18,0x4c,0x02,0x40]        
prefetcht0 64(%rdx,%rax) 

// CHECK: prefetcht0 (%rdx) 
// CHECK: encoding: [0x0f,0x18,0x0a]        
prefetcht0 (%rdx) 

// CHECK: prefetcht1 485498096 
// CHECK: encoding: [0x0f,0x18,0x14,0x25,0xf0,0x1c,0xf0,0x1c]        
prefetcht1 485498096 

// CHECK: prefetcht1 64(%rdx) 
// CHECK: encoding: [0x0f,0x18,0x52,0x40]        
prefetcht1 64(%rdx) 

// CHECK: prefetcht1 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x54,0x82,0x40]        
prefetcht1 64(%rdx,%rax,4) 

// CHECK: prefetcht1 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x54,0x82,0xc0]        
prefetcht1 -64(%rdx,%rax,4) 

// CHECK: prefetcht1 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x18,0x54,0x02,0x40]        
prefetcht1 64(%rdx,%rax) 

// CHECK: prefetcht1 (%rdx) 
// CHECK: encoding: [0x0f,0x18,0x12]        
prefetcht1 (%rdx) 

// CHECK: prefetcht2 485498096 
// CHECK: encoding: [0x0f,0x18,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]        
prefetcht2 485498096 

// CHECK: prefetcht2 64(%rdx) 
// CHECK: encoding: [0x0f,0x18,0x5a,0x40]        
prefetcht2 64(%rdx) 

// CHECK: prefetcht2 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x5c,0x82,0x40]        
prefetcht2 64(%rdx,%rax,4) 

// CHECK: prefetcht2 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x18,0x5c,0x82,0xc0]        
prefetcht2 -64(%rdx,%rax,4) 

// CHECK: prefetcht2 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x18,0x5c,0x02,0x40]        
prefetcht2 64(%rdx,%rax) 

// CHECK: prefetcht2 (%rdx) 
// CHECK: encoding: [0x0f,0x18,0x1a]        
prefetcht2 (%rdx) 

// CHECK: prefetchw 485498096 
// CHECK: encoding: [0x0f,0x0d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]        
prefetchw 485498096 

// CHECK: prefetchw 64(%rdx) 
// CHECK: encoding: [0x0f,0x0d,0x4a,0x40]        
prefetchw 64(%rdx) 

// CHECK: prefetchw 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x0d,0x4c,0x82,0x40]        
prefetchw 64(%rdx,%rax,4) 

// CHECK: prefetchw -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x0d,0x4c,0x82,0xc0]        
prefetchw -64(%rdx,%rax,4) 

// CHECK: prefetchw 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x0d,0x4c,0x02,0x40]        
prefetchw 64(%rdx,%rax) 

// CHECK: prefetchw (%rdx) 
// CHECK: encoding: [0x0f,0x0d,0x0a]        
prefetchw (%rdx) 

// CHECK: prefetchwt1 485498096 
// CHECK: encoding:  [0x0f,0x0d,0x14,0x25,0xf0,0x1c,0xf0,0x1c]
prefetchwt1 485498096 

// CHECK: prefetchwt1 64(%rdx) 
// CHECK: encoding: [0x0f,0x0d,0x52,0x40]
prefetchwt1 64(%rdx) 

// CHECK: prefetchwt1 64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x0d,0x54,0x82,0x40]
prefetchwt1 64(%rdx,%rax,4) 

// CHECK: prefetchwt1 -64(%rdx,%rax,4) 
// CHECK: encoding: [0x0f,0x0d,0x54,0x82,0xc0]
prefetchwt1 -64(%rdx,%rax,4) 

// CHECK: prefetchwt1 64(%rdx,%rax) 
// CHECK: encoding: [0x0f,0x0d,0x54,0x02,0x40]        
prefetchwt1 64(%rdx,%rax) 

// CHECK: prefetchwt1 (%rdx) 
// CHECK: encoding: [0x0f,0x0d,0x12]        
prefetchwt1 (%rdx) 

