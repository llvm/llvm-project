// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: sha1msg1 485498096, %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sha1msg1 485498096, %xmm15 

// CHECK: sha1msg1 485498096, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc9,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
sha1msg1 485498096, %xmm6 

// CHECK: sha1msg1 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc9,0x7c,0x82,0x40]        
sha1msg1 64(%rdx,%rax,4), %xmm15 

// CHECK: sha1msg1 -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc9,0x7c,0x82,0xc0]        
sha1msg1 -64(%rdx,%rax,4), %xmm15 

// CHECK: sha1msg1 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc9,0x74,0x82,0x40]        
sha1msg1 64(%rdx,%rax,4), %xmm6 

// CHECK: sha1msg1 -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc9,0x74,0x82,0xc0]        
sha1msg1 -64(%rdx,%rax,4), %xmm6 

// CHECK: sha1msg1 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc9,0x7c,0x02,0x40]        
sha1msg1 64(%rdx,%rax), %xmm15 

// CHECK: sha1msg1 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc9,0x74,0x02,0x40]        
sha1msg1 64(%rdx,%rax), %xmm6 

// CHECK: sha1msg1 64(%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc9,0x7a,0x40]        
sha1msg1 64(%rdx), %xmm15 

// CHECK: sha1msg1 64(%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc9,0x72,0x40]        
sha1msg1 64(%rdx), %xmm6 

// CHECK: sha1msg1 (%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc9,0x3a]        
sha1msg1 (%rdx), %xmm15 

// CHECK: sha1msg1 (%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc9,0x32]        
sha1msg1 (%rdx), %xmm6 

// CHECK: sha1msg1 %xmm15, %xmm15 
// CHECK: encoding: [0x45,0x0f,0x38,0xc9,0xff]        
sha1msg1 %xmm15, %xmm15 

// CHECK: sha1msg1 %xmm6, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc9,0xf6]        
sha1msg1 %xmm6, %xmm6 

// CHECK: sha1msg2 485498096, %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xca,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sha1msg2 485498096, %xmm15 

// CHECK: sha1msg2 485498096, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xca,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
sha1msg2 485498096, %xmm6 

// CHECK: sha1msg2 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xca,0x7c,0x82,0x40]        
sha1msg2 64(%rdx,%rax,4), %xmm15 

// CHECK: sha1msg2 -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xca,0x7c,0x82,0xc0]        
sha1msg2 -64(%rdx,%rax,4), %xmm15 

// CHECK: sha1msg2 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xca,0x74,0x82,0x40]        
sha1msg2 64(%rdx,%rax,4), %xmm6 

// CHECK: sha1msg2 -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xca,0x74,0x82,0xc0]        
sha1msg2 -64(%rdx,%rax,4), %xmm6 

// CHECK: sha1msg2 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xca,0x7c,0x02,0x40]        
sha1msg2 64(%rdx,%rax), %xmm15 

// CHECK: sha1msg2 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xca,0x74,0x02,0x40]        
sha1msg2 64(%rdx,%rax), %xmm6 

// CHECK: sha1msg2 64(%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xca,0x7a,0x40]        
sha1msg2 64(%rdx), %xmm15 

// CHECK: sha1msg2 64(%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xca,0x72,0x40]        
sha1msg2 64(%rdx), %xmm6 

// CHECK: sha1msg2 (%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xca,0x3a]        
sha1msg2 (%rdx), %xmm15 

// CHECK: sha1msg2 (%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xca,0x32]        
sha1msg2 (%rdx), %xmm6 

// CHECK: sha1msg2 %xmm15, %xmm15 
// CHECK: encoding: [0x45,0x0f,0x38,0xca,0xff]        
sha1msg2 %xmm15, %xmm15 

// CHECK: sha1msg2 %xmm6, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xca,0xf6]        
sha1msg2 %xmm6, %xmm6 

// CHECK: sha1nexte 485498096, %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sha1nexte 485498096, %xmm15 

// CHECK: sha1nexte 485498096, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc8,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
sha1nexte 485498096, %xmm6 

// CHECK: sha1nexte 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc8,0x7c,0x82,0x40]        
sha1nexte 64(%rdx,%rax,4), %xmm15 

// CHECK: sha1nexte -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc8,0x7c,0x82,0xc0]        
sha1nexte -64(%rdx,%rax,4), %xmm15 

// CHECK: sha1nexte 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc8,0x74,0x82,0x40]        
sha1nexte 64(%rdx,%rax,4), %xmm6 

// CHECK: sha1nexte -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc8,0x74,0x82,0xc0]        
sha1nexte -64(%rdx,%rax,4), %xmm6 

// CHECK: sha1nexte 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc8,0x7c,0x02,0x40]        
sha1nexte 64(%rdx,%rax), %xmm15 

// CHECK: sha1nexte 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc8,0x74,0x02,0x40]        
sha1nexte 64(%rdx,%rax), %xmm6 

// CHECK: sha1nexte 64(%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc8,0x7a,0x40]        
sha1nexte 64(%rdx), %xmm15 

// CHECK: sha1nexte 64(%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc8,0x72,0x40]        
sha1nexte 64(%rdx), %xmm6 

// CHECK: sha1nexte (%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xc8,0x3a]        
sha1nexte (%rdx), %xmm15 

// CHECK: sha1nexte (%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc8,0x32]        
sha1nexte (%rdx), %xmm6 

// CHECK: sha1nexte %xmm15, %xmm15 
// CHECK: encoding: [0x45,0x0f,0x38,0xc8,0xff]        
sha1nexte %xmm15, %xmm15 

// CHECK: sha1nexte %xmm6, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xc8,0xf6]        
sha1nexte %xmm6, %xmm6 

// CHECK: sha1rnds4 $0, 485498096, %xmm15 
// CHECK: encoding: [0x44,0x0f,0x3a,0xcc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]       
sha1rnds4 $0, 485498096, %xmm15 

// CHECK: sha1rnds4 $0, 485498096, %xmm6 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]       
sha1rnds4 $0, 485498096, %xmm6 

// CHECK: sha1rnds4 $0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x3a,0xcc,0x7c,0x82,0x40,0x00]       
sha1rnds4 $0, 64(%rdx,%rax,4), %xmm15 

// CHECK: sha1rnds4 $0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x3a,0xcc,0x7c,0x82,0xc0,0x00]       
sha1rnds4 $0, -64(%rdx,%rax,4), %xmm15 

// CHECK: sha1rnds4 $0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x74,0x82,0x40,0x00]       
sha1rnds4 $0, 64(%rdx,%rax,4), %xmm6 

// CHECK: sha1rnds4 $0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x74,0x82,0xc0,0x00]       
sha1rnds4 $0, -64(%rdx,%rax,4), %xmm6 

// CHECK: sha1rnds4 $0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x3a,0xcc,0x7c,0x02,0x40,0x00]       
sha1rnds4 $0, 64(%rdx,%rax), %xmm15 

// CHECK: sha1rnds4 $0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x74,0x02,0x40,0x00]       
sha1rnds4 $0, 64(%rdx,%rax), %xmm6 

// CHECK: sha1rnds4 $0, 64(%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x3a,0xcc,0x7a,0x40,0x00]       
sha1rnds4 $0, 64(%rdx), %xmm15 

// CHECK: sha1rnds4 $0, 64(%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x72,0x40,0x00]       
sha1rnds4 $0, 64(%rdx), %xmm6 

// CHECK: sha1rnds4 $0, (%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x3a,0xcc,0x3a,0x00]       
sha1rnds4 $0, (%rdx), %xmm15 

// CHECK: sha1rnds4 $0, (%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x32,0x00]       
sha1rnds4 $0, (%rdx), %xmm6 

// CHECK: sha1rnds4 $0, %xmm15, %xmm15 
// CHECK: encoding: [0x45,0x0f,0x3a,0xcc,0xff,0x00]       
sha1rnds4 $0, %xmm15, %xmm15 

// CHECK: sha1rnds4 $0, %xmm6, %xmm6 
// CHECK: encoding: [0x0f,0x3a,0xcc,0xf6,0x00]       
sha1rnds4 $0, %xmm6, %xmm6 

// CHECK: sha256msg1 485498096, %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sha256msg1 485498096, %xmm15 

// CHECK: sha256msg1 485498096, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcc,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
sha256msg1 485498096, %xmm6 

// CHECK: sha256msg1 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcc,0x7c,0x82,0x40]        
sha256msg1 64(%rdx,%rax,4), %xmm15 

// CHECK: sha256msg1 -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcc,0x7c,0x82,0xc0]        
sha256msg1 -64(%rdx,%rax,4), %xmm15 

// CHECK: sha256msg1 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcc,0x74,0x82,0x40]        
sha256msg1 64(%rdx,%rax,4), %xmm6 

// CHECK: sha256msg1 -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcc,0x74,0x82,0xc0]        
sha256msg1 -64(%rdx,%rax,4), %xmm6 

// CHECK: sha256msg1 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcc,0x7c,0x02,0x40]        
sha256msg1 64(%rdx,%rax), %xmm15 

// CHECK: sha256msg1 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcc,0x74,0x02,0x40]        
sha256msg1 64(%rdx,%rax), %xmm6 

// CHECK: sha256msg1 64(%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcc,0x7a,0x40]        
sha256msg1 64(%rdx), %xmm15 

// CHECK: sha256msg1 64(%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcc,0x72,0x40]        
sha256msg1 64(%rdx), %xmm6 

// CHECK: sha256msg1 (%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcc,0x3a]        
sha256msg1 (%rdx), %xmm15 

// CHECK: sha256msg1 (%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcc,0x32]        
sha256msg1 (%rdx), %xmm6 

// CHECK: sha256msg1 %xmm15, %xmm15 
// CHECK: encoding: [0x45,0x0f,0x38,0xcc,0xff]        
sha256msg1 %xmm15, %xmm15 

// CHECK: sha256msg1 %xmm6, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcc,0xf6]        
sha256msg1 %xmm6, %xmm6 

// CHECK: sha256msg2 485498096, %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]        
sha256msg2 485498096, %xmm15 

// CHECK: sha256msg2 485498096, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcd,0x34,0x25,0xf0,0x1c,0xf0,0x1c]        
sha256msg2 485498096, %xmm6 

// CHECK: sha256msg2 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcd,0x7c,0x82,0x40]        
sha256msg2 64(%rdx,%rax,4), %xmm15 

// CHECK: sha256msg2 -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcd,0x7c,0x82,0xc0]        
sha256msg2 -64(%rdx,%rax,4), %xmm15 

// CHECK: sha256msg2 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcd,0x74,0x82,0x40]        
sha256msg2 64(%rdx,%rax,4), %xmm6 

// CHECK: sha256msg2 -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcd,0x74,0x82,0xc0]        
sha256msg2 -64(%rdx,%rax,4), %xmm6 

// CHECK: sha256msg2 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcd,0x7c,0x02,0x40]        
sha256msg2 64(%rdx,%rax), %xmm15 

// CHECK: sha256msg2 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcd,0x74,0x02,0x40]        
sha256msg2 64(%rdx,%rax), %xmm6 

// CHECK: sha256msg2 64(%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcd,0x7a,0x40]        
sha256msg2 64(%rdx), %xmm15 

// CHECK: sha256msg2 64(%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcd,0x72,0x40]        
sha256msg2 64(%rdx), %xmm6 

// CHECK: sha256msg2 (%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcd,0x3a]        
sha256msg2 (%rdx), %xmm15 

// CHECK: sha256msg2 (%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcd,0x32]        
sha256msg2 (%rdx), %xmm6 

// CHECK: sha256msg2 %xmm15, %xmm15 
// CHECK: encoding: [0x45,0x0f,0x38,0xcd,0xff]        
sha256msg2 %xmm15, %xmm15 

// CHECK: sha256msg2 %xmm6, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcd,0xf6]        
sha256msg2 %xmm6, %xmm6 

// CHECK: sha256rnds2 %xmm0, 485498096, %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
sha256rnds2 %xmm0, 485498096, %xmm15 

// CHECK: sha256rnds2 %xmm0, 485498096, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcb,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
sha256rnds2 %xmm0, 485498096, %xmm6 

// CHECK: sha256rnds2 %xmm0, 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcb,0x7c,0x82,0x40]       
sha256rnds2 %xmm0, 64(%rdx,%rax,4), %xmm15 

// CHECK: sha256rnds2 %xmm0, -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcb,0x7c,0x82,0xc0]       
sha256rnds2 %xmm0, -64(%rdx,%rax,4), %xmm15 

// CHECK: sha256rnds2 %xmm0, 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcb,0x74,0x82,0x40]       
sha256rnds2 %xmm0, 64(%rdx,%rax,4), %xmm6 

// CHECK: sha256rnds2 %xmm0, -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcb,0x74,0x82,0xc0]       
sha256rnds2 %xmm0, -64(%rdx,%rax,4), %xmm6 

// CHECK: sha256rnds2 %xmm0, 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcb,0x7c,0x02,0x40]       
sha256rnds2 %xmm0, 64(%rdx,%rax), %xmm15 

// CHECK: sha256rnds2 %xmm0, 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcb,0x74,0x02,0x40]       
sha256rnds2 %xmm0, 64(%rdx,%rax), %xmm6 

// CHECK: sha256rnds2 %xmm0, 64(%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcb,0x7a,0x40]       
sha256rnds2 %xmm0, 64(%rdx), %xmm15 

// CHECK: sha256rnds2 %xmm0, 64(%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcb,0x72,0x40]       
sha256rnds2 %xmm0, 64(%rdx), %xmm6 

// CHECK: sha256rnds2 %xmm0, (%rdx), %xmm15 
// CHECK: encoding: [0x44,0x0f,0x38,0xcb,0x3a]       
sha256rnds2 %xmm0, (%rdx), %xmm15 

// CHECK: sha256rnds2 %xmm0, (%rdx), %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcb,0x32]       
sha256rnds2 %xmm0, (%rdx), %xmm6 

// CHECK: sha256rnds2 %xmm0, %xmm15, %xmm15 
// CHECK: encoding: [0x45,0x0f,0x38,0xcb,0xff]       
sha256rnds2 %xmm0, %xmm15, %xmm15 

// CHECK: sha256rnds2 %xmm0, %xmm6, %xmm6 
// CHECK: encoding: [0x0f,0x38,0xcb,0xf6]       
sha256rnds2 %xmm0, %xmm6, %xmm6 

