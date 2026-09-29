// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: sha1msg1 -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
sha1msg1 -485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1msg1 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
sha1msg1 485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1msg1 485498096(%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc9,0x8a,0xf0,0x1c,0xf0,0x1c]        
sha1msg1 485498096(%edx), %xmm1 

// CHECK: sha1msg1 485498096, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc9,0x0d,0xf0,0x1c,0xf0,0x1c]        
sha1msg1 485498096, %xmm1 

// CHECK: sha1msg1 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc9,0x4c,0x02,0x40]        
sha1msg1 64(%edx,%eax), %xmm1 

// CHECK: sha1msg1 (%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc9,0x0a]        
sha1msg1 (%edx), %xmm1 

// CHECK: sha1msg1 %xmm1, %xmm1 
// CHECK: encoding: [0x0f,        
sha1msg1 %xmm1, %xmm1 

// CHECK: sha1msg1 %xmm1, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc9,0xc9]        
sha1msg1 %xmm1, %xmm1 

// CHECK: sha1msg2 -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xca,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
sha1msg2 -485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1msg2 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xca,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
sha1msg2 485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1msg2 485498096(%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xca,0x8a,0xf0,0x1c,0xf0,0x1c]        
sha1msg2 485498096(%edx), %xmm1 

// CHECK: sha1msg2 485498096, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xca,0x0d,0xf0,0x1c,0xf0,0x1c]        
sha1msg2 485498096, %xmm1 

// CHECK: sha1msg2 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xca,0x4c,0x02,0x40]        
sha1msg2 64(%edx,%eax), %xmm1 

// CHECK: sha1msg2 (%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xca,0x0a]        
sha1msg2 (%edx), %xmm1 

// CHECK: sha1msg2 %xmm1, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xca,0xc9]        
sha1msg2 %xmm1, %xmm1 

// CHECK: sha1nexte -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc8,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
sha1nexte -485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1nexte 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc8,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
sha1nexte 485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1nexte 485498096(%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc8,0x8a,0xf0,0x1c,0xf0,0x1c]        
sha1nexte 485498096(%edx), %xmm1 

// CHECK: sha1nexte 485498096, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc8,0x0d,0xf0,0x1c,0xf0,0x1c]        
sha1nexte 485498096, %xmm1 

// CHECK: sha1nexte 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc8,0x4c,0x02,0x40]        
sha1nexte 64(%edx,%eax), %xmm1 

// CHECK: sha1nexte (%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc8,0x0a]        
sha1nexte (%edx), %xmm1 

// CHECK: sha1nexte %xmm1, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xc8,0xc9]        
sha1nexte %xmm1, %xmm1 

// CHECK: sha1rnds4 $0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]       
sha1rnds4 $0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1rnds4 $0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]       
sha1rnds4 $0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: sha1rnds4 $0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]       
sha1rnds4 $0, 485498096(%edx), %xmm1 

// CHECK: sha1rnds4 $0, 485498096, %xmm1 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]       
sha1rnds4 $0, 485498096, %xmm1 

// CHECK: sha1rnds4 $0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x4c,0x02,0x40,0x00]       
sha1rnds4 $0, 64(%edx,%eax), %xmm1 

// CHECK: sha1rnds4 $0, (%edx), %xmm1 
// CHECK: encoding: [0x0f,0x3a,0xcc,0x0a,0x00]       
sha1rnds4 $0, (%edx), %xmm1 

// CHECK: sha1rnds4 $0, %xmm1, %xmm1 
// CHECK: encoding: [0x0f,0x3a,0xcc,0xc9,0x00]       
sha1rnds4 $0, %xmm1, %xmm1 

// CHECK: sha256msg1 -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcc,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
sha256msg1 -485498096(%edx,%eax,4), %xmm1 

// CHECK: sha256msg1 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcc,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
sha256msg1 485498096(%edx,%eax,4), %xmm1 

// CHECK: sha256msg1 485498096(%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcc,0x8a,0xf0,0x1c,0xf0,0x1c]        
sha256msg1 485498096(%edx), %xmm1 

// CHECK: sha256msg1 485498096, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcc,0x0d,0xf0,0x1c,0xf0,0x1c]        
sha256msg1 485498096, %xmm1 

// CHECK: sha256msg1 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcc,0x4c,0x02,0x40]        
sha256msg1 64(%edx,%eax), %xmm1 

// CHECK: sha256msg1 (%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcc,0x0a]        
sha256msg1 (%edx), %xmm1 

// CHECK: sha256msg1 %xmm1, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcc,0xc9]        
sha256msg1 %xmm1, %xmm1 

// CHECK: sha256msg2 -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]        
sha256msg2 -485498096(%edx,%eax,4), %xmm1 

// CHECK: sha256msg2 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]        
sha256msg2 485498096(%edx,%eax,4), %xmm1 

// CHECK: sha256msg2 485498096(%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcd,0x8a,0xf0,0x1c,0xf0,0x1c]        
sha256msg2 485498096(%edx), %xmm1 

// CHECK: sha256msg2 485498096, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcd,0x0d,0xf0,0x1c,0xf0,0x1c]        
sha256msg2 485498096, %xmm1 

// CHECK: sha256msg2 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcd,0x4c,0x02,0x40]        
sha256msg2 64(%edx,%eax), %xmm1 

// CHECK: sha256msg2 (%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcd,0x0a]        
sha256msg2 (%edx), %xmm1 

// CHECK: sha256msg2 %xmm1, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcd,0xc9]        
sha256msg2 %xmm1, %xmm1 

// CHECK: sha256rnds2 %xmm0, -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
sha256rnds2 %xmm0, -485498096(%edx,%eax,4), %xmm1 

// CHECK: sha256rnds2 %xmm0, 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
sha256rnds2 %xmm0, 485498096(%edx,%eax,4), %xmm1 

// CHECK: sha256rnds2 %xmm0, 485498096(%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcb,0x8a,0xf0,0x1c,0xf0,0x1c]       
sha256rnds2 %xmm0, 485498096(%edx), %xmm1 

// CHECK: sha256rnds2 %xmm0, 485498096, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcb,0x0d,0xf0,0x1c,0xf0,0x1c]       
sha256rnds2 %xmm0, 485498096, %xmm1 

// CHECK: sha256rnds2 %xmm0, 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcb,0x4c,0x02,0x40]       
sha256rnds2 %xmm0, 64(%edx,%eax), %xmm1 

// CHECK: sha256rnds2 %xmm0, (%edx), %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcb,0x0a]       
sha256rnds2 %xmm0, (%edx), %xmm1 

// CHECK: sha256rnds2 %xmm0, %xmm1, %xmm1 
// CHECK: encoding: [0x0f,0x38,0xcb,0xc9]       
sha256rnds2 %xmm0, %xmm1, %xmm1 

