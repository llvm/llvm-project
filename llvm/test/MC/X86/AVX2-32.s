// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vbroadcasti128 -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vbroadcasti128 -485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcasti128 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vbroadcasti128 485498096(%edx,%eax,4), %ymm4 

// CHECK: vbroadcasti128 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0xa2,0xf0,0x1c,0xf0,0x1c]       
vbroadcasti128 485498096(%edx), %ymm4 

// CHECK: vbroadcasti128 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcasti128 485498096, %ymm4 

// CHECK: vbroadcasti128 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x64,0x02,0x40]       
vbroadcasti128 64(%edx,%eax), %ymm4 

// CHECK: vbroadcasti128 (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x22]       
vbroadcasti128 (%edx), %ymm4 

// CHECK: vbroadcastsd %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0xe1]       
vbroadcastsd %xmm1, %ymm4 

// CHECK: vbroadcastss %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0xc9]       
vbroadcastss %xmm1, %xmm1 

// CHECK: vbroadcastss %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0xe1]       
vbroadcastss %xmm1, %ymm4 

// CHECK: vextracti128 $0, %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vextracti128 $0, %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vextracti128 $0, %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vextracti128 $0, %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vextracti128 $0, %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vextracti128 $0, %ymm4, 485498096(%edx) 

// CHECK: vextracti128 $0, %ymm4, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextracti128 $0, %ymm4, 485498096 

// CHECK: vextracti128 $0, %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x64,0x02,0x40,0x00]      
vextracti128 $0, %ymm4, 64(%edx,%eax) 

// CHECK: vextracti128 $0, %ymm4, (%edx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x22,0x00]      
vextracti128 $0, %ymm4, (%edx) 

// CHECK: vextracti128 $0, %ymm4, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0xe1,0x00]      
vextracti128 $0, %ymm4, %xmm1 

// CHECK: vinserti128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x38,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vinserti128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vinserti128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x38,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vinserti128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vinserti128 $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x38,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vinserti128 $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vinserti128 $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x38,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinserti128 $0, 485498096, %ymm4, %ymm4 

// CHECK: vinserti128 $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x38,0x64,0x02,0x40,0x00]     
vinserti128 $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vinserti128 $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x38,0x22,0x00]     
vinserti128 $0, (%edx), %ymm4, %ymm4 

// CHECK: vinserti128 $0, %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x38,0xe1,0x00]     
vinserti128 $0, %xmm1, %ymm4, %ymm4 

// CHECK: vmovntdqa -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vmovntdqa -485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovntdqa 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096(%edx,%eax,4), %ymm4 

// CHECK: vmovntdqa 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0xa2,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096(%edx), %ymm4 

// CHECK: vmovntdqa 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096, %ymm4 

// CHECK: vmovntdqa 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x64,0x02,0x40]       
vmovntdqa 64(%edx,%eax), %ymm4 

// CHECK: vmovntdqa (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x22]       
vmovntdqa (%edx), %ymm4 

// CHECK: vmpsadbw $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x42,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vmpsadbw $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmpsadbw $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x42,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vmpsadbw $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x42,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vmpsadbw $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x42,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096, %ymm4, %ymm4 

// CHECK: vmpsadbw $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x42,0x64,0x02,0x40,0x00]     
vmpsadbw $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vmpsadbw $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x42,0x22,0x00]     
vmpsadbw $0, (%edx), %ymm4, %ymm4 

// CHECK: vmpsadbw $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x42,0xe4,0x00]     
vmpsadbw $0, %ymm4, %ymm4, %ymm4 

// CHECK: vpabsb -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpabsb -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpabsb 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpabsb 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096(%edx), %ymm4 

// CHECK: vpabsb 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096, %ymm4 

// CHECK: vpabsb 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x64,0x02,0x40]       
vpabsb 64(%edx,%eax), %ymm4 

// CHECK: vpabsb (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x22]       
vpabsb (%edx), %ymm4 

// CHECK: vpabsb %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0xe4]       
vpabsb %ymm4, %ymm4 

// CHECK: vpabsd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpabsd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpabsd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpabsd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096(%edx), %ymm4 

// CHECK: vpabsd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096, %ymm4 

// CHECK: vpabsd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x64,0x02,0x40]       
vpabsd 64(%edx,%eax), %ymm4 

// CHECK: vpabsd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x22]       
vpabsd (%edx), %ymm4 

// CHECK: vpabsd %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0xe4]       
vpabsd %ymm4, %ymm4 

// CHECK: vpabsw -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpabsw -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpabsw 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpabsw 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096(%edx), %ymm4 

// CHECK: vpabsw 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096, %ymm4 

// CHECK: vpabsw 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x64,0x02,0x40]       
vpabsw 64(%edx,%eax), %ymm4 

// CHECK: vpabsw (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x22]       
vpabsw (%edx), %ymm4 

// CHECK: vpabsw %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0xe4]       
vpabsw %ymm4, %ymm4 

// CHECK: vpackssdw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpackssdw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpackssdw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpackssdw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6b,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpackssdw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6b,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096, %ymm4, %ymm4 

// CHECK: vpackssdw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6b,0x64,0x02,0x40]      
vpackssdw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpackssdw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6b,0x22]      
vpackssdw (%edx), %ymm4, %ymm4 

// CHECK: vpackssdw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6b,0xe4]      
vpackssdw %ymm4, %ymm4, %ymm4 

// CHECK: vpacksswb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x63,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpacksswb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpacksswb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x63,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpacksswb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x63,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpacksswb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x63,0x25,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096, %ymm4, %ymm4 

// CHECK: vpacksswb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x63,0x64,0x02,0x40]      
vpacksswb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpacksswb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x63,0x22]      
vpacksswb (%edx), %ymm4, %ymm4 

// CHECK: vpacksswb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x63,0xe4]      
vpacksswb %ymm4, %ymm4, %ymm4 

// CHECK: vpackusdw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpackusdw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpackusdw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpackusdw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2b,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpackusdw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2b,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096, %ymm4, %ymm4 

// CHECK: vpackusdw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2b,0x64,0x02,0x40]      
vpackusdw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpackusdw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2b,0x22]      
vpackusdw (%edx), %ymm4, %ymm4 

// CHECK: vpackusdw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x2b,0xe4]      
vpackusdw %ymm4, %ymm4, %ymm4 

// CHECK: vpackuswb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x67,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpackuswb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpackuswb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x67,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpackuswb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x67,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpackuswb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x67,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096, %ymm4, %ymm4 

// CHECK: vpackuswb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x67,0x64,0x02,0x40]      
vpackuswb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpackuswb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x67,0x22]      
vpackuswb (%edx), %ymm4, %ymm4 

// CHECK: vpackuswb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x67,0xe4]      
vpackuswb %ymm4, %ymm4, %ymm4 

// CHECK: vpaddb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfc,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfc,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfc,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfc,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096, %ymm4, %ymm4 

// CHECK: vpaddb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfc,0x64,0x02,0x40]      
vpaddb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfc,0x22]      
vpaddb (%edx), %ymm4, %ymm4 

// CHECK: vpaddb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfc,0xe4]      
vpaddb %ymm4, %ymm4, %ymm4 

// CHECK: vpaddd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfe,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfe,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfe,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfe,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096, %ymm4, %ymm4 

// CHECK: vpaddd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfe,0x64,0x02,0x40]      
vpaddd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfe,0x22]      
vpaddd (%edx), %ymm4, %ymm4 

// CHECK: vpaddd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfe,0xe4]      
vpaddd %ymm4, %ymm4, %ymm4 

// CHECK: vpaddq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd4,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd4,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd4,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd4,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096, %ymm4, %ymm4 

// CHECK: vpaddq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd4,0x64,0x02,0x40]      
vpaddq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd4,0x22]      
vpaddq (%edx), %ymm4, %ymm4 

// CHECK: vpaddq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd4,0xe4]      
vpaddq %ymm4, %ymm4, %ymm4 

// CHECK: vpaddsb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xec,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddsb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddsb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xec,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddsb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xec,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddsb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xec,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096, %ymm4, %ymm4 

// CHECK: vpaddsb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xec,0x64,0x02,0x40]      
vpaddsb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddsb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xec,0x22]      
vpaddsb (%edx), %ymm4, %ymm4 

// CHECK: vpaddsb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xec,0xe4]      
vpaddsb %ymm4, %ymm4, %ymm4 

// CHECK: vpaddsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xed,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xed,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xed,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xed,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096, %ymm4, %ymm4 

// CHECK: vpaddsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xed,0x64,0x02,0x40]      
vpaddsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xed,0x22]      
vpaddsw (%edx), %ymm4, %ymm4 

// CHECK: vpaddsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xed,0xe4]      
vpaddsw %ymm4, %ymm4, %ymm4 

// CHECK: vpaddusb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdc,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddusb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddusb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdc,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddusb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdc,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddusb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdc,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096, %ymm4, %ymm4 

// CHECK: vpaddusb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdc,0x64,0x02,0x40]      
vpaddusb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddusb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdc,0x22]      
vpaddusb (%edx), %ymm4, %ymm4 

// CHECK: vpaddusb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdc,0xe4]      
vpaddusb %ymm4, %ymm4, %ymm4 

// CHECK: vpaddusw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdd,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddusw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddusw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdd,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddusw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdd,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddusw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdd,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096, %ymm4, %ymm4 

// CHECK: vpaddusw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdd,0x64,0x02,0x40]      
vpaddusw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddusw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdd,0x22]      
vpaddusw (%edx), %ymm4, %ymm4 

// CHECK: vpaddusw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdd,0xe4]      
vpaddusw %ymm4, %ymm4, %ymm4 

// CHECK: vpaddw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfd,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpaddw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfd,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpaddw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfd,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpaddw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfd,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096, %ymm4, %ymm4 

// CHECK: vpaddw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfd,0x64,0x02,0x40]      
vpaddw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpaddw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfd,0x22]      
vpaddw (%edx), %ymm4, %ymm4 

// CHECK: vpaddw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfd,0xe4]      
vpaddw %ymm4, %ymm4, %ymm4 

// CHECK: vpalignr $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0f,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpalignr $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpalignr $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpalignr $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0f,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpalignr $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0f,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096, %ymm4, %ymm4 

// CHECK: vpalignr $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0f,0x64,0x02,0x40,0x00]     
vpalignr $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpalignr $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0f,0x22,0x00]     
vpalignr $0, (%edx), %ymm4, %ymm4 

// CHECK: vpalignr $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0f,0xe4,0x00]     
vpalignr $0, %ymm4, %ymm4, %ymm4 

// CHECK: vpand -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdb,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpand -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpand 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdb,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpand 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdb,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpand 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdb,0x25,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096, %ymm4, %ymm4 

// CHECK: vpand 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdb,0x64,0x02,0x40]      
vpand 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpand (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdb,0x22]      
vpand (%edx), %ymm4, %ymm4 

// CHECK: vpandn -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdf,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpandn -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpandn 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdf,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpandn 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdf,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpandn 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdf,0x25,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096, %ymm4, %ymm4 

// CHECK: vpandn 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdf,0x64,0x02,0x40]      
vpandn 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpandn (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdf,0x22]      
vpandn (%edx), %ymm4, %ymm4 

// CHECK: vpandn %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdf,0xe4]      
vpandn %ymm4, %ymm4, %ymm4 

// CHECK: vpand %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xdb,0xe4]      
vpand %ymm4, %ymm4, %ymm4 

// CHECK: vpavgb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe0,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpavgb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpavgb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe0,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpavgb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe0,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpavgb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe0,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096, %ymm4, %ymm4 

// CHECK: vpavgb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe0,0x64,0x02,0x40]      
vpavgb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpavgb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe0,0x22]      
vpavgb (%edx), %ymm4, %ymm4 

// CHECK: vpavgb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe0,0xe4]      
vpavgb %ymm4, %ymm4, %ymm4 

// CHECK: vpavgw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpavgw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpavgw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpavgw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe3,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpavgw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe3,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096, %ymm4, %ymm4 

// CHECK: vpavgw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe3,0x64,0x02,0x40]      
vpavgw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpavgw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe3,0x22]      
vpavgw (%edx), %ymm4, %ymm4 

// CHECK: vpavgw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe3,0xe4]      
vpavgw %ymm4, %ymm4, %ymm4 

// CHECK: vpblendd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x02,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpblendd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpblendd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x02,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpblendd $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x02,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpblendd $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpblendd $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x02,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpblendd $0, 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x02,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpblendd $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x02,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpblendd $0, 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x02,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096, %xmm1, %xmm1 

// CHECK: vpblendd $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x02,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096, %ymm4, %ymm4 

// CHECK: vpblendd $0, 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x02,0x4c,0x02,0x40,0x00]     
vpblendd $0, 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpblendd $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x02,0x64,0x02,0x40,0x00]     
vpblendd $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpblendd $0, (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x02,0x0a,0x00]     
vpblendd $0, (%edx), %xmm1, %xmm1 

// CHECK: vpblendd $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x02,0x22,0x00]     
vpblendd $0, (%edx), %ymm4, %ymm4 

// CHECK: vpblendd $0, %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe3,0x71,0x02,0xc9,0x00]     
vpblendd $0, %xmm1, %xmm1, %xmm1 

// CHECK: vpblendd $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x02,0xe4,0x00]     
vpblendd $0, %ymm4, %ymm4, %ymm4 

// CHECK: vpblendvb %ymm4, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4c,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x40]     
vpblendvb %ymm4, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpblendvb %ymm4, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x40]     
vpblendvb %ymm4, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpblendvb %ymm4, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4c,0xa2,0xf0,0x1c,0xf0,0x1c,0x40]     
vpblendvb %ymm4, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpblendvb %ymm4, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4c,0x25,0xf0,0x1c,0xf0,0x1c,0x40]     
vpblendvb %ymm4, 485498096, %ymm4, %ymm4 

// CHECK: vpblendvb %ymm4, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4c,0x64,0x02,0x40,0x40]     
vpblendvb %ymm4, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpblendvb %ymm4, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4c,0x22,0x40]     
vpblendvb %ymm4, (%edx), %ymm4, %ymm4 

// CHECK: vpblendvb %ymm4, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x4c,0xe4,0x40]     
vpblendvb %ymm4, %ymm4, %ymm4, %ymm4 

// CHECK: vpblendw $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0e,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vpblendw $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpblendw $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpblendw $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0e,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpblendw $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0e,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096, %ymm4, %ymm4 

// CHECK: vpblendw $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0e,0x64,0x02,0x40,0x00]     
vpblendw $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpblendw $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0e,0x22,0x00]     
vpblendw $0, (%edx), %ymm4, %ymm4 

// CHECK: vpblendw $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x0e,0xe4,0x00]     
vpblendw $0, %ymm4, %ymm4, %ymm4 

// CHECK: vpbroadcastb -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastb -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastb 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastb -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastb -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastb 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastb 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096(%edx), %xmm1 

// CHECK: vpbroadcastb 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096(%edx), %ymm4 

// CHECK: vpbroadcastb 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096, %xmm1 

// CHECK: vpbroadcastb 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096, %ymm4 

// CHECK: vpbroadcastb 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x4c,0x02,0x40]       
vpbroadcastb 64(%edx,%eax), %xmm1 

// CHECK: vpbroadcastb 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x64,0x02,0x40]       
vpbroadcastb 64(%edx,%eax), %ymm4 

// CHECK: vpbroadcastb (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x0a]       
vpbroadcastb (%edx), %xmm1 

// CHECK: vpbroadcastb (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x22]       
vpbroadcastb (%edx), %ymm4 

// CHECK: vpbroadcastb %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0xc9]       
vpbroadcastb %xmm1, %xmm1 

// CHECK: vpbroadcastb %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0xe1]       
vpbroadcastb %xmm1, %ymm4 

// CHECK: vpbroadcastd -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastd -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastd 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastd 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096(%edx), %xmm1 

// CHECK: vpbroadcastd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096(%edx), %ymm4 

// CHECK: vpbroadcastd 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096, %xmm1 

// CHECK: vpbroadcastd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096, %ymm4 

// CHECK: vpbroadcastd 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x4c,0x02,0x40]       
vpbroadcastd 64(%edx,%eax), %xmm1 

// CHECK: vpbroadcastd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x64,0x02,0x40]       
vpbroadcastd 64(%edx,%eax), %ymm4 

// CHECK: vpbroadcastd (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x0a]       
vpbroadcastd (%edx), %xmm1 

// CHECK: vpbroadcastd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x22]       
vpbroadcastd (%edx), %ymm4 

// CHECK: vpbroadcastd %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0xc9]       
vpbroadcastd %xmm1, %xmm1 

// CHECK: vpbroadcastd %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0xe1]       
vpbroadcastd %xmm1, %ymm4 

// CHECK: vpbroadcastq -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastq -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastq 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastq 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096(%edx), %xmm1 

// CHECK: vpbroadcastq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096(%edx), %ymm4 

// CHECK: vpbroadcastq 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096, %xmm1 

// CHECK: vpbroadcastq 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096, %ymm4 

// CHECK: vpbroadcastq 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x4c,0x02,0x40]       
vpbroadcastq 64(%edx,%eax), %xmm1 

// CHECK: vpbroadcastq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x64,0x02,0x40]       
vpbroadcastq 64(%edx,%eax), %ymm4 

// CHECK: vpbroadcastq (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x0a]       
vpbroadcastq (%edx), %xmm1 

// CHECK: vpbroadcastq (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x22]       
vpbroadcastq (%edx), %ymm4 

// CHECK: vpbroadcastq %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0xc9]       
vpbroadcastq %xmm1, %xmm1 

// CHECK: vpbroadcastq %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0xe1]       
vpbroadcastq %xmm1, %ymm4 

// CHECK: vpbroadcastw -485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x8c,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastw -485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastw 485498096(%edx,%eax,4), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096(%edx,%eax,4), %xmm1 

// CHECK: vpbroadcastw -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpbroadcastw -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastw 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpbroadcastw 485498096(%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x8a,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096(%edx), %xmm1 

// CHECK: vpbroadcastw 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096(%edx), %ymm4 

// CHECK: vpbroadcastw 485498096, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x0d,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096, %xmm1 

// CHECK: vpbroadcastw 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096, %ymm4 

// CHECK: vpbroadcastw 64(%edx,%eax), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x4c,0x02,0x40]       
vpbroadcastw 64(%edx,%eax), %xmm1 

// CHECK: vpbroadcastw 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x64,0x02,0x40]       
vpbroadcastw 64(%edx,%eax), %ymm4 

// CHECK: vpbroadcastw (%edx), %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x0a]       
vpbroadcastw (%edx), %xmm1 

// CHECK: vpbroadcastw (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x22]       
vpbroadcastw (%edx), %ymm4 

// CHECK: vpbroadcastw %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0xc9]       
vpbroadcastw %xmm1, %xmm1 

// CHECK: vpbroadcastw %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0xe1]       
vpbroadcastw %xmm1, %ymm4 

// CHECK: vpcmpeqb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x74,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x74,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x74,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x74,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096, %ymm4, %ymm4 

// CHECK: vpcmpeqb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x74,0x64,0x02,0x40]      
vpcmpeqb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpeqb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x74,0x22]      
vpcmpeqb (%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x74,0xe4]      
vpcmpeqb %ymm4, %ymm4, %ymm4 

// CHECK: vpcmpeqd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x76,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x76,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x76,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x76,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096, %ymm4, %ymm4 

// CHECK: vpcmpeqd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x76,0x64,0x02,0x40]      
vpcmpeqd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpeqd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x76,0x22]      
vpcmpeqd (%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x76,0xe4]      
vpcmpeqd %ymm4, %ymm4, %ymm4 

// CHECK: vpcmpeqq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x29,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x29,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x29,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x29,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096, %ymm4, %ymm4 

// CHECK: vpcmpeqq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x29,0x64,0x02,0x40]      
vpcmpeqq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpeqq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x29,0x22]      
vpcmpeqq (%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x29,0xe4]      
vpcmpeqq %ymm4, %ymm4, %ymm4 

// CHECK: vpcmpeqw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x75,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpeqw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x75,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpeqw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x75,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x75,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096, %ymm4, %ymm4 

// CHECK: vpcmpeqw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x75,0x64,0x02,0x40]      
vpcmpeqw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpeqw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x75,0x22]      
vpcmpeqw (%edx), %ymm4, %ymm4 

// CHECK: vpcmpeqw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x75,0xe4]      
vpcmpeqw %ymm4, %ymm4, %ymm4 

// CHECK: vpcmpgtb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x64,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x64,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x64,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x64,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096, %ymm4, %ymm4 

// CHECK: vpcmpgtb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x64,0x64,0x02,0x40]      
vpcmpgtb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpgtb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x64,0x22]      
vpcmpgtb (%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x64,0xe4]      
vpcmpgtb %ymm4, %ymm4, %ymm4 

// CHECK: vpcmpgtd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x66,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x66,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x66,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x66,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096, %ymm4, %ymm4 

// CHECK: vpcmpgtd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x66,0x64,0x02,0x40]      
vpcmpgtd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpgtd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x66,0x22]      
vpcmpgtd (%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x66,0xe4]      
vpcmpgtd %ymm4, %ymm4, %ymm4 

// CHECK: vpcmpgtq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x37,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x37,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x37,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x37,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096, %ymm4, %ymm4 

// CHECK: vpcmpgtq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x37,0x64,0x02,0x40]      
vpcmpgtq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpgtq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x37,0x22]      
vpcmpgtq (%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x37,0xe4]      
vpcmpgtq %ymm4, %ymm4, %ymm4 

// CHECK: vpcmpgtw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x65,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpcmpgtw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x65,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpcmpgtw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x65,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x65,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096, %ymm4, %ymm4 

// CHECK: vpcmpgtw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x65,0x64,0x02,0x40]      
vpcmpgtw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpcmpgtw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x65,0x22]      
vpcmpgtw (%edx), %ymm4, %ymm4 

// CHECK: vpcmpgtw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x65,0xe4]      
vpcmpgtw %ymm4, %ymm4, %ymm4 

// CHECK: vperm2i128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x46,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]     
vperm2i128 $0, -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vperm2i128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x46,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2i128 $0, 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vperm2i128 $0, 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x46,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2i128 $0, 485498096(%edx), %ymm4, %ymm4 

// CHECK: vperm2i128 $0, 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x46,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2i128 $0, 485498096, %ymm4, %ymm4 

// CHECK: vperm2i128 $0, 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x46,0x64,0x02,0x40,0x00]     
vperm2i128 $0, 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vperm2i128 $0, (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x46,0x22,0x00]     
vperm2i128 $0, (%edx), %ymm4, %ymm4 

// CHECK: vperm2i128 $0, %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0x5d,0x46,0xe4,0x00]     
vperm2i128 $0, %ymm4, %ymm4, %ymm4 

// CHECK: vpermd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x36,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpermd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x36,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpermd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x36,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpermd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpermd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x36,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermd 485498096, %ymm4, %ymm4 

// CHECK: vpermd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x36,0x64,0x02,0x40]      
vpermd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpermd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x36,0x22]      
vpermd (%edx), %ymm4, %ymm4 

// CHECK: vpermd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x36,0xe4]      
vpermd %ymm4, %ymm4, %ymm4 

// CHECK: vpermpd $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpermpd $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermpd $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermpd $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermpd $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermpd $0, 485498096(%edx), %ymm4 

// CHECK: vpermpd $0, 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermpd $0, 485498096, %ymm4 

// CHECK: vpermpd $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x64,0x02,0x40,0x00]      
vpermpd $0, 64(%edx,%eax), %ymm4 

// CHECK: vpermpd $0, (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x22,0x00]      
vpermpd $0, (%edx), %ymm4 

// CHECK: vpermpd $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0xe4,0x00]      
vpermpd $0, %ymm4, %ymm4 

// CHECK: vpermps -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x16,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpermps -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermps 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x16,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpermps 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpermps 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x16,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpermps 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpermps 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x16,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermps 485498096, %ymm4, %ymm4 

// CHECK: vpermps 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x16,0x64,0x02,0x40]      
vpermps 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpermps (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x16,0x22]      
vpermps (%edx), %ymm4, %ymm4 

// CHECK: vpermps %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x16,0xe4]      
vpermps %ymm4, %ymm4, %ymm4 

// CHECK: vpermq $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpermq $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermq $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermq $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpermq $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermq $0, 485498096(%edx), %ymm4 

// CHECK: vpermq $0, 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermq $0, 485498096, %ymm4 

// CHECK: vpermq $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x64,0x02,0x40,0x00]      
vpermq $0, 64(%edx,%eax), %ymm4 

// CHECK: vpermq $0, (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x22,0x00]      
vpermq $0, (%edx), %ymm4 

// CHECK: vpermq $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0xe4,0x00]      
vpermq $0, %ymm4, %ymm4 

// CHECK: vphaddd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x02,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vphaddd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphaddd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x02,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphaddd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x02,0xa2,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vphaddd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x02,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096, %ymm4, %ymm4 

// CHECK: vphaddd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x02,0x64,0x02,0x40]      
vphaddd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vphaddd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x02,0x22]      
vphaddd (%edx), %ymm4, %ymm4 

// CHECK: vphaddd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x02,0xe4]      
vphaddd %ymm4, %ymm4, %ymm4 

// CHECK: vphaddsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x03,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vphaddsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphaddsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x03,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphaddsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x03,0xa2,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vphaddsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x03,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096, %ymm4, %ymm4 

// CHECK: vphaddsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x03,0x64,0x02,0x40]      
vphaddsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vphaddsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x03,0x22]      
vphaddsw (%edx), %ymm4, %ymm4 

// CHECK: vphaddsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x03,0xe4]      
vphaddsw %ymm4, %ymm4, %ymm4 

// CHECK: vphaddw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x01,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vphaddw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphaddw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x01,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphaddw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x01,0xa2,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vphaddw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x01,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096, %ymm4, %ymm4 

// CHECK: vphaddw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x01,0x64,0x02,0x40]      
vphaddw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vphaddw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x01,0x22]      
vphaddw (%edx), %ymm4, %ymm4 

// CHECK: vphaddw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x01,0xe4]      
vphaddw %ymm4, %ymm4, %ymm4 

// CHECK: vphsubd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x06,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vphsubd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphsubd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x06,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphsubd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x06,0xa2,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vphsubd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x06,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096, %ymm4, %ymm4 

// CHECK: vphsubd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x06,0x64,0x02,0x40]      
vphsubd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vphsubd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x06,0x22]      
vphsubd (%edx), %ymm4, %ymm4 

// CHECK: vphsubd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x06,0xe4]      
vphsubd %ymm4, %ymm4, %ymm4 

// CHECK: vphsubsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x07,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vphsubsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphsubsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x07,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphsubsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x07,0xa2,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vphsubsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x07,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096, %ymm4, %ymm4 

// CHECK: vphsubsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x07,0x64,0x02,0x40]      
vphsubsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vphsubsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x07,0x22]      
vphsubsw (%edx), %ymm4, %ymm4 

// CHECK: vphsubsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x07,0xe4]      
vphsubsw %ymm4, %ymm4, %ymm4 

// CHECK: vphsubw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x05,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vphsubw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphsubw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x05,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vphsubw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x05,0xa2,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vphsubw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x05,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096, %ymm4, %ymm4 

// CHECK: vphsubw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x05,0x64,0x02,0x40]      
vphsubw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vphsubw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x05,0x22]      
vphsubw (%edx), %ymm4, %ymm4 

// CHECK: vphsubw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x05,0xe4]      
vphsubw %ymm4, %ymm4, %ymm4 

// CHECK: vpmaddubsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x04,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaddubsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaddubsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x04,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaddubsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x04,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaddubsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x04,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096, %ymm4, %ymm4 

// CHECK: vpmaddubsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x04,0x64,0x02,0x40]      
vpmaddubsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaddubsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x04,0x22]      
vpmaddubsw (%edx), %ymm4, %ymm4 

// CHECK: vpmaddubsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x04,0xe4]      
vpmaddubsw %ymm4, %ymm4, %ymm4 

// CHECK: vpmaddwd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf5,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaddwd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaddwd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf5,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaddwd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf5,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaddwd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf5,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096, %ymm4, %ymm4 

// CHECK: vpmaddwd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf5,0x64,0x02,0x40]      
vpmaddwd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaddwd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf5,0x22]      
vpmaddwd (%edx), %ymm4, %ymm4 

// CHECK: vpmaddwd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf5,0xe4]      
vpmaddwd %ymm4, %ymm4, %ymm4 

// CHECK: vpmaskmovd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaskmovd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaskmovd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaskmovd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaskmovd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaskmovd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaskmovd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096, %xmm1, %xmm1 

// CHECK: vpmaskmovd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096, %ymm4, %ymm4 

// CHECK: vpmaskmovd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8c,0x4c,0x02,0x40]      
vpmaskmovd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaskmovd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8c,0x64,0x02,0x40]      
vpmaskmovd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaskmovd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8c,0x0a]      
vpmaskmovd (%edx), %xmm1, %xmm1 

// CHECK: vpmaskmovd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8c,0x22]      
vpmaskmovd (%edx), %ymm4, %ymm4 

// CHECK: vpmaskmovd %xmm1, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovd %xmm1, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vpmaskmovd %xmm1, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %xmm1, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vpmaskmovd %xmm1, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %xmm1, %xmm1, 485498096(%edx) 

// CHECK: vpmaskmovd %xmm1, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %xmm1, %xmm1, 485498096 

// CHECK: vpmaskmovd %xmm1, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8e,0x4c,0x02,0x40]      
vpmaskmovd %xmm1, %xmm1, 64(%edx,%eax) 

// CHECK: vpmaskmovd %xmm1, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0x71,0x8e,0x0a]      
vpmaskmovd %xmm1, %xmm1, (%edx) 

// CHECK: vpmaskmovd %ymm4, %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovd %ymm4, %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vpmaskmovd %ymm4, %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %ymm4, %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vpmaskmovd %ymm4, %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %ymm4, %ymm4, 485498096(%edx) 

// CHECK: vpmaskmovd %ymm4, %ymm4, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8e,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %ymm4, %ymm4, 485498096 

// CHECK: vpmaskmovd %ymm4, %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8e,0x64,0x02,0x40]      
vpmaskmovd %ymm4, %ymm4, 64(%edx,%eax) 

// CHECK: vpmaskmovd %ymm4, %ymm4, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x8e,0x22]      
vpmaskmovd %ymm4, %ymm4, (%edx) 

// CHECK: vpmaskmovq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaskmovq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpmaskmovq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaskmovq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaskmovq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8c,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpmaskmovq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaskmovq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8c,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096, %xmm1, %xmm1 

// CHECK: vpmaskmovq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096, %ymm4, %ymm4 

// CHECK: vpmaskmovq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8c,0x4c,0x02,0x40]      
vpmaskmovq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpmaskmovq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8c,0x64,0x02,0x40]      
vpmaskmovq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaskmovq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8c,0x0a]      
vpmaskmovq (%edx), %xmm1, %xmm1 

// CHECK: vpmaskmovq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8c,0x22]      
vpmaskmovq (%edx), %ymm4, %ymm4 

// CHECK: vpmaskmovq %xmm1, %xmm1, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovq %xmm1, %xmm1, -485498096(%edx,%eax,4) 

// CHECK: vpmaskmovq %xmm1, %xmm1, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %xmm1, %xmm1, 485498096(%edx,%eax,4) 

// CHECK: vpmaskmovq %xmm1, %xmm1, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8e,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %xmm1, %xmm1, 485498096(%edx) 

// CHECK: vpmaskmovq %xmm1, %xmm1, 485498096 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8e,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %xmm1, %xmm1, 485498096 

// CHECK: vpmaskmovq %xmm1, %xmm1, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8e,0x4c,0x02,0x40]      
vpmaskmovq %xmm1, %xmm1, 64(%edx,%eax) 

// CHECK: vpmaskmovq %xmm1, %xmm1, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x8e,0x0a]      
vpmaskmovq %xmm1, %xmm1, (%edx) 

// CHECK: vpmaskmovq %ymm4, %ymm4, -485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaskmovq %ymm4, %ymm4, -485498096(%edx,%eax,4) 

// CHECK: vpmaskmovq %ymm4, %ymm4, 485498096(%edx,%eax,4) 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %ymm4, %ymm4, 485498096(%edx,%eax,4) 

// CHECK: vpmaskmovq %ymm4, %ymm4, 485498096(%edx) 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %ymm4, %ymm4, 485498096(%edx) 

// CHECK: vpmaskmovq %ymm4, %ymm4, 485498096 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8e,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %ymm4, %ymm4, 485498096 

// CHECK: vpmaskmovq %ymm4, %ymm4, 64(%edx,%eax) 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8e,0x64,0x02,0x40]      
vpmaskmovq %ymm4, %ymm4, 64(%edx,%eax) 

// CHECK: vpmaskmovq %ymm4, %ymm4, (%edx) 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x8e,0x22]      
vpmaskmovq %ymm4, %ymm4, (%edx) 

// CHECK: vpmaxsb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxsb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxsb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxsb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaxsb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096, %ymm4, %ymm4 

// CHECK: vpmaxsb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3c,0x64,0x02,0x40]      
vpmaxsb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaxsb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3c,0x22]      
vpmaxsb (%edx), %ymm4, %ymm4 

// CHECK: vpmaxsb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3c,0xe4]      
vpmaxsb %ymm4, %ymm4, %ymm4 

// CHECK: vpmaxsd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxsd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxsd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxsd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaxsd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3d,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096, %ymm4, %ymm4 

// CHECK: vpmaxsd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3d,0x64,0x02,0x40]      
vpmaxsd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaxsd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3d,0x22]      
vpmaxsd (%edx), %ymm4, %ymm4 

// CHECK: vpmaxsd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3d,0xe4]      
vpmaxsd %ymm4, %ymm4, %ymm4 

// CHECK: vpmaxsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xee,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xee,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xee,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaxsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xee,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096, %ymm4, %ymm4 

// CHECK: vpmaxsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xee,0x64,0x02,0x40]      
vpmaxsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaxsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xee,0x22]      
vpmaxsw (%edx), %ymm4, %ymm4 

// CHECK: vpmaxsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xee,0xe4]      
vpmaxsw %ymm4, %ymm4, %ymm4 

// CHECK: vpmaxub -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xde,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxub -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxub 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xde,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxub 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xde,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaxub 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xde,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096, %ymm4, %ymm4 

// CHECK: vpmaxub 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xde,0x64,0x02,0x40]      
vpmaxub 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaxub (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xde,0x22]      
vpmaxub (%edx), %ymm4, %ymm4 

// CHECK: vpmaxub %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xde,0xe4]      
vpmaxub %ymm4, %ymm4, %ymm4 

// CHECK: vpmaxud -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3f,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxud -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxud 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3f,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxud 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3f,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaxud 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3f,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096, %ymm4, %ymm4 

// CHECK: vpmaxud 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3f,0x64,0x02,0x40]      
vpmaxud 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaxud (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3f,0x22]      
vpmaxud (%edx), %ymm4, %ymm4 

// CHECK: vpmaxud %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3f,0xe4]      
vpmaxud %ymm4, %ymm4, %ymm4 

// CHECK: vpmaxuw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3e,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmaxuw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxuw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3e,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmaxuw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3e,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmaxuw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3e,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096, %ymm4, %ymm4 

// CHECK: vpmaxuw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3e,0x64,0x02,0x40]      
vpmaxuw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmaxuw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3e,0x22]      
vpmaxuw (%edx), %ymm4, %ymm4 

// CHECK: vpmaxuw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3e,0xe4]      
vpmaxuw %ymm4, %ymm4, %ymm4 

// CHECK: vpminsb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x38,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpminsb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminsb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x38,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminsb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x38,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpminsb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x38,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096, %ymm4, %ymm4 

// CHECK: vpminsb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x38,0x64,0x02,0x40]      
vpminsb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpminsb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x38,0x22]      
vpminsb (%edx), %ymm4, %ymm4 

// CHECK: vpminsb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x38,0xe4]      
vpminsb %ymm4, %ymm4, %ymm4 

// CHECK: vpminsd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x39,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpminsd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminsd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x39,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminsd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x39,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpminsd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x39,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096, %ymm4, %ymm4 

// CHECK: vpminsd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x39,0x64,0x02,0x40]      
vpminsd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpminsd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x39,0x22]      
vpminsd (%edx), %ymm4, %ymm4 

// CHECK: vpminsd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x39,0xe4]      
vpminsd %ymm4, %ymm4, %ymm4 

// CHECK: vpminsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xea,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpminsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xea,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xea,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpminsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xea,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096, %ymm4, %ymm4 

// CHECK: vpminsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xea,0x64,0x02,0x40]      
vpminsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpminsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xea,0x22]      
vpminsw (%edx), %ymm4, %ymm4 

// CHECK: vpminsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xea,0xe4]      
vpminsw %ymm4, %ymm4, %ymm4 

// CHECK: vpminub -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xda,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpminub -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminub 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xda,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminub 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xda,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpminub 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xda,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096, %ymm4, %ymm4 

// CHECK: vpminub 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xda,0x64,0x02,0x40]      
vpminub 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpminub (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xda,0x22]      
vpminub (%edx), %ymm4, %ymm4 

// CHECK: vpminub %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xda,0xe4]      
vpminub %ymm4, %ymm4, %ymm4 

// CHECK: vpminud -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpminud -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminud 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminud 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3b,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpminud 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3b,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096, %ymm4, %ymm4 

// CHECK: vpminud 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3b,0x64,0x02,0x40]      
vpminud 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpminud (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3b,0x22]      
vpminud (%edx), %ymm4, %ymm4 

// CHECK: vpminud %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3b,0xe4]      
vpminud %ymm4, %ymm4, %ymm4 

// CHECK: vpminuw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpminuw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminuw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpminuw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3a,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpminuw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3a,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096, %ymm4, %ymm4 

// CHECK: vpminuw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3a,0x64,0x02,0x40]      
vpminuw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpminuw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3a,0x22]      
vpminuw (%edx), %ymm4, %ymm4 

// CHECK: vpminuw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x3a,0xe4]      
vpminuw %ymm4, %ymm4, %ymm4 

// CHECK: vpmovsxbd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxbd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxbd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxbd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096(%edx), %ymm4 

// CHECK: vpmovsxbd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096, %ymm4 

// CHECK: vpmovsxbd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x64,0x02,0x40]       
vpmovsxbd 64(%edx,%eax), %ymm4 

// CHECK: vpmovsxbd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x22]       
vpmovsxbd (%edx), %ymm4 

// CHECK: vpmovsxbd %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0xe1]       
vpmovsxbd %xmm1, %ymm4 

// CHECK: vpmovsxbq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxbq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxbq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxbq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096(%edx), %ymm4 

// CHECK: vpmovsxbq 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096, %ymm4 

// CHECK: vpmovsxbq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x64,0x02,0x40]       
vpmovsxbq 64(%edx,%eax), %ymm4 

// CHECK: vpmovsxbq (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x22]       
vpmovsxbq (%edx), %ymm4 

// CHECK: vpmovsxbq %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0xe1]       
vpmovsxbq %xmm1, %ymm4 

// CHECK: vpmovsxbw -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxbw -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxbw 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxbw 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096(%edx), %ymm4 

// CHECK: vpmovsxbw 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096, %ymm4 

// CHECK: vpmovsxbw 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x64,0x02,0x40]       
vpmovsxbw 64(%edx,%eax), %ymm4 

// CHECK: vpmovsxbw (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x22]       
vpmovsxbw (%edx), %ymm4 

// CHECK: vpmovsxbw %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0xe1]       
vpmovsxbw %xmm1, %ymm4 

// CHECK: vpmovsxdq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxdq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxdq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxdq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096(%edx), %ymm4 

// CHECK: vpmovsxdq 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096, %ymm4 

// CHECK: vpmovsxdq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x64,0x02,0x40]       
vpmovsxdq 64(%edx,%eax), %ymm4 

// CHECK: vpmovsxdq (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x22]       
vpmovsxdq (%edx), %ymm4 

// CHECK: vpmovsxdq %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0xe1]       
vpmovsxdq %xmm1, %ymm4 

// CHECK: vpmovsxwd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxwd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxwd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxwd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096(%edx), %ymm4 

// CHECK: vpmovsxwd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096, %ymm4 

// CHECK: vpmovsxwd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x64,0x02,0x40]       
vpmovsxwd 64(%edx,%eax), %ymm4 

// CHECK: vpmovsxwd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x22]       
vpmovsxwd (%edx), %ymm4 

// CHECK: vpmovsxwd %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0xe1]       
vpmovsxwd %xmm1, %ymm4 

// CHECK: vpmovsxwq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovsxwq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxwq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovsxwq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096(%edx), %ymm4 

// CHECK: vpmovsxwq 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096, %ymm4 

// CHECK: vpmovsxwq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x64,0x02,0x40]       
vpmovsxwq 64(%edx,%eax), %ymm4 

// CHECK: vpmovsxwq (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x22]       
vpmovsxwq (%edx), %ymm4 

// CHECK: vpmovsxwq %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0xe1]       
vpmovsxwq %xmm1, %ymm4 

// CHECK: vpmovzxbd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxbd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxbd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxbd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096(%edx), %ymm4 

// CHECK: vpmovzxbd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096, %ymm4 

// CHECK: vpmovzxbd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x64,0x02,0x40]       
vpmovzxbd 64(%edx,%eax), %ymm4 

// CHECK: vpmovzxbd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x22]       
vpmovzxbd (%edx), %ymm4 

// CHECK: vpmovzxbd %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0xe1]       
vpmovzxbd %xmm1, %ymm4 

// CHECK: vpmovzxbq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxbq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxbq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxbq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096(%edx), %ymm4 

// CHECK: vpmovzxbq 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096, %ymm4 

// CHECK: vpmovzxbq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x64,0x02,0x40]       
vpmovzxbq 64(%edx,%eax), %ymm4 

// CHECK: vpmovzxbq (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x22]       
vpmovzxbq (%edx), %ymm4 

// CHECK: vpmovzxbq %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0xe1]       
vpmovzxbq %xmm1, %ymm4 

// CHECK: vpmovzxbw -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxbw -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxbw 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxbw 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096(%edx), %ymm4 

// CHECK: vpmovzxbw 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096, %ymm4 

// CHECK: vpmovzxbw 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x64,0x02,0x40]       
vpmovzxbw 64(%edx,%eax), %ymm4 

// CHECK: vpmovzxbw (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x22]       
vpmovzxbw (%edx), %ymm4 

// CHECK: vpmovzxbw %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0xe1]       
vpmovzxbw %xmm1, %ymm4 

// CHECK: vpmovzxdq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxdq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxdq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxdq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096(%edx), %ymm4 

// CHECK: vpmovzxdq 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096, %ymm4 

// CHECK: vpmovzxdq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x64,0x02,0x40]       
vpmovzxdq 64(%edx,%eax), %ymm4 

// CHECK: vpmovzxdq (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x22]       
vpmovzxdq (%edx), %ymm4 

// CHECK: vpmovzxdq %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0xe1]       
vpmovzxdq %xmm1, %ymm4 

// CHECK: vpmovzxwd -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxwd -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxwd 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxwd 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096(%edx), %ymm4 

// CHECK: vpmovzxwd 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096, %ymm4 

// CHECK: vpmovzxwd 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x64,0x02,0x40]       
vpmovzxwd 64(%edx,%eax), %ymm4 

// CHECK: vpmovzxwd (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x22]       
vpmovzxwd (%edx), %ymm4 

// CHECK: vpmovzxwd %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0xe1]       
vpmovzxwd %xmm1, %ymm4 

// CHECK: vpmovzxwq -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0xa4,0x82,0x10,0xe3,0x0f,0xe3]       
vpmovzxwq -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxwq 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpmovzxwq 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0xa2,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096(%edx), %ymm4 

// CHECK: vpmovzxwq 485498096, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096, %ymm4 

// CHECK: vpmovzxwq 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x64,0x02,0x40]       
vpmovzxwq 64(%edx,%eax), %ymm4 

// CHECK: vpmovzxwq (%edx), %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x22]       
vpmovzxwq (%edx), %ymm4 

// CHECK: vpmovzxwq %xmm1, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0xe1]       
vpmovzxwq %xmm1, %ymm4 

// CHECK: vpmuldq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x28,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmuldq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmuldq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x28,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmuldq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x28,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmuldq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x28,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096, %ymm4, %ymm4 

// CHECK: vpmuldq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x28,0x64,0x02,0x40]      
vpmuldq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmuldq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x28,0x22]      
vpmuldq (%edx), %ymm4, %ymm4 

// CHECK: vpmuldq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x28,0xe4]      
vpmuldq %ymm4, %ymm4, %ymm4 

// CHECK: vpmulhrsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0b,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulhrsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulhrsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0b,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulhrsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0b,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmulhrsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0b,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096, %ymm4, %ymm4 

// CHECK: vpmulhrsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0b,0x64,0x02,0x40]      
vpmulhrsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmulhrsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0b,0x22]      
vpmulhrsw (%edx), %ymm4, %ymm4 

// CHECK: vpmulhrsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0b,0xe4]      
vpmulhrsw %ymm4, %ymm4, %ymm4 

// CHECK: vpmulhuw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe4,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulhuw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulhuw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe4,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulhuw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe4,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmulhuw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe4,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096, %ymm4, %ymm4 

// CHECK: vpmulhuw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe4,0x64,0x02,0x40]      
vpmulhuw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmulhuw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe4,0x22]      
vpmulhuw (%edx), %ymm4, %ymm4 

// CHECK: vpmulhuw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe4,0xe4]      
vpmulhuw %ymm4, %ymm4, %ymm4 

// CHECK: vpmulhw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe5,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulhw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulhw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe5,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulhw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe5,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmulhw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe5,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096, %ymm4, %ymm4 

// CHECK: vpmulhw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe5,0x64,0x02,0x40]      
vpmulhw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmulhw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe5,0x22]      
vpmulhw (%edx), %ymm4, %ymm4 

// CHECK: vpmulhw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe5,0xe4]      
vpmulhw %ymm4, %ymm4, %ymm4 

// CHECK: vpmulld -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x40,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmulld -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulld 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x40,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmulld 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x40,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmulld 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x40,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096, %ymm4, %ymm4 

// CHECK: vpmulld 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x40,0x64,0x02,0x40]      
vpmulld 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmulld (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x40,0x22]      
vpmulld (%edx), %ymm4, %ymm4 

// CHECK: vpmulld %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x40,0xe4]      
vpmulld %ymm4, %ymm4, %ymm4 

// CHECK: vpmullw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd5,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmullw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmullw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd5,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmullw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd5,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmullw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd5,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096, %ymm4, %ymm4 

// CHECK: vpmullw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd5,0x64,0x02,0x40]      
vpmullw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmullw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd5,0x22]      
vpmullw (%edx), %ymm4, %ymm4 

// CHECK: vpmullw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd5,0xe4]      
vpmullw %ymm4, %ymm4, %ymm4 

// CHECK: vpmuludq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf4,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpmuludq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmuludq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf4,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpmuludq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf4,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpmuludq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf4,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096, %ymm4, %ymm4 

// CHECK: vpmuludq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf4,0x64,0x02,0x40]      
vpmuludq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpmuludq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf4,0x22]      
vpmuludq (%edx), %ymm4, %ymm4 

// CHECK: vpmuludq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf4,0xe4]      
vpmuludq %ymm4, %ymm4, %ymm4 

// CHECK: vpor -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xeb,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpor -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpor 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xeb,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpor 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xeb,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpor 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xeb,0x25,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096, %ymm4, %ymm4 

// CHECK: vpor 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xeb,0x64,0x02,0x40]      
vpor 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpor (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xeb,0x22]      
vpor (%edx), %ymm4, %ymm4 

// CHECK: vpor %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xeb,0xe4]      
vpor %ymm4, %ymm4, %ymm4 

// CHECK: vpsadbw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf6,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsadbw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsadbw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf6,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsadbw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf6,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsadbw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf6,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096, %ymm4, %ymm4 

// CHECK: vpsadbw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf6,0x64,0x02,0x40]      
vpsadbw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsadbw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf6,0x22]      
vpsadbw (%edx), %ymm4, %ymm4 

// CHECK: vpsadbw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf6,0xe4]      
vpsadbw %ymm4, %ymm4, %ymm4 

// CHECK: vpshufb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x00,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpshufb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpshufb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x00,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpshufb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x00,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpshufb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x00,0x25,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096, %ymm4, %ymm4 

// CHECK: vpshufb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x00,0x64,0x02,0x40]      
vpshufb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpshufb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x00,0x22]      
vpshufb (%edx), %ymm4, %ymm4 

// CHECK: vpshufb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x00,0xe4]      
vpshufb %ymm4, %ymm4, %ymm4 

// CHECK: vpshufd $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x70,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpshufd $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpshufd $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x70,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpshufd $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x70,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096(%edx), %ymm4 

// CHECK: vpshufd $0, 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x70,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096, %ymm4 

// CHECK: vpshufd $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x70,0x64,0x02,0x40,0x00]      
vpshufd $0, 64(%edx,%eax), %ymm4 

// CHECK: vpshufd $0, (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x70,0x22,0x00]      
vpshufd $0, (%edx), %ymm4 

// CHECK: vpshufd $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfd,0x70,0xe4,0x00]      
vpshufd $0, %ymm4, %ymm4 

// CHECK: vpshufhw $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x70,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpshufhw $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpshufhw $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x70,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpshufhw $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x70,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096(%edx), %ymm4 

// CHECK: vpshufhw $0, 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x70,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096, %ymm4 

// CHECK: vpshufhw $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x70,0x64,0x02,0x40,0x00]      
vpshufhw $0, 64(%edx,%eax), %ymm4 

// CHECK: vpshufhw $0, (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x70,0x22,0x00]      
vpshufhw $0, (%edx), %ymm4 

// CHECK: vpshufhw $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xfe,0x70,0xe4,0x00]      
vpshufhw $0, %ymm4, %ymm4 

// CHECK: vpshuflw $0, -485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x70,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]      
vpshuflw $0, -485498096(%edx,%eax,4), %ymm4 

// CHECK: vpshuflw $0, 485498096(%edx,%eax,4), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x70,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096(%edx,%eax,4), %ymm4 

// CHECK: vpshuflw $0, 485498096(%edx), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x70,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096(%edx), %ymm4 

// CHECK: vpshuflw $0, 485498096, %ymm4 
// CHECK: encoding: [0xc5,0xff,0x70,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096, %ymm4 

// CHECK: vpshuflw $0, 64(%edx,%eax), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x70,0x64,0x02,0x40,0x00]      
vpshuflw $0, 64(%edx,%eax), %ymm4 

// CHECK: vpshuflw $0, (%edx), %ymm4 
// CHECK: encoding: [0xc5,0xff,0x70,0x22,0x00]      
vpshuflw $0, (%edx), %ymm4 

// CHECK: vpshuflw $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xff,0x70,0xe4,0x00]      
vpshuflw $0, %ymm4, %ymm4 

// CHECK: vpsignb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x08,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsignb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsignb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x08,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsignb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x08,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsignb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x08,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096, %ymm4, %ymm4 

// CHECK: vpsignb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x08,0x64,0x02,0x40]      
vpsignb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsignb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x08,0x22]      
vpsignb (%edx), %ymm4, %ymm4 

// CHECK: vpsignb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x08,0xe4]      
vpsignb %ymm4, %ymm4, %ymm4 

// CHECK: vpsignd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsignd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsignd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsignd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0a,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsignd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0a,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096, %ymm4, %ymm4 

// CHECK: vpsignd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0a,0x64,0x02,0x40]      
vpsignd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsignd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0a,0x22]      
vpsignd (%edx), %ymm4, %ymm4 

// CHECK: vpsignd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x0a,0xe4]      
vpsignd %ymm4, %ymm4, %ymm4 

// CHECK: vpsignw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x09,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsignw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsignw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x09,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsignw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x09,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsignw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x09,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096, %ymm4, %ymm4 

// CHECK: vpsignw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x09,0x64,0x02,0x40]      
vpsignw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsignw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x09,0x22]      
vpsignw (%edx), %ymm4, %ymm4 

// CHECK: vpsignw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x09,0xe4]      
vpsignw %ymm4, %ymm4, %ymm4 

// CHECK: vpslld $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x72,0xf4,0x00]      
vpslld $0, %ymm4, %ymm4 

// CHECK: vpslld -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf2,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpslld -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpslld 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpslld 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf2,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpslld 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf2,0x25,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096, %ymm4, %ymm4 

// CHECK: vpslld 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf2,0x64,0x02,0x40]      
vpslld 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpslld (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf2,0x22]      
vpslld (%edx), %ymm4, %ymm4 

// CHECK: vpslldq $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x73,0xfc,0x00]      
vpslldq $0, %ymm4, %ymm4 

// CHECK: vpslld %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf2,0xe1]      
vpslld %xmm1, %ymm4, %ymm4 

// CHECK: vpsllq $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x73,0xf4,0x00]      
vpsllq $0, %ymm4, %ymm4 

// CHECK: vpsllq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf3,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsllq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf3,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096, %ymm4, %ymm4 

// CHECK: vpsllq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf3,0x64,0x02,0x40]      
vpsllq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsllq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf3,0x22]      
vpsllq (%edx), %ymm4, %ymm4 

// CHECK: vpsllq %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf3,0xe1]      
vpsllq %xmm1, %ymm4, %ymm4 

// CHECK: vpsllvd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x47,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllvd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllvd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x47,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllvd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x47,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllvd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllvd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x47,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllvd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x47,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsllvd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x47,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsllvd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x47,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096, %xmm1, %xmm1 

// CHECK: vpsllvd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x47,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096, %ymm4, %ymm4 

// CHECK: vpsllvd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x47,0x4c,0x02,0x40]      
vpsllvd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsllvd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x47,0x64,0x02,0x40]      
vpsllvd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsllvd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x47,0x0a]      
vpsllvd (%edx), %xmm1, %xmm1 

// CHECK: vpsllvd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x47,0x22]      
vpsllvd (%edx), %ymm4, %ymm4 

// CHECK: vpsllvd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x47,0xc9]      
vpsllvd %xmm1, %xmm1, %xmm1 

// CHECK: vpsllvd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x47,0xe4]      
vpsllvd %ymm4, %ymm4, %ymm4 

// CHECK: vpsllvq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x47,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllvq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllvq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x47,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsllvq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x47,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllvq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllvq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x47,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllvq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x47,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsllvq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x47,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsllvq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x47,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096, %xmm1, %xmm1 

// CHECK: vpsllvq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x47,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096, %ymm4, %ymm4 

// CHECK: vpsllvq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x47,0x4c,0x02,0x40]      
vpsllvq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsllvq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x47,0x64,0x02,0x40]      
vpsllvq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsllvq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x47,0x0a]      
vpsllvq (%edx), %xmm1, %xmm1 

// CHECK: vpsllvq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x47,0x22]      
vpsllvq (%edx), %ymm4, %ymm4 

// CHECK: vpsllvq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x47,0xc9]      
vpsllvq %xmm1, %xmm1, %xmm1 

// CHECK: vpsllvq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x47,0xe4]      
vpsllvq %ymm4, %ymm4, %ymm4 

// CHECK: vpsllw $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x71,0xf4,0x00]      
vpsllw $0, %ymm4, %ymm4 

// CHECK: vpsllw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsllw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsllw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf1,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsllw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf1,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096, %ymm4, %ymm4 

// CHECK: vpsllw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf1,0x64,0x02,0x40]      
vpsllw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsllw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf1,0x22]      
vpsllw (%edx), %ymm4, %ymm4 

// CHECK: vpsllw %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf1,0xe1]      
vpsllw %xmm1, %ymm4, %ymm4 

// CHECK: vpsrad $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x72,0xe4,0x00]      
vpsrad $0, %ymm4, %ymm4 

// CHECK: vpsrad -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe2,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrad -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrad 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrad 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe2,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsrad 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe2,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096, %ymm4, %ymm4 

// CHECK: vpsrad 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe2,0x64,0x02,0x40]      
vpsrad 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsrad (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe2,0x22]      
vpsrad (%edx), %ymm4, %ymm4 

// CHECK: vpsrad %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe2,0xe1]      
vpsrad %xmm1, %ymm4, %ymm4 

// CHECK: vpsravd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x46,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsravd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsravd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x46,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsravd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x46,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsravd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsravd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x46,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsravd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x46,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsravd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x46,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsravd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x46,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096, %xmm1, %xmm1 

// CHECK: vpsravd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x46,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096, %ymm4, %ymm4 

// CHECK: vpsravd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x46,0x4c,0x02,0x40]      
vpsravd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsravd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x46,0x64,0x02,0x40]      
vpsravd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsravd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x46,0x0a]      
vpsravd (%edx), %xmm1, %xmm1 

// CHECK: vpsravd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x46,0x22]      
vpsravd (%edx), %ymm4, %ymm4 

// CHECK: vpsravd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x46,0xc9]      
vpsravd %xmm1, %xmm1, %xmm1 

// CHECK: vpsravd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x46,0xe4]      
vpsravd %ymm4, %ymm4, %ymm4 

// CHECK: vpsraw $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x71,0xe4,0x00]      
vpsraw $0, %ymm4, %ymm4 

// CHECK: vpsraw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsraw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsraw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsraw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe1,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsraw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe1,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096, %ymm4, %ymm4 

// CHECK: vpsraw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe1,0x64,0x02,0x40]      
vpsraw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsraw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe1,0x22]      
vpsraw (%edx), %ymm4, %ymm4 

// CHECK: vpsraw %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe1,0xe1]      
vpsraw %xmm1, %ymm4, %ymm4 

// CHECK: vpsrld $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x72,0xd4,0x00]      
vpsrld $0, %ymm4, %ymm4 

// CHECK: vpsrld -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd2,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrld -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrld 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd2,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrld 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd2,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsrld 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd2,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096, %ymm4, %ymm4 

// CHECK: vpsrld 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd2,0x64,0x02,0x40]      
vpsrld 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsrld (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd2,0x22]      
vpsrld (%edx), %ymm4, %ymm4 

// CHECK: vpsrldq $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x73,0xdc,0x00]      
vpsrldq $0, %ymm4, %ymm4 

// CHECK: vpsrld %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd2,0xe1]      
vpsrld %xmm1, %ymm4, %ymm4 

// CHECK: vpsrlq $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x73,0xd4,0x00]      
vpsrlq $0, %ymm4, %ymm4 

// CHECK: vpsrlq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd3,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd3,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd3,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsrlq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd3,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096, %ymm4, %ymm4 

// CHECK: vpsrlq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd3,0x64,0x02,0x40]      
vpsrlq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsrlq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd3,0x22]      
vpsrlq (%edx), %ymm4, %ymm4 

// CHECK: vpsrlq %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd3,0xe1]      
vpsrlq %xmm1, %ymm4, %ymm4 

// CHECK: vpsrlvd -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x45,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlvd -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlvd 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x45,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlvd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x45,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlvd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlvd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x45,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlvd 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x45,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsrlvd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x45,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsrlvd 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x45,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096, %xmm1, %xmm1 

// CHECK: vpsrlvd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x45,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096, %ymm4, %ymm4 

// CHECK: vpsrlvd 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x45,0x4c,0x02,0x40]      
vpsrlvd 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsrlvd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x45,0x64,0x02,0x40]      
vpsrlvd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsrlvd (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x45,0x0a]      
vpsrlvd (%edx), %xmm1, %xmm1 

// CHECK: vpsrlvd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x45,0x22]      
vpsrlvd (%edx), %ymm4, %ymm4 

// CHECK: vpsrlvd %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0x71,0x45,0xc9]      
vpsrlvd %xmm1, %xmm1, %xmm1 

// CHECK: vpsrlvd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0x5d,0x45,0xe4]      
vpsrlvd %ymm4, %ymm4, %ymm4 

// CHECK: vpsrlvq -485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x45,0x8c,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlvq -485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlvq 485498096(%edx,%eax,4), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x45,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096(%edx,%eax,4), %xmm1, %xmm1 

// CHECK: vpsrlvq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x45,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlvq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlvq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x45,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlvq 485498096(%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x45,0x8a,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096(%edx), %xmm1, %xmm1 

// CHECK: vpsrlvq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x45,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsrlvq 485498096, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x45,0x0d,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096, %xmm1, %xmm1 

// CHECK: vpsrlvq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x45,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096, %ymm4, %ymm4 

// CHECK: vpsrlvq 64(%edx,%eax), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x45,0x4c,0x02,0x40]      
vpsrlvq 64(%edx,%eax), %xmm1, %xmm1 

// CHECK: vpsrlvq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x45,0x64,0x02,0x40]      
vpsrlvq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsrlvq (%edx), %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x45,0x0a]      
vpsrlvq (%edx), %xmm1, %xmm1 

// CHECK: vpsrlvq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x45,0x22]      
vpsrlvq (%edx), %ymm4, %ymm4 

// CHECK: vpsrlvq %xmm1, %xmm1, %xmm1 
// CHECK: encoding: [0xc4,0xe2,0xf1,0x45,0xc9]      
vpsrlvq %xmm1, %xmm1, %xmm1 

// CHECK: vpsrlvq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc4,0xe2,0xdd,0x45,0xe4]      
vpsrlvq %ymm4, %ymm4, %ymm4 

// CHECK: vpsrlw $0, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x71,0xd4,0x00]      
vpsrlw $0, %ymm4, %ymm4 

// CHECK: vpsrlw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd1,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsrlw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd1,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsrlw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd1,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsrlw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd1,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096, %ymm4, %ymm4 

// CHECK: vpsrlw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd1,0x64,0x02,0x40]      
vpsrlw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsrlw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd1,0x22]      
vpsrlw (%edx), %ymm4, %ymm4 

// CHECK: vpsrlw %xmm1, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd1,0xe1]      
vpsrlw %xmm1, %ymm4, %ymm4 

// CHECK: vpsubb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf8,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf8,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096, %ymm4, %ymm4 

// CHECK: vpsubb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf8,0x64,0x02,0x40]      
vpsubb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf8,0x22]      
vpsubb (%edx), %ymm4, %ymm4 

// CHECK: vpsubb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf8,0xe4]      
vpsubb %ymm4, %ymm4, %ymm4 

// CHECK: vpsubd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfa,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfa,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfa,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfa,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096, %ymm4, %ymm4 

// CHECK: vpsubd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfa,0x64,0x02,0x40]      
vpsubd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfa,0x22]      
vpsubd (%edx), %ymm4, %ymm4 

// CHECK: vpsubd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfa,0xe4]      
vpsubd %ymm4, %ymm4, %ymm4 

// CHECK: vpsubq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfb,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfb,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfb,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfb,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096, %ymm4, %ymm4 

// CHECK: vpsubq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfb,0x64,0x02,0x40]      
vpsubq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfb,0x22]      
vpsubq (%edx), %ymm4, %ymm4 

// CHECK: vpsubq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xfb,0xe4]      
vpsubq %ymm4, %ymm4, %ymm4 

// CHECK: vpsubsb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubsb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubsb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubsb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe8,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubsb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe8,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096, %ymm4, %ymm4 

// CHECK: vpsubsb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe8,0x64,0x02,0x40]      
vpsubsb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubsb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe8,0x22]      
vpsubsb (%edx), %ymm4, %ymm4 

// CHECK: vpsubsb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe8,0xe4]      
vpsubsb %ymm4, %ymm4, %ymm4 

// CHECK: vpsubsw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe9,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubsw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubsw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe9,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubsw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe9,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubsw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe9,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096, %ymm4, %ymm4 

// CHECK: vpsubsw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe9,0x64,0x02,0x40]      
vpsubsw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubsw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe9,0x22]      
vpsubsw (%edx), %ymm4, %ymm4 

// CHECK: vpsubsw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xe9,0xe4]      
vpsubsw %ymm4, %ymm4, %ymm4 

// CHECK: vpsubusb -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd8,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubusb -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubusb 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd8,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubusb 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd8,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubusb 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd8,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096, %ymm4, %ymm4 

// CHECK: vpsubusb 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd8,0x64,0x02,0x40]      
vpsubusb 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubusb (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd8,0x22]      
vpsubusb (%edx), %ymm4, %ymm4 

// CHECK: vpsubusb %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd8,0xe4]      
vpsubusb %ymm4, %ymm4, %ymm4 

// CHECK: vpsubusw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd9,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubusw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubusw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd9,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubusw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd9,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubusw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd9,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096, %ymm4, %ymm4 

// CHECK: vpsubusw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd9,0x64,0x02,0x40]      
vpsubusw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubusw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd9,0x22]      
vpsubusw (%edx), %ymm4, %ymm4 

// CHECK: vpsubusw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xd9,0xe4]      
vpsubusw %ymm4, %ymm4, %ymm4 

// CHECK: vpsubw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf9,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpsubw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf9,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpsubw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf9,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpsubw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf9,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096, %ymm4, %ymm4 

// CHECK: vpsubw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf9,0x64,0x02,0x40]      
vpsubw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpsubw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf9,0x22]      
vpsubw (%edx), %ymm4, %ymm4 

// CHECK: vpsubw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xf9,0xe4]      
vpsubw %ymm4, %ymm4, %ymm4 

// CHECK: vpunpckhbw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x68,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhbw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhbw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x68,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhbw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x68,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpckhbw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x68,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096, %ymm4, %ymm4 

// CHECK: vpunpckhbw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x68,0x64,0x02,0x40]      
vpunpckhbw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpckhbw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x68,0x22]      
vpunpckhbw (%edx), %ymm4, %ymm4 

// CHECK: vpunpckhbw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x68,0xe4]      
vpunpckhbw %ymm4, %ymm4, %ymm4 

// CHECK: vpunpckhdq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6a,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhdq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhdq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6a,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhdq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6a,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpckhdq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6a,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096, %ymm4, %ymm4 

// CHECK: vpunpckhdq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6a,0x64,0x02,0x40]      
vpunpckhdq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpckhdq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6a,0x22]      
vpunpckhdq (%edx), %ymm4, %ymm4 

// CHECK: vpunpckhdq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6a,0xe4]      
vpunpckhdq %ymm4, %ymm4, %ymm4 

// CHECK: vpunpckhqdq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6d,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhqdq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhqdq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6d,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhqdq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6d,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpckhqdq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6d,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096, %ymm4, %ymm4 

// CHECK: vpunpckhqdq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6d,0x64,0x02,0x40]      
vpunpckhqdq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpckhqdq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6d,0x22]      
vpunpckhqdq (%edx), %ymm4, %ymm4 

// CHECK: vpunpckhqdq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6d,0xe4]      
vpunpckhqdq %ymm4, %ymm4, %ymm4 

// CHECK: vpunpckhwd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x69,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckhwd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhwd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x69,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckhwd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x69,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpckhwd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x69,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096, %ymm4, %ymm4 

// CHECK: vpunpckhwd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x69,0x64,0x02,0x40]      
vpunpckhwd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpckhwd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x69,0x22]      
vpunpckhwd (%edx), %ymm4, %ymm4 

// CHECK: vpunpckhwd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x69,0xe4]      
vpunpckhwd %ymm4, %ymm4, %ymm4 

// CHECK: vpunpcklbw -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x60,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpcklbw -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpcklbw 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x60,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpcklbw 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x60,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpcklbw 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x60,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096, %ymm4, %ymm4 

// CHECK: vpunpcklbw 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x60,0x64,0x02,0x40]      
vpunpcklbw 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpcklbw (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x60,0x22]      
vpunpcklbw (%edx), %ymm4, %ymm4 

// CHECK: vpunpcklbw %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x60,0xe4]      
vpunpcklbw %ymm4, %ymm4, %ymm4 

// CHECK: vpunpckldq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x62,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpckldq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckldq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x62,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpckldq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x62,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpckldq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x62,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096, %ymm4, %ymm4 

// CHECK: vpunpckldq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x62,0x64,0x02,0x40]      
vpunpckldq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpckldq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x62,0x22]      
vpunpckldq (%edx), %ymm4, %ymm4 

// CHECK: vpunpckldq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x62,0xe4]      
vpunpckldq %ymm4, %ymm4, %ymm4 

// CHECK: vpunpcklqdq -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6c,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpcklqdq -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpcklqdq 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6c,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpcklqdq 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6c,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpcklqdq 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096, %ymm4, %ymm4 

// CHECK: vpunpcklqdq 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6c,0x64,0x02,0x40]      
vpunpcklqdq 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpcklqdq (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6c,0x22]      
vpunpcklqdq (%edx), %ymm4, %ymm4 

// CHECK: vpunpcklqdq %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x6c,0xe4]      
vpunpcklqdq %ymm4, %ymm4, %ymm4 

// CHECK: vpunpcklwd -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x61,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpunpcklwd -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpcklwd 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x61,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpunpcklwd 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x61,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpunpcklwd 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x61,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096, %ymm4, %ymm4 

// CHECK: vpunpcklwd 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x61,0x64,0x02,0x40]      
vpunpcklwd 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpunpcklwd (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x61,0x22]      
vpunpcklwd (%edx), %ymm4, %ymm4 

// CHECK: vpunpcklwd %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0x61,0xe4]      
vpunpcklwd %ymm4, %ymm4, %ymm4 

// CHECK: vpxor -485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xef,0xa4,0x82,0x10,0xe3,0x0f,0xe3]      
vpxor -485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpxor 485498096(%edx,%eax,4), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xef,0xa4,0x82,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096(%edx,%eax,4), %ymm4, %ymm4 

// CHECK: vpxor 485498096(%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xef,0xa2,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096(%edx), %ymm4, %ymm4 

// CHECK: vpxor 485498096, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xef,0x25,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096, %ymm4, %ymm4 

// CHECK: vpxor 64(%edx,%eax), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xef,0x64,0x02,0x40]      
vpxor 64(%edx,%eax), %ymm4, %ymm4 

// CHECK: vpxor (%edx), %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xef,0x22]      
vpxor (%edx), %ymm4, %ymm4 

// CHECK: vpxor %ymm4, %ymm4, %ymm4 
// CHECK: encoding: [0xc5,0xdd,0xef,0xe4]      
vpxor %ymm4, %ymm4, %ymm4 

