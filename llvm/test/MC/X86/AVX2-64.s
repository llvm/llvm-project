// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vbroadcasti128 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcasti128 485498096, %ymm7 

// CHECK: vbroadcasti128 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vbroadcasti128 485498096, %ymm9 

// CHECK: vbroadcasti128 -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x7c,0x82,0xc0]       
vbroadcasti128 -64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcasti128 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x7c,0x82,0x40]       
vbroadcasti128 64(%rdx,%rax,4), %ymm7 

// CHECK: vbroadcasti128 -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x5a,0x4c,0x82,0xc0]       
vbroadcasti128 -64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcasti128 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x5a,0x4c,0x82,0x40]       
vbroadcasti128 64(%rdx,%rax,4), %ymm9 

// CHECK: vbroadcasti128 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x7c,0x02,0x40]       
vbroadcasti128 64(%rdx,%rax), %ymm7 

// CHECK: vbroadcasti128 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x5a,0x4c,0x02,0x40]       
vbroadcasti128 64(%rdx,%rax), %ymm9 

// CHECK: vbroadcasti128 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x7a,0x40]       
vbroadcasti128 64(%rdx), %ymm7 

// CHECK: vbroadcasti128 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x5a,0x4a,0x40]       
vbroadcasti128 64(%rdx), %ymm9 

// CHECK: vbroadcasti128 (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x5a,0x3a]       
vbroadcasti128 (%rdx), %ymm7 

// CHECK: vbroadcasti128 (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x5a,0x0a]       
vbroadcasti128 (%rdx), %ymm9 

// CHECK: vbroadcastsd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x19,0xcf]       
vbroadcastsd %xmm15, %ymm9 

// CHECK: vbroadcastsd %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0xfe]       
vbroadcastsd %xmm6, %ymm7 

// CHECK: vbroadcastss %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x18,0xff]       
vbroadcastss %xmm15, %xmm15 

// CHECK: vbroadcastss %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x18,0xcf]       
vbroadcastss %xmm15, %ymm9 

// CHECK: vbroadcastss %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0xf6]       
vbroadcastss %xmm6, %xmm6 

// CHECK: vbroadcastss %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0xfe]       
vbroadcastss %xmm6, %ymm7 

// CHECK: vextracti128 $0, %ymm7, 485498096 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextracti128 $0, %ymm7, 485498096 

// CHECK: vextracti128 $0, %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x7a,0x40,0x00]      
vextracti128 $0, %ymm7, 64(%rdx) 

// CHECK: vextracti128 $0, %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x7c,0x02,0x40,0x00]      
vextracti128 $0, %ymm7, 64(%rdx,%rax) 

// CHECK: vextracti128 $0, %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x7c,0x82,0xc0,0x00]      
vextracti128 $0, %ymm7, -64(%rdx,%rax,4) 

// CHECK: vextracti128 $0, %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x7c,0x82,0x40,0x00]      
vextracti128 $0, %ymm7, 64(%rdx,%rax,4) 

// CHECK: vextracti128 $0, %ymm7, (%rdx) 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0x3a,0x00]      
vextracti128 $0, %ymm7, (%rdx) 

// CHECK: vextracti128 $0, %ymm7, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x7d,0x39,0xfe,0x00]      
vextracti128 $0, %ymm7, %xmm6 

// CHECK: vextracti128 $0, %ymm9, 485498096 
// CHECK: encoding: [0xc4,0x63,0x7d,0x39,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vextracti128 $0, %ymm9, 485498096 

// CHECK: vextracti128 $0, %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x39,0x4a,0x40,0x00]      
vextracti128 $0, %ymm9, 64(%rdx) 

// CHECK: vextracti128 $0, %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x39,0x4c,0x02,0x40,0x00]      
vextracti128 $0, %ymm9, 64(%rdx,%rax) 

// CHECK: vextracti128 $0, %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x39,0x4c,0x82,0xc0,0x00]      
vextracti128 $0, %ymm9, -64(%rdx,%rax,4) 

// CHECK: vextracti128 $0, %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x39,0x4c,0x82,0x40,0x00]      
vextracti128 $0, %ymm9, 64(%rdx,%rax,4) 

// CHECK: vextracti128 $0, %ymm9, (%rdx) 
// CHECK: encoding: [0xc4,0x63,0x7d,0x39,0x0a,0x00]      
vextracti128 $0, %ymm9, (%rdx) 

// CHECK: vextracti128 $0, %ymm9, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x7d,0x39,0xcf,0x00]      
vextracti128 $0, %ymm9, %xmm15 

// CHECK: vinserti128 $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x38,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinserti128 $0, 485498096, %ymm7, %ymm7 

// CHECK: vinserti128 $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x38,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vinserti128 $0, 485498096, %ymm9, %ymm9 

// CHECK: vinserti128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x38,0x7c,0x82,0xc0,0x00]     
vinserti128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vinserti128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x38,0x7c,0x82,0x40,0x00]     
vinserti128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vinserti128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x38,0x4c,0x82,0xc0,0x00]     
vinserti128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vinserti128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x38,0x4c,0x82,0x40,0x00]     
vinserti128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vinserti128 $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x38,0x7c,0x02,0x40,0x00]     
vinserti128 $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vinserti128 $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x38,0x4c,0x02,0x40,0x00]     
vinserti128 $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vinserti128 $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x38,0x7a,0x40,0x00]     
vinserti128 $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vinserti128 $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x38,0x4a,0x40,0x00]     
vinserti128 $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vinserti128 $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x38,0x3a,0x00]     
vinserti128 $0, (%rdx), %ymm7, %ymm7 

// CHECK: vinserti128 $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x38,0x0a,0x00]     
vinserti128 $0, (%rdx), %ymm9, %ymm9 

// CHECK: vinserti128 $0, %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x38,0xcf,0x00]     
vinserti128 $0, %xmm15, %ymm9, %ymm9 

// CHECK: vinserti128 $0, %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x38,0xfe,0x00]     
vinserti128 $0, %xmm6, %ymm7, %ymm7 

// CHECK: vmovntdqa 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096, %ymm7 

// CHECK: vmovntdqa 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x2a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vmovntdqa 485498096, %ymm9 

// CHECK: vmovntdqa -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x7c,0x82,0xc0]       
vmovntdqa -64(%rdx,%rax,4), %ymm7 

// CHECK: vmovntdqa 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x7c,0x82,0x40]       
vmovntdqa 64(%rdx,%rax,4), %ymm7 

// CHECK: vmovntdqa -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x2a,0x4c,0x82,0xc0]       
vmovntdqa -64(%rdx,%rax,4), %ymm9 

// CHECK: vmovntdqa 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x2a,0x4c,0x82,0x40]       
vmovntdqa 64(%rdx,%rax,4), %ymm9 

// CHECK: vmovntdqa 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x7c,0x02,0x40]       
vmovntdqa 64(%rdx,%rax), %ymm7 

// CHECK: vmovntdqa 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x2a,0x4c,0x02,0x40]       
vmovntdqa 64(%rdx,%rax), %ymm9 

// CHECK: vmovntdqa 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x7a,0x40]       
vmovntdqa 64(%rdx), %ymm7 

// CHECK: vmovntdqa 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x2a,0x4a,0x40]       
vmovntdqa 64(%rdx), %ymm9 

// CHECK: vmovntdqa (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x2a,0x3a]       
vmovntdqa (%rdx), %ymm7 

// CHECK: vmovntdqa (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x2a,0x0a]       
vmovntdqa (%rdx), %ymm9 

// CHECK: vmpsadbw $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x42,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096, %ymm7, %ymm7 

// CHECK: vmpsadbw $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x42,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vmpsadbw $0, 485498096, %ymm9, %ymm9 

// CHECK: vmpsadbw $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x42,0x7c,0x82,0xc0,0x00]     
vmpsadbw $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmpsadbw $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x42,0x7c,0x82,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vmpsadbw $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x42,0x4c,0x82,0xc0,0x00]     
vmpsadbw $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmpsadbw $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x42,0x4c,0x82,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vmpsadbw $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x42,0x7c,0x02,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vmpsadbw $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x42,0x4c,0x02,0x40,0x00]     
vmpsadbw $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vmpsadbw $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x42,0x7a,0x40,0x00]     
vmpsadbw $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vmpsadbw $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x42,0x4a,0x40,0x00]     
vmpsadbw $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vmpsadbw $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x42,0x3a,0x00]     
vmpsadbw $0, (%rdx), %ymm7, %ymm7 

// CHECK: vmpsadbw $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x42,0x0a,0x00]     
vmpsadbw $0, (%rdx), %ymm9, %ymm9 

// CHECK: vmpsadbw $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x42,0xff,0x00]     
vmpsadbw $0, %ymm7, %ymm7, %ymm7 

// CHECK: vmpsadbw $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x42,0xc9,0x00]     
vmpsadbw $0, %ymm9, %ymm9, %ymm9 

// CHECK: vpabsb 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096, %ymm7 

// CHECK: vpabsb 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsb 485498096, %ymm9 

// CHECK: vpabsb -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x7c,0x82,0xc0]       
vpabsb -64(%rdx,%rax,4), %ymm7 

// CHECK: vpabsb 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x7c,0x82,0x40]       
vpabsb 64(%rdx,%rax,4), %ymm7 

// CHECK: vpabsb -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1c,0x4c,0x82,0xc0]       
vpabsb -64(%rdx,%rax,4), %ymm9 

// CHECK: vpabsb 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1c,0x4c,0x82,0x40]       
vpabsb 64(%rdx,%rax,4), %ymm9 

// CHECK: vpabsb 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x7c,0x02,0x40]       
vpabsb 64(%rdx,%rax), %ymm7 

// CHECK: vpabsb 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1c,0x4c,0x02,0x40]       
vpabsb 64(%rdx,%rax), %ymm9 

// CHECK: vpabsb 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x7a,0x40]       
vpabsb 64(%rdx), %ymm7 

// CHECK: vpabsb 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1c,0x4a,0x40]       
vpabsb 64(%rdx), %ymm9 

// CHECK: vpabsb (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0x3a]       
vpabsb (%rdx), %ymm7 

// CHECK: vpabsb (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1c,0x0a]       
vpabsb (%rdx), %ymm9 

// CHECK: vpabsb %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1c,0xff]       
vpabsb %ymm7, %ymm7 

// CHECK: vpabsb %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x1c,0xc9]       
vpabsb %ymm9, %ymm9 

// CHECK: vpabsd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096, %ymm7 

// CHECK: vpabsd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsd 485498096, %ymm9 

// CHECK: vpabsd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x7c,0x82,0xc0]       
vpabsd -64(%rdx,%rax,4), %ymm7 

// CHECK: vpabsd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x7c,0x82,0x40]       
vpabsd 64(%rdx,%rax,4), %ymm7 

// CHECK: vpabsd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1e,0x4c,0x82,0xc0]       
vpabsd -64(%rdx,%rax,4), %ymm9 

// CHECK: vpabsd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1e,0x4c,0x82,0x40]       
vpabsd 64(%rdx,%rax,4), %ymm9 

// CHECK: vpabsd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x7c,0x02,0x40]       
vpabsd 64(%rdx,%rax), %ymm7 

// CHECK: vpabsd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1e,0x4c,0x02,0x40]       
vpabsd 64(%rdx,%rax), %ymm9 

// CHECK: vpabsd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x7a,0x40]       
vpabsd 64(%rdx), %ymm7 

// CHECK: vpabsd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1e,0x4a,0x40]       
vpabsd 64(%rdx), %ymm9 

// CHECK: vpabsd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0x3a]       
vpabsd (%rdx), %ymm7 

// CHECK: vpabsd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1e,0x0a]       
vpabsd (%rdx), %ymm9 

// CHECK: vpabsd %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1e,0xff]       
vpabsd %ymm7, %ymm7 

// CHECK: vpabsd %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x1e,0xc9]       
vpabsd %ymm9, %ymm9 

// CHECK: vpabsw 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096, %ymm7 

// CHECK: vpabsw 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpabsw 485498096, %ymm9 

// CHECK: vpabsw -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x7c,0x82,0xc0]       
vpabsw -64(%rdx,%rax,4), %ymm7 

// CHECK: vpabsw 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x7c,0x82,0x40]       
vpabsw 64(%rdx,%rax,4), %ymm7 

// CHECK: vpabsw -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1d,0x4c,0x82,0xc0]       
vpabsw -64(%rdx,%rax,4), %ymm9 

// CHECK: vpabsw 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1d,0x4c,0x82,0x40]       
vpabsw 64(%rdx,%rax,4), %ymm9 

// CHECK: vpabsw 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x7c,0x02,0x40]       
vpabsw 64(%rdx,%rax), %ymm7 

// CHECK: vpabsw 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1d,0x4c,0x02,0x40]       
vpabsw 64(%rdx,%rax), %ymm9 

// CHECK: vpabsw 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x7a,0x40]       
vpabsw 64(%rdx), %ymm7 

// CHECK: vpabsw 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1d,0x4a,0x40]       
vpabsw 64(%rdx), %ymm9 

// CHECK: vpabsw (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0x3a]       
vpabsw (%rdx), %ymm7 

// CHECK: vpabsw (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x1d,0x0a]       
vpabsw (%rdx), %ymm9 

// CHECK: vpabsw %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1d,0xff]       
vpabsw %ymm7, %ymm7 

// CHECK: vpabsw %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x1d,0xc9]       
vpabsw %ymm9, %ymm9 

// CHECK: vpackssdw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096, %ymm7, %ymm7 

// CHECK: vpackssdw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackssdw 485498096, %ymm9, %ymm9 

// CHECK: vpackssdw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6b,0x7c,0x82,0xc0]      
vpackssdw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpackssdw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6b,0x7c,0x82,0x40]      
vpackssdw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpackssdw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6b,0x4c,0x82,0xc0]      
vpackssdw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpackssdw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6b,0x4c,0x82,0x40]      
vpackssdw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpackssdw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6b,0x7c,0x02,0x40]      
vpackssdw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpackssdw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6b,0x4c,0x02,0x40]      
vpackssdw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpackssdw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6b,0x7a,0x40]      
vpackssdw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpackssdw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6b,0x4a,0x40]      
vpackssdw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpackssdw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6b,0x3a]      
vpackssdw (%rdx), %ymm7, %ymm7 

// CHECK: vpackssdw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6b,0x0a]      
vpackssdw (%rdx), %ymm9, %ymm9 

// CHECK: vpackssdw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6b,0xff]      
vpackssdw %ymm7, %ymm7, %ymm7 

// CHECK: vpackssdw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x6b,0xc9]      
vpackssdw %ymm9, %ymm9, %ymm9 

// CHECK: vpacksswb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x63,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096, %ymm7, %ymm7 

// CHECK: vpacksswb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x63,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpacksswb 485498096, %ymm9, %ymm9 

// CHECK: vpacksswb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x63,0x7c,0x82,0xc0]      
vpacksswb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpacksswb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x63,0x7c,0x82,0x40]      
vpacksswb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpacksswb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x63,0x4c,0x82,0xc0]      
vpacksswb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpacksswb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x63,0x4c,0x82,0x40]      
vpacksswb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpacksswb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x63,0x7c,0x02,0x40]      
vpacksswb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpacksswb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x63,0x4c,0x02,0x40]      
vpacksswb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpacksswb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x63,0x7a,0x40]      
vpacksswb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpacksswb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x63,0x4a,0x40]      
vpacksswb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpacksswb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x63,0x3a]      
vpacksswb (%rdx), %ymm7, %ymm7 

// CHECK: vpacksswb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x63,0x0a]      
vpacksswb (%rdx), %ymm9, %ymm9 

// CHECK: vpacksswb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x63,0xff]      
vpacksswb %ymm7, %ymm7, %ymm7 

// CHECK: vpacksswb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x63,0xc9]      
vpacksswb %ymm9, %ymm9, %ymm9 

// CHECK: vpackusdw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096, %ymm7, %ymm7 

// CHECK: vpackusdw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackusdw 485498096, %ymm9, %ymm9 

// CHECK: vpackusdw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2b,0x7c,0x82,0xc0]      
vpackusdw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpackusdw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2b,0x7c,0x82,0x40]      
vpackusdw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpackusdw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2b,0x4c,0x82,0xc0]      
vpackusdw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpackusdw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2b,0x4c,0x82,0x40]      
vpackusdw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpackusdw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2b,0x7c,0x02,0x40]      
vpackusdw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpackusdw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2b,0x4c,0x02,0x40]      
vpackusdw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpackusdw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2b,0x7a,0x40]      
vpackusdw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpackusdw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2b,0x4a,0x40]      
vpackusdw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpackusdw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2b,0x3a]      
vpackusdw (%rdx), %ymm7, %ymm7 

// CHECK: vpackusdw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x2b,0x0a]      
vpackusdw (%rdx), %ymm9, %ymm9 

// CHECK: vpackusdw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x2b,0xff]      
vpackusdw %ymm7, %ymm7, %ymm7 

// CHECK: vpackusdw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x2b,0xc9]      
vpackusdw %ymm9, %ymm9, %ymm9 

// CHECK: vpackuswb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x67,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096, %ymm7, %ymm7 

// CHECK: vpackuswb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x67,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpackuswb 485498096, %ymm9, %ymm9 

// CHECK: vpackuswb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x67,0x7c,0x82,0xc0]      
vpackuswb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpackuswb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x67,0x7c,0x82,0x40]      
vpackuswb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpackuswb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x67,0x4c,0x82,0xc0]      
vpackuswb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpackuswb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x67,0x4c,0x82,0x40]      
vpackuswb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpackuswb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x67,0x7c,0x02,0x40]      
vpackuswb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpackuswb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x67,0x4c,0x02,0x40]      
vpackuswb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpackuswb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x67,0x7a,0x40]      
vpackuswb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpackuswb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x67,0x4a,0x40]      
vpackuswb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpackuswb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x67,0x3a]      
vpackuswb (%rdx), %ymm7, %ymm7 

// CHECK: vpackuswb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x67,0x0a]      
vpackuswb (%rdx), %ymm9, %ymm9 

// CHECK: vpackuswb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x67,0xff]      
vpackuswb %ymm7, %ymm7, %ymm7 

// CHECK: vpackuswb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x67,0xc9]      
vpackuswb %ymm9, %ymm9, %ymm9 

// CHECK: vpaddb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096, %ymm7, %ymm7 

// CHECK: vpaddb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfc,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddb 485498096, %ymm9, %ymm9 

// CHECK: vpaddb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfc,0x7c,0x82,0xc0]      
vpaddb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfc,0x7c,0x82,0x40]      
vpaddb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfc,0x4c,0x82,0xc0]      
vpaddb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfc,0x4c,0x82,0x40]      
vpaddb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfc,0x7c,0x02,0x40]      
vpaddb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfc,0x4c,0x02,0x40]      
vpaddb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfc,0x7a,0x40]      
vpaddb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfc,0x4a,0x40]      
vpaddb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfc,0x3a]      
vpaddb (%rdx), %ymm7, %ymm7 

// CHECK: vpaddb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfc,0x0a]      
vpaddb (%rdx), %ymm9, %ymm9 

// CHECK: vpaddb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfc,0xff]      
vpaddb %ymm7, %ymm7, %ymm7 

// CHECK: vpaddb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xfc,0xc9]      
vpaddb %ymm9, %ymm9, %ymm9 

// CHECK: vpaddd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfe,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096, %ymm7, %ymm7 

// CHECK: vpaddd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfe,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddd 485498096, %ymm9, %ymm9 

// CHECK: vpaddd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfe,0x7c,0x82,0xc0]      
vpaddd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfe,0x7c,0x82,0x40]      
vpaddd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfe,0x4c,0x82,0xc0]      
vpaddd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfe,0x4c,0x82,0x40]      
vpaddd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfe,0x7c,0x02,0x40]      
vpaddd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfe,0x4c,0x02,0x40]      
vpaddd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfe,0x7a,0x40]      
vpaddd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfe,0x4a,0x40]      
vpaddd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfe,0x3a]      
vpaddd (%rdx), %ymm7, %ymm7 

// CHECK: vpaddd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfe,0x0a]      
vpaddd (%rdx), %ymm9, %ymm9 

// CHECK: vpaddd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfe,0xff]      
vpaddd %ymm7, %ymm7, %ymm7 

// CHECK: vpaddd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xfe,0xc9]      
vpaddd %ymm9, %ymm9, %ymm9 

// CHECK: vpaddq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd4,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096, %ymm7, %ymm7 

// CHECK: vpaddq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd4,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddq 485498096, %ymm9, %ymm9 

// CHECK: vpaddq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd4,0x7c,0x82,0xc0]      
vpaddq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd4,0x7c,0x82,0x40]      
vpaddq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd4,0x4c,0x82,0xc0]      
vpaddq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd4,0x4c,0x82,0x40]      
vpaddq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd4,0x7c,0x02,0x40]      
vpaddq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd4,0x4c,0x02,0x40]      
vpaddq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd4,0x7a,0x40]      
vpaddq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd4,0x4a,0x40]      
vpaddq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd4,0x3a]      
vpaddq (%rdx), %ymm7, %ymm7 

// CHECK: vpaddq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd4,0x0a]      
vpaddq (%rdx), %ymm9, %ymm9 

// CHECK: vpaddq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd4,0xff]      
vpaddq %ymm7, %ymm7, %ymm7 

// CHECK: vpaddq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd4,0xc9]      
vpaddq %ymm9, %ymm9, %ymm9 

// CHECK: vpaddsb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xec,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096, %ymm7, %ymm7 

// CHECK: vpaddsb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xec,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsb 485498096, %ymm9, %ymm9 

// CHECK: vpaddsb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xec,0x7c,0x82,0xc0]      
vpaddsb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddsb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xec,0x7c,0x82,0x40]      
vpaddsb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddsb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xec,0x4c,0x82,0xc0]      
vpaddsb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddsb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xec,0x4c,0x82,0x40]      
vpaddsb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddsb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xec,0x7c,0x02,0x40]      
vpaddsb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddsb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xec,0x4c,0x02,0x40]      
vpaddsb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddsb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xec,0x7a,0x40]      
vpaddsb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddsb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xec,0x4a,0x40]      
vpaddsb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddsb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xec,0x3a]      
vpaddsb (%rdx), %ymm7, %ymm7 

// CHECK: vpaddsb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xec,0x0a]      
vpaddsb (%rdx), %ymm9, %ymm9 

// CHECK: vpaddsb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xec,0xff]      
vpaddsb %ymm7, %ymm7, %ymm7 

// CHECK: vpaddsb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xec,0xc9]      
vpaddsb %ymm9, %ymm9, %ymm9 

// CHECK: vpaddsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xed,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096, %ymm7, %ymm7 

// CHECK: vpaddsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xed,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddsw 485498096, %ymm9, %ymm9 

// CHECK: vpaddsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xed,0x7c,0x82,0xc0]      
vpaddsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xed,0x7c,0x82,0x40]      
vpaddsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xed,0x4c,0x82,0xc0]      
vpaddsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xed,0x4c,0x82,0x40]      
vpaddsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xed,0x7c,0x02,0x40]      
vpaddsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xed,0x4c,0x02,0x40]      
vpaddsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xed,0x7a,0x40]      
vpaddsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xed,0x4a,0x40]      
vpaddsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xed,0x3a]      
vpaddsw (%rdx), %ymm7, %ymm7 

// CHECK: vpaddsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xed,0x0a]      
vpaddsw (%rdx), %ymm9, %ymm9 

// CHECK: vpaddsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xed,0xff]      
vpaddsw %ymm7, %ymm7, %ymm7 

// CHECK: vpaddsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xed,0xc9]      
vpaddsw %ymm9, %ymm9, %ymm9 

// CHECK: vpaddusb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdc,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096, %ymm7, %ymm7 

// CHECK: vpaddusb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdc,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusb 485498096, %ymm9, %ymm9 

// CHECK: vpaddusb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdc,0x7c,0x82,0xc0]      
vpaddusb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddusb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdc,0x7c,0x82,0x40]      
vpaddusb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddusb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdc,0x4c,0x82,0xc0]      
vpaddusb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddusb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdc,0x4c,0x82,0x40]      
vpaddusb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddusb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdc,0x7c,0x02,0x40]      
vpaddusb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddusb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdc,0x4c,0x02,0x40]      
vpaddusb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddusb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdc,0x7a,0x40]      
vpaddusb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddusb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdc,0x4a,0x40]      
vpaddusb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddusb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdc,0x3a]      
vpaddusb (%rdx), %ymm7, %ymm7 

// CHECK: vpaddusb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdc,0x0a]      
vpaddusb (%rdx), %ymm9, %ymm9 

// CHECK: vpaddusb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdc,0xff]      
vpaddusb %ymm7, %ymm7, %ymm7 

// CHECK: vpaddusb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xdc,0xc9]      
vpaddusb %ymm9, %ymm9, %ymm9 

// CHECK: vpaddusw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096, %ymm7, %ymm7 

// CHECK: vpaddusw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddusw 485498096, %ymm9, %ymm9 

// CHECK: vpaddusw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdd,0x7c,0x82,0xc0]      
vpaddusw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddusw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdd,0x7c,0x82,0x40]      
vpaddusw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddusw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdd,0x4c,0x82,0xc0]      
vpaddusw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddusw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdd,0x4c,0x82,0x40]      
vpaddusw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddusw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdd,0x7c,0x02,0x40]      
vpaddusw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddusw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdd,0x4c,0x02,0x40]      
vpaddusw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddusw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdd,0x7a,0x40]      
vpaddusw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddusw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdd,0x4a,0x40]      
vpaddusw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddusw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdd,0x3a]      
vpaddusw (%rdx), %ymm7, %ymm7 

// CHECK: vpaddusw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdd,0x0a]      
vpaddusw (%rdx), %ymm9, %ymm9 

// CHECK: vpaddusw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdd,0xff]      
vpaddusw %ymm7, %ymm7, %ymm7 

// CHECK: vpaddusw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xdd,0xc9]      
vpaddusw %ymm9, %ymm9, %ymm9 

// CHECK: vpaddw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096, %ymm7, %ymm7 

// CHECK: vpaddw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpaddw 485498096, %ymm9, %ymm9 

// CHECK: vpaddw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfd,0x7c,0x82,0xc0]      
vpaddw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfd,0x7c,0x82,0x40]      
vpaddw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpaddw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfd,0x4c,0x82,0xc0]      
vpaddw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfd,0x4c,0x82,0x40]      
vpaddw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpaddw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfd,0x7c,0x02,0x40]      
vpaddw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpaddw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfd,0x4c,0x02,0x40]      
vpaddw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpaddw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfd,0x7a,0x40]      
vpaddw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpaddw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfd,0x4a,0x40]      
vpaddw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpaddw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfd,0x3a]      
vpaddw (%rdx), %ymm7, %ymm7 

// CHECK: vpaddw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfd,0x0a]      
vpaddw (%rdx), %ymm9, %ymm9 

// CHECK: vpaddw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfd,0xff]      
vpaddw %ymm7, %ymm7, %ymm7 

// CHECK: vpaddw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xfd,0xc9]      
vpaddw %ymm9, %ymm9, %ymm9 

// CHECK: vpalignr $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096, %ymm7, %ymm7 

// CHECK: vpalignr $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpalignr $0, 485498096, %ymm9, %ymm9 

// CHECK: vpalignr $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0f,0x7c,0x82,0xc0,0x00]     
vpalignr $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpalignr $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0f,0x7c,0x82,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpalignr $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0f,0x4c,0x82,0xc0,0x00]     
vpalignr $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpalignr $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0f,0x4c,0x82,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpalignr $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0f,0x7c,0x02,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpalignr $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0f,0x4c,0x02,0x40,0x00]     
vpalignr $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpalignr $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0f,0x7a,0x40,0x00]     
vpalignr $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vpalignr $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0f,0x4a,0x40,0x00]     
vpalignr $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vpalignr $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0f,0x3a,0x00]     
vpalignr $0, (%rdx), %ymm7, %ymm7 

// CHECK: vpalignr $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0f,0x0a,0x00]     
vpalignr $0, (%rdx), %ymm9, %ymm9 

// CHECK: vpalignr $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0f,0xff,0x00]     
vpalignr $0, %ymm7, %ymm7, %ymm7 

// CHECK: vpalignr $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x0f,0xc9,0x00]     
vpalignr $0, %ymm9, %ymm9, %ymm9 

// CHECK: vpand 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096, %ymm7, %ymm7 

// CHECK: vpand 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpand 485498096, %ymm9, %ymm9 

// CHECK: vpand -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdb,0x7c,0x82,0xc0]      
vpand -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpand 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdb,0x7c,0x82,0x40]      
vpand 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpand -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdb,0x4c,0x82,0xc0]      
vpand -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpand 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdb,0x4c,0x82,0x40]      
vpand 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpand 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdb,0x7c,0x02,0x40]      
vpand 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpand 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdb,0x4c,0x02,0x40]      
vpand 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpand 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdb,0x7a,0x40]      
vpand 64(%rdx), %ymm7, %ymm7 

// CHECK: vpand 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdb,0x4a,0x40]      
vpand 64(%rdx), %ymm9, %ymm9 

// CHECK: vpandn 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096, %ymm7, %ymm7 

// CHECK: vpandn 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpandn 485498096, %ymm9, %ymm9 

// CHECK: vpandn -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdf,0x7c,0x82,0xc0]      
vpandn -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpandn 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdf,0x7c,0x82,0x40]      
vpandn 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpandn -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdf,0x4c,0x82,0xc0]      
vpandn -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpandn 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdf,0x4c,0x82,0x40]      
vpandn 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpandn 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdf,0x7c,0x02,0x40]      
vpandn 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpandn 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdf,0x4c,0x02,0x40]      
vpandn 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpandn 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdf,0x7a,0x40]      
vpandn 64(%rdx), %ymm7, %ymm7 

// CHECK: vpandn 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdf,0x4a,0x40]      
vpandn 64(%rdx), %ymm9, %ymm9 

// CHECK: vpandn (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdf,0x3a]      
vpandn (%rdx), %ymm7, %ymm7 

// CHECK: vpandn (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdf,0x0a]      
vpandn (%rdx), %ymm9, %ymm9 

// CHECK: vpandn %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdf,0xff]      
vpandn %ymm7, %ymm7, %ymm7 

// CHECK: vpandn %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xdf,0xc9]      
vpandn %ymm9, %ymm9, %ymm9 

// CHECK: vpand (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdb,0x3a]      
vpand (%rdx), %ymm7, %ymm7 

// CHECK: vpand (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xdb,0x0a]      
vpand (%rdx), %ymm9, %ymm9 

// CHECK: vpand %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xdb,0xff]      
vpand %ymm7, %ymm7, %ymm7 

// CHECK: vpand %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xdb,0xc9]      
vpand %ymm9, %ymm9, %ymm9 

// CHECK: vpavgb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe0,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096, %ymm7, %ymm7 

// CHECK: vpavgb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe0,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgb 485498096, %ymm9, %ymm9 

// CHECK: vpavgb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe0,0x7c,0x82,0xc0]      
vpavgb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpavgb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe0,0x7c,0x82,0x40]      
vpavgb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpavgb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe0,0x4c,0x82,0xc0]      
vpavgb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpavgb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe0,0x4c,0x82,0x40]      
vpavgb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpavgb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe0,0x7c,0x02,0x40]      
vpavgb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpavgb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe0,0x4c,0x02,0x40]      
vpavgb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpavgb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe0,0x7a,0x40]      
vpavgb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpavgb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe0,0x4a,0x40]      
vpavgb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpavgb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe0,0x3a]      
vpavgb (%rdx), %ymm7, %ymm7 

// CHECK: vpavgb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe0,0x0a]      
vpavgb (%rdx), %ymm9, %ymm9 

// CHECK: vpavgb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe0,0xff]      
vpavgb %ymm7, %ymm7, %ymm7 

// CHECK: vpavgb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe0,0xc9]      
vpavgb %ymm9, %ymm9, %ymm9 

// CHECK: vpavgw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096, %ymm7, %ymm7 

// CHECK: vpavgw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe3,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpavgw 485498096, %ymm9, %ymm9 

// CHECK: vpavgw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe3,0x7c,0x82,0xc0]      
vpavgw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpavgw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe3,0x7c,0x82,0x40]      
vpavgw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpavgw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe3,0x4c,0x82,0xc0]      
vpavgw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpavgw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe3,0x4c,0x82,0x40]      
vpavgw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpavgw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe3,0x7c,0x02,0x40]      
vpavgw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpavgw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe3,0x4c,0x02,0x40]      
vpavgw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpavgw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe3,0x7a,0x40]      
vpavgw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpavgw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe3,0x4a,0x40]      
vpavgw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpavgw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe3,0x3a]      
vpavgw (%rdx), %ymm7, %ymm7 

// CHECK: vpavgw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe3,0x0a]      
vpavgw (%rdx), %ymm9, %ymm9 

// CHECK: vpavgw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe3,0xff]      
vpavgw %ymm7, %ymm7, %ymm7 

// CHECK: vpavgw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe3,0xc9]      
vpavgw %ymm9, %ymm9, %ymm9 

// CHECK: vpblendd $0, 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x02,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096, %xmm15, %xmm15 

// CHECK: vpblendd $0, 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x02,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096, %xmm6, %xmm6 

// CHECK: vpblendd $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x02,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096, %ymm7, %ymm7 

// CHECK: vpblendd $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x02,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendd $0, 485498096, %ymm9, %ymm9 

// CHECK: vpblendd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x02,0x7c,0x82,0xc0,0x00]     
vpblendd $0, -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpblendd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x02,0x7c,0x82,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpblendd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x02,0x74,0x82,0xc0,0x00]     
vpblendd $0, -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpblendd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x02,0x74,0x82,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpblendd $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x02,0x7c,0x82,0xc0,0x00]     
vpblendd $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpblendd $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x02,0x7c,0x82,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpblendd $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x02,0x4c,0x82,0xc0,0x00]     
vpblendd $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpblendd $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x02,0x4c,0x82,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpblendd $0, 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x02,0x7c,0x02,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpblendd $0, 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x02,0x74,0x02,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpblendd $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x02,0x7c,0x02,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpblendd $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x02,0x4c,0x02,0x40,0x00]     
vpblendd $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpblendd $0, 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x02,0x7a,0x40,0x00]     
vpblendd $0, 64(%rdx), %xmm15, %xmm15 

// CHECK: vpblendd $0, 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x02,0x72,0x40,0x00]     
vpblendd $0, 64(%rdx), %xmm6, %xmm6 

// CHECK: vpblendd $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x02,0x7a,0x40,0x00]     
vpblendd $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vpblendd $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x02,0x4a,0x40,0x00]     
vpblendd $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vpblendd $0, (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x63,0x01,0x02,0x3a,0x00]     
vpblendd $0, (%rdx), %xmm15, %xmm15 

// CHECK: vpblendd $0, (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x02,0x32,0x00]     
vpblendd $0, (%rdx), %xmm6, %xmm6 

// CHECK: vpblendd $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x02,0x3a,0x00]     
vpblendd $0, (%rdx), %ymm7, %ymm7 

// CHECK: vpblendd $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x02,0x0a,0x00]     
vpblendd $0, (%rdx), %ymm9, %ymm9 

// CHECK: vpblendd $0, %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x43,0x01,0x02,0xff,0x00]     
vpblendd $0, %xmm15, %xmm15, %xmm15 

// CHECK: vpblendd $0, %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe3,0x49,0x02,0xf6,0x00]     
vpblendd $0, %xmm6, %xmm6, %xmm6 

// CHECK: vpblendd $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x02,0xff,0x00]     
vpblendd $0, %ymm7, %ymm7, %ymm7 

// CHECK: vpblendd $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x02,0xc9,0x00]     
vpblendd $0, %ymm9, %ymm9, %ymm9 

// CHECK: vpblendvb %ymm7, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x70]     
vpblendvb %ymm7, 485498096, %ymm7, %ymm7 

// CHECK: vpblendvb %ymm7, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4c,0x7c,0x82,0xc0,0x70]     
vpblendvb %ymm7, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpblendvb %ymm7, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4c,0x7c,0x82,0x40,0x70]     
vpblendvb %ymm7, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpblendvb %ymm7, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4c,0x7c,0x02,0x40,0x70]     
vpblendvb %ymm7, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpblendvb %ymm7, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4c,0x7a,0x40,0x70]     
vpblendvb %ymm7, 64(%rdx), %ymm7, %ymm7 

// CHECK: vpblendvb %ymm7, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4c,0x3a,0x70]     
vpblendvb %ymm7, (%rdx), %ymm7, %ymm7 

// CHECK: vpblendvb %ymm7, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x4c,0xff,0x70]     
vpblendvb %ymm7, %ymm7, %ymm7, %ymm7 

// CHECK: vpblendvb %ymm9, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x90]     
vpblendvb %ymm9, 485498096, %ymm9, %ymm9 

// CHECK: vpblendvb %ymm9, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4c,0x4c,0x82,0xc0,0x90]     
vpblendvb %ymm9, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpblendvb %ymm9, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4c,0x4c,0x82,0x40,0x90]     
vpblendvb %ymm9, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpblendvb %ymm9, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4c,0x4c,0x02,0x40,0x90]     
vpblendvb %ymm9, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpblendvb %ymm9, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4c,0x4a,0x40,0x90]     
vpblendvb %ymm9, 64(%rdx), %ymm9, %ymm9 

// CHECK: vpblendvb %ymm9, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x4c,0x0a,0x90]     
vpblendvb %ymm9, (%rdx), %ymm9, %ymm9 

// CHECK: vpblendvb %ymm9, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x4c,0xc9,0x90]     
vpblendvb %ymm9, %ymm9, %ymm9, %ymm9 

// CHECK: vpblendw $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096, %ymm7, %ymm7 

// CHECK: vpblendw $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vpblendw $0, 485498096, %ymm9, %ymm9 

// CHECK: vpblendw $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0e,0x7c,0x82,0xc0,0x00]     
vpblendw $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpblendw $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0e,0x7c,0x82,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpblendw $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0e,0x4c,0x82,0xc0,0x00]     
vpblendw $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpblendw $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0e,0x4c,0x82,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpblendw $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0e,0x7c,0x02,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpblendw $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0e,0x4c,0x02,0x40,0x00]     
vpblendw $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpblendw $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0e,0x7a,0x40,0x00]     
vpblendw $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vpblendw $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0e,0x4a,0x40,0x00]     
vpblendw $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vpblendw $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0e,0x3a,0x00]     
vpblendw $0, (%rdx), %ymm7, %ymm7 

// CHECK: vpblendw $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x0e,0x0a,0x00]     
vpblendw $0, (%rdx), %ymm9, %ymm9 

// CHECK: vpblendw $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x0e,0xff,0x00]     
vpblendw $0, %ymm7, %ymm7, %ymm7 

// CHECK: vpblendw $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x0e,0xc9,0x00]     
vpblendw $0, %ymm9, %ymm9, %ymm9 

// CHECK: vpbroadcastb 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x78,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096, %xmm15 

// CHECK: vpbroadcastb 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096, %xmm6 

// CHECK: vpbroadcastb 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096, %ymm7 

// CHECK: vpbroadcastb 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x78,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastb 485498096, %ymm9 

// CHECK: vpbroadcastb -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x78,0x7c,0x82,0xc0]       
vpbroadcastb -64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastb 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x78,0x7c,0x82,0x40]       
vpbroadcastb 64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastb -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x74,0x82,0xc0]       
vpbroadcastb -64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastb 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x74,0x82,0x40]       
vpbroadcastb 64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastb -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x7c,0x82,0xc0]       
vpbroadcastb -64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastb 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x7c,0x82,0x40]       
vpbroadcastb 64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastb -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x78,0x4c,0x82,0xc0]       
vpbroadcastb -64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastb 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x78,0x4c,0x82,0x40]       
vpbroadcastb 64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastb 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x78,0x7c,0x02,0x40]       
vpbroadcastb 64(%rdx,%rax), %xmm15 

// CHECK: vpbroadcastb 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x74,0x02,0x40]       
vpbroadcastb 64(%rdx,%rax), %xmm6 

// CHECK: vpbroadcastb 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x7c,0x02,0x40]       
vpbroadcastb 64(%rdx,%rax), %ymm7 

// CHECK: vpbroadcastb 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x78,0x4c,0x02,0x40]       
vpbroadcastb 64(%rdx,%rax), %ymm9 

// CHECK: vpbroadcastb 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x78,0x7a,0x40]       
vpbroadcastb 64(%rdx), %xmm15 

// CHECK: vpbroadcastb 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x72,0x40]       
vpbroadcastb 64(%rdx), %xmm6 

// CHECK: vpbroadcastb 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x7a,0x40]       
vpbroadcastb 64(%rdx), %ymm7 

// CHECK: vpbroadcastb 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x78,0x4a,0x40]       
vpbroadcastb 64(%rdx), %ymm9 

// CHECK: vpbroadcastb (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x78,0x3a]       
vpbroadcastb (%rdx), %xmm15 

// CHECK: vpbroadcastb (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0x32]       
vpbroadcastb (%rdx), %xmm6 

// CHECK: vpbroadcastb (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0x3a]       
vpbroadcastb (%rdx), %ymm7 

// CHECK: vpbroadcastb (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x78,0x0a]       
vpbroadcastb (%rdx), %ymm9 

// CHECK: vpbroadcastb %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x78,0xff]       
vpbroadcastb %xmm15, %xmm15 

// CHECK: vpbroadcastb %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x78,0xcf]       
vpbroadcastb %xmm15, %ymm9 

// CHECK: vpbroadcastb %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x78,0xf6]       
vpbroadcastb %xmm6, %xmm6 

// CHECK: vpbroadcastb %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x78,0xfe]       
vpbroadcastb %xmm6, %ymm7 

// CHECK: vpbroadcastd 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096, %xmm15 

// CHECK: vpbroadcastd 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096, %xmm6 

// CHECK: vpbroadcastd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096, %ymm7 

// CHECK: vpbroadcastd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastd 485498096, %ymm9 

// CHECK: vpbroadcastd -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x58,0x7c,0x82,0xc0]       
vpbroadcastd -64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastd 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x58,0x7c,0x82,0x40]       
vpbroadcastd 64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastd -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x74,0x82,0xc0]       
vpbroadcastd -64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastd 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x74,0x82,0x40]       
vpbroadcastd 64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x7c,0x82,0xc0]       
vpbroadcastd -64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x7c,0x82,0x40]       
vpbroadcastd 64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x58,0x4c,0x82,0xc0]       
vpbroadcastd -64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x58,0x4c,0x82,0x40]       
vpbroadcastd 64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastd 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x58,0x7c,0x02,0x40]       
vpbroadcastd 64(%rdx,%rax), %xmm15 

// CHECK: vpbroadcastd 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x74,0x02,0x40]       
vpbroadcastd 64(%rdx,%rax), %xmm6 

// CHECK: vpbroadcastd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x7c,0x02,0x40]       
vpbroadcastd 64(%rdx,%rax), %ymm7 

// CHECK: vpbroadcastd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x58,0x4c,0x02,0x40]       
vpbroadcastd 64(%rdx,%rax), %ymm9 

// CHECK: vpbroadcastd 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x58,0x7a,0x40]       
vpbroadcastd 64(%rdx), %xmm15 

// CHECK: vpbroadcastd 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x72,0x40]       
vpbroadcastd 64(%rdx), %xmm6 

// CHECK: vpbroadcastd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x7a,0x40]       
vpbroadcastd 64(%rdx), %ymm7 

// CHECK: vpbroadcastd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x58,0x4a,0x40]       
vpbroadcastd 64(%rdx), %ymm9 

// CHECK: vpbroadcastd (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x58,0x3a]       
vpbroadcastd (%rdx), %xmm15 

// CHECK: vpbroadcastd (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0x32]       
vpbroadcastd (%rdx), %xmm6 

// CHECK: vpbroadcastd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0x3a]       
vpbroadcastd (%rdx), %ymm7 

// CHECK: vpbroadcastd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x58,0x0a]       
vpbroadcastd (%rdx), %ymm9 

// CHECK: vpbroadcastd %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x58,0xff]       
vpbroadcastd %xmm15, %xmm15 

// CHECK: vpbroadcastd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x58,0xcf]       
vpbroadcastd %xmm15, %ymm9 

// CHECK: vpbroadcastd %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x58,0xf6]       
vpbroadcastd %xmm6, %xmm6 

// CHECK: vpbroadcastd %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x58,0xfe]       
vpbroadcastd %xmm6, %ymm7 

// CHECK: vpbroadcastq 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096, %xmm15 

// CHECK: vpbroadcastq 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096, %xmm6 

// CHECK: vpbroadcastq 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096, %ymm7 

// CHECK: vpbroadcastq 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastq 485498096, %ymm9 

// CHECK: vpbroadcastq -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x59,0x7c,0x82,0xc0]       
vpbroadcastq -64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastq 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x59,0x7c,0x82,0x40]       
vpbroadcastq 64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastq -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x74,0x82,0xc0]       
vpbroadcastq -64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastq 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x74,0x82,0x40]       
vpbroadcastq 64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x7c,0x82,0xc0]       
vpbroadcastq -64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x7c,0x82,0x40]       
vpbroadcastq 64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x59,0x4c,0x82,0xc0]       
vpbroadcastq -64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x59,0x4c,0x82,0x40]       
vpbroadcastq 64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastq 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x59,0x7c,0x02,0x40]       
vpbroadcastq 64(%rdx,%rax), %xmm15 

// CHECK: vpbroadcastq 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x74,0x02,0x40]       
vpbroadcastq 64(%rdx,%rax), %xmm6 

// CHECK: vpbroadcastq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x7c,0x02,0x40]       
vpbroadcastq 64(%rdx,%rax), %ymm7 

// CHECK: vpbroadcastq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x59,0x4c,0x02,0x40]       
vpbroadcastq 64(%rdx,%rax), %ymm9 

// CHECK: vpbroadcastq 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x59,0x7a,0x40]       
vpbroadcastq 64(%rdx), %xmm15 

// CHECK: vpbroadcastq 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x72,0x40]       
vpbroadcastq 64(%rdx), %xmm6 

// CHECK: vpbroadcastq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x7a,0x40]       
vpbroadcastq 64(%rdx), %ymm7 

// CHECK: vpbroadcastq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x59,0x4a,0x40]       
vpbroadcastq 64(%rdx), %ymm9 

// CHECK: vpbroadcastq (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x59,0x3a]       
vpbroadcastq (%rdx), %xmm15 

// CHECK: vpbroadcastq (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0x32]       
vpbroadcastq (%rdx), %xmm6 

// CHECK: vpbroadcastq (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0x3a]       
vpbroadcastq (%rdx), %ymm7 

// CHECK: vpbroadcastq (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x59,0x0a]       
vpbroadcastq (%rdx), %ymm9 

// CHECK: vpbroadcastq %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x59,0xff]       
vpbroadcastq %xmm15, %xmm15 

// CHECK: vpbroadcastq %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x59,0xcf]       
vpbroadcastq %xmm15, %ymm9 

// CHECK: vpbroadcastq %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x59,0xf6]       
vpbroadcastq %xmm6, %xmm6 

// CHECK: vpbroadcastq %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x59,0xfe]       
vpbroadcastq %xmm6, %ymm7 

// CHECK: vpbroadcastw 485498096, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x79,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096, %xmm15 

// CHECK: vpbroadcastw 485498096, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x34,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096, %xmm6 

// CHECK: vpbroadcastw 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096, %ymm7 

// CHECK: vpbroadcastw 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x79,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpbroadcastw 485498096, %ymm9 

// CHECK: vpbroadcastw -64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x79,0x7c,0x82,0xc0]       
vpbroadcastw -64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastw 64(%rdx,%rax,4), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x79,0x7c,0x82,0x40]       
vpbroadcastw 64(%rdx,%rax,4), %xmm15 

// CHECK: vpbroadcastw -64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x74,0x82,0xc0]       
vpbroadcastw -64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastw 64(%rdx,%rax,4), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x74,0x82,0x40]       
vpbroadcastw 64(%rdx,%rax,4), %xmm6 

// CHECK: vpbroadcastw -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x7c,0x82,0xc0]       
vpbroadcastw -64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastw 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x7c,0x82,0x40]       
vpbroadcastw 64(%rdx,%rax,4), %ymm7 

// CHECK: vpbroadcastw -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x79,0x4c,0x82,0xc0]       
vpbroadcastw -64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastw 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x79,0x4c,0x82,0x40]       
vpbroadcastw 64(%rdx,%rax,4), %ymm9 

// CHECK: vpbroadcastw 64(%rdx,%rax), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x79,0x7c,0x02,0x40]       
vpbroadcastw 64(%rdx,%rax), %xmm15 

// CHECK: vpbroadcastw 64(%rdx,%rax), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x74,0x02,0x40]       
vpbroadcastw 64(%rdx,%rax), %xmm6 

// CHECK: vpbroadcastw 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x7c,0x02,0x40]       
vpbroadcastw 64(%rdx,%rax), %ymm7 

// CHECK: vpbroadcastw 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x79,0x4c,0x02,0x40]       
vpbroadcastw 64(%rdx,%rax), %ymm9 

// CHECK: vpbroadcastw 64(%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x79,0x7a,0x40]       
vpbroadcastw 64(%rdx), %xmm15 

// CHECK: vpbroadcastw 64(%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x72,0x40]       
vpbroadcastw 64(%rdx), %xmm6 

// CHECK: vpbroadcastw 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x7a,0x40]       
vpbroadcastw 64(%rdx), %ymm7 

// CHECK: vpbroadcastw 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x79,0x4a,0x40]       
vpbroadcastw 64(%rdx), %ymm9 

// CHECK: vpbroadcastw (%rdx), %xmm15 
// CHECK: encoding: [0xc4,0x62,0x79,0x79,0x3a]       
vpbroadcastw (%rdx), %xmm15 

// CHECK: vpbroadcastw (%rdx), %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0x32]       
vpbroadcastw (%rdx), %xmm6 

// CHECK: vpbroadcastw (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0x3a]       
vpbroadcastw (%rdx), %ymm7 

// CHECK: vpbroadcastw (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x79,0x0a]       
vpbroadcastw (%rdx), %ymm9 

// CHECK: vpbroadcastw %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x79,0x79,0xff]       
vpbroadcastw %xmm15, %xmm15 

// CHECK: vpbroadcastw %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x79,0xcf]       
vpbroadcastw %xmm15, %ymm9 

// CHECK: vpbroadcastw %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x79,0x79,0xf6]       
vpbroadcastw %xmm6, %xmm6 

// CHECK: vpbroadcastw %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x79,0xfe]       
vpbroadcastw %xmm6, %ymm7 

// CHECK: vpcmpeqb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x74,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096, %ymm7, %ymm7 

// CHECK: vpcmpeqb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x74,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqb 485498096, %ymm9, %ymm9 

// CHECK: vpcmpeqb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x74,0x7c,0x82,0xc0]      
vpcmpeqb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x74,0x7c,0x82,0x40]      
vpcmpeqb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x74,0x4c,0x82,0xc0]      
vpcmpeqb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x74,0x4c,0x82,0x40]      
vpcmpeqb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x74,0x7c,0x02,0x40]      
vpcmpeqb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpeqb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x74,0x4c,0x02,0x40]      
vpcmpeqb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpeqb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x74,0x7a,0x40]      
vpcmpeqb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x74,0x4a,0x40]      
vpcmpeqb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x74,0x3a]      
vpcmpeqb (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x74,0x0a]      
vpcmpeqb (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x74,0xff]      
vpcmpeqb %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpeqb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x74,0xc9]      
vpcmpeqb %ymm9, %ymm9, %ymm9 

// CHECK: vpcmpeqd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x76,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096, %ymm7, %ymm7 

// CHECK: vpcmpeqd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x76,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqd 485498096, %ymm9, %ymm9 

// CHECK: vpcmpeqd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x76,0x7c,0x82,0xc0]      
vpcmpeqd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x76,0x7c,0x82,0x40]      
vpcmpeqd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x76,0x4c,0x82,0xc0]      
vpcmpeqd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x76,0x4c,0x82,0x40]      
vpcmpeqd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x76,0x7c,0x02,0x40]      
vpcmpeqd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpeqd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x76,0x4c,0x02,0x40]      
vpcmpeqd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpeqd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x76,0x7a,0x40]      
vpcmpeqd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x76,0x4a,0x40]      
vpcmpeqd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x76,0x3a]      
vpcmpeqd (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x76,0x0a]      
vpcmpeqd (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x76,0xff]      
vpcmpeqd %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpeqd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x76,0xc9]      
vpcmpeqd %ymm9, %ymm9, %ymm9 

// CHECK: vpcmpeqq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x29,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096, %ymm7, %ymm7 

// CHECK: vpcmpeqq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x29,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqq 485498096, %ymm9, %ymm9 

// CHECK: vpcmpeqq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x29,0x7c,0x82,0xc0]      
vpcmpeqq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x29,0x7c,0x82,0x40]      
vpcmpeqq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x29,0x4c,0x82,0xc0]      
vpcmpeqq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x29,0x4c,0x82,0x40]      
vpcmpeqq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x29,0x7c,0x02,0x40]      
vpcmpeqq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpeqq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x29,0x4c,0x02,0x40]      
vpcmpeqq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpeqq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x29,0x7a,0x40]      
vpcmpeqq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x29,0x4a,0x40]      
vpcmpeqq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x29,0x3a]      
vpcmpeqq (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x29,0x0a]      
vpcmpeqq (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x29,0xff]      
vpcmpeqq %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpeqq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x29,0xc9]      
vpcmpeqq %ymm9, %ymm9, %ymm9 

// CHECK: vpcmpeqw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x75,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096, %ymm7, %ymm7 

// CHECK: vpcmpeqw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x75,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpeqw 485498096, %ymm9, %ymm9 

// CHECK: vpcmpeqw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x75,0x7c,0x82,0xc0]      
vpcmpeqw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x75,0x7c,0x82,0x40]      
vpcmpeqw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpeqw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x75,0x4c,0x82,0xc0]      
vpcmpeqw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x75,0x4c,0x82,0x40]      
vpcmpeqw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpeqw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x75,0x7c,0x02,0x40]      
vpcmpeqw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpeqw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x75,0x4c,0x02,0x40]      
vpcmpeqw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpeqw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x75,0x7a,0x40]      
vpcmpeqw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x75,0x4a,0x40]      
vpcmpeqw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x75,0x3a]      
vpcmpeqw (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpeqw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x75,0x0a]      
vpcmpeqw (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpeqw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x75,0xff]      
vpcmpeqw %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpeqw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x75,0xc9]      
vpcmpeqw %ymm9, %ymm9, %ymm9 

// CHECK: vpcmpgtb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x64,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096, %ymm7, %ymm7 

// CHECK: vpcmpgtb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x64,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtb 485498096, %ymm9, %ymm9 

// CHECK: vpcmpgtb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x64,0x7c,0x82,0xc0]      
vpcmpgtb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x64,0x7c,0x82,0x40]      
vpcmpgtb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x64,0x4c,0x82,0xc0]      
vpcmpgtb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x64,0x4c,0x82,0x40]      
vpcmpgtb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x64,0x7c,0x02,0x40]      
vpcmpgtb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpgtb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x64,0x4c,0x02,0x40]      
vpcmpgtb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpgtb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x64,0x7a,0x40]      
vpcmpgtb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x64,0x4a,0x40]      
vpcmpgtb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x64,0x3a]      
vpcmpgtb (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x64,0x0a]      
vpcmpgtb (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x64,0xff]      
vpcmpgtb %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpgtb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x64,0xc9]      
vpcmpgtb %ymm9, %ymm9, %ymm9 

// CHECK: vpcmpgtd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x66,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096, %ymm7, %ymm7 

// CHECK: vpcmpgtd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x66,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtd 485498096, %ymm9, %ymm9 

// CHECK: vpcmpgtd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x66,0x7c,0x82,0xc0]      
vpcmpgtd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x66,0x7c,0x82,0x40]      
vpcmpgtd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x66,0x4c,0x82,0xc0]      
vpcmpgtd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x66,0x4c,0x82,0x40]      
vpcmpgtd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x66,0x7c,0x02,0x40]      
vpcmpgtd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpgtd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x66,0x4c,0x02,0x40]      
vpcmpgtd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpgtd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x66,0x7a,0x40]      
vpcmpgtd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x66,0x4a,0x40]      
vpcmpgtd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x66,0x3a]      
vpcmpgtd (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x66,0x0a]      
vpcmpgtd (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x66,0xff]      
vpcmpgtd %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpgtd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x66,0xc9]      
vpcmpgtd %ymm9, %ymm9, %ymm9 

// CHECK: vpcmpgtq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x37,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096, %ymm7, %ymm7 

// CHECK: vpcmpgtq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x37,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtq 485498096, %ymm9, %ymm9 

// CHECK: vpcmpgtq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x37,0x7c,0x82,0xc0]      
vpcmpgtq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x37,0x7c,0x82,0x40]      
vpcmpgtq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x37,0x4c,0x82,0xc0]      
vpcmpgtq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x37,0x4c,0x82,0x40]      
vpcmpgtq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x37,0x7c,0x02,0x40]      
vpcmpgtq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpgtq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x37,0x4c,0x02,0x40]      
vpcmpgtq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpgtq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x37,0x7a,0x40]      
vpcmpgtq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x37,0x4a,0x40]      
vpcmpgtq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x37,0x3a]      
vpcmpgtq (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x37,0x0a]      
vpcmpgtq (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x37,0xff]      
vpcmpgtq %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpgtq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x37,0xc9]      
vpcmpgtq %ymm9, %ymm9, %ymm9 

// CHECK: vpcmpgtw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x65,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096, %ymm7, %ymm7 

// CHECK: vpcmpgtw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x65,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpcmpgtw 485498096, %ymm9, %ymm9 

// CHECK: vpcmpgtw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x65,0x7c,0x82,0xc0]      
vpcmpgtw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x65,0x7c,0x82,0x40]      
vpcmpgtw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpcmpgtw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x65,0x4c,0x82,0xc0]      
vpcmpgtw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x65,0x4c,0x82,0x40]      
vpcmpgtw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpcmpgtw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x65,0x7c,0x02,0x40]      
vpcmpgtw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpcmpgtw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x65,0x4c,0x02,0x40]      
vpcmpgtw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpcmpgtw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x65,0x7a,0x40]      
vpcmpgtw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x65,0x4a,0x40]      
vpcmpgtw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x65,0x3a]      
vpcmpgtw (%rdx), %ymm7, %ymm7 

// CHECK: vpcmpgtw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x65,0x0a]      
vpcmpgtw (%rdx), %ymm9, %ymm9 

// CHECK: vpcmpgtw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x65,0xff]      
vpcmpgtw %ymm7, %ymm7, %ymm7 

// CHECK: vpcmpgtw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x65,0xc9]      
vpcmpgtw %ymm9, %ymm9, %ymm9 

// CHECK: vperm2i128 $0, 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x46,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2i128 $0, 485498096, %ymm7, %ymm7 

// CHECK: vperm2i128 $0, 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x46,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]     
vperm2i128 $0, 485498096, %ymm9, %ymm9 

// CHECK: vperm2i128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x46,0x7c,0x82,0xc0,0x00]     
vperm2i128 $0, -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vperm2i128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x46,0x7c,0x82,0x40,0x00]     
vperm2i128 $0, 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vperm2i128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x46,0x4c,0x82,0xc0,0x00]     
vperm2i128 $0, -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vperm2i128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x46,0x4c,0x82,0x40,0x00]     
vperm2i128 $0, 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vperm2i128 $0, 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x46,0x7c,0x02,0x40,0x00]     
vperm2i128 $0, 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vperm2i128 $0, 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x46,0x4c,0x02,0x40,0x00]     
vperm2i128 $0, 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vperm2i128 $0, 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x46,0x7a,0x40,0x00]     
vperm2i128 $0, 64(%rdx), %ymm7, %ymm7 

// CHECK: vperm2i128 $0, 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x46,0x4a,0x40,0x00]     
vperm2i128 $0, 64(%rdx), %ymm9, %ymm9 

// CHECK: vperm2i128 $0, (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x46,0x3a,0x00]     
vperm2i128 $0, (%rdx), %ymm7, %ymm7 

// CHECK: vperm2i128 $0, (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x63,0x35,0x46,0x0a,0x00]     
vperm2i128 $0, (%rdx), %ymm9, %ymm9 

// CHECK: vperm2i128 $0, %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0x45,0x46,0xff,0x00]     
vperm2i128 $0, %ymm7, %ymm7, %ymm7 

// CHECK: vperm2i128 $0, %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0x35,0x46,0xc9,0x00]     
vperm2i128 $0, %ymm9, %ymm9, %ymm9 

// CHECK: vpermd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x36,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermd 485498096, %ymm7, %ymm7 

// CHECK: vpermd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x36,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermd 485498096, %ymm9, %ymm9 

// CHECK: vpermd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x36,0x7c,0x82,0xc0]      
vpermd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x36,0x7c,0x82,0x40]      
vpermd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x36,0x4c,0x82,0xc0]      
vpermd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x36,0x4c,0x82,0x40]      
vpermd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x36,0x7c,0x02,0x40]      
vpermd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpermd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x36,0x4c,0x02,0x40]      
vpermd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpermd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x36,0x7a,0x40]      
vpermd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpermd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x36,0x4a,0x40]      
vpermd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpermd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x36,0x3a]      
vpermd (%rdx), %ymm7, %ymm7 

// CHECK: vpermd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x36,0x0a]      
vpermd (%rdx), %ymm9, %ymm9 

// CHECK: vpermd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x36,0xff]      
vpermd %ymm7, %ymm7, %ymm7 

// CHECK: vpermd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x36,0xc9]      
vpermd %ymm9, %ymm9, %ymm9 

// CHECK: vpermpd $0, 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermpd $0, 485498096, %ymm7 

// CHECK: vpermpd $0, 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x01,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermpd $0, 485498096, %ymm9 

// CHECK: vpermpd $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x7c,0x82,0xc0,0x00]      
vpermpd $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vpermpd $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x7c,0x82,0x40,0x00]      
vpermpd $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vpermpd $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x01,0x4c,0x82,0xc0,0x00]      
vpermpd $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vpermpd $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x01,0x4c,0x82,0x40,0x00]      
vpermpd $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vpermpd $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x7c,0x02,0x40,0x00]      
vpermpd $0, 64(%rdx,%rax), %ymm7 

// CHECK: vpermpd $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x01,0x4c,0x02,0x40,0x00]      
vpermpd $0, 64(%rdx,%rax), %ymm9 

// CHECK: vpermpd $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x7a,0x40,0x00]      
vpermpd $0, 64(%rdx), %ymm7 

// CHECK: vpermpd $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x01,0x4a,0x40,0x00]      
vpermpd $0, 64(%rdx), %ymm9 

// CHECK: vpermpd $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0x3a,0x00]      
vpermpd $0, (%rdx), %ymm7 

// CHECK: vpermpd $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x01,0x0a,0x00]      
vpermpd $0, (%rdx), %ymm9 

// CHECK: vpermpd $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x01,0xff,0x00]      
vpermpd $0, %ymm7, %ymm7 

// CHECK: vpermpd $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0xfd,0x01,0xc9,0x00]      
vpermpd $0, %ymm9, %ymm9 

// CHECK: vpermps 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x16,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermps 485498096, %ymm7, %ymm7 

// CHECK: vpermps 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x16,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpermps 485498096, %ymm9, %ymm9 

// CHECK: vpermps -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x16,0x7c,0x82,0xc0]      
vpermps -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermps 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x16,0x7c,0x82,0x40]      
vpermps 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpermps -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x16,0x4c,0x82,0xc0]      
vpermps -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermps 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x16,0x4c,0x82,0x40]      
vpermps 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpermps 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x16,0x7c,0x02,0x40]      
vpermps 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpermps 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x16,0x4c,0x02,0x40]      
vpermps 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpermps 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x16,0x7a,0x40]      
vpermps 64(%rdx), %ymm7, %ymm7 

// CHECK: vpermps 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x16,0x4a,0x40]      
vpermps 64(%rdx), %ymm9, %ymm9 

// CHECK: vpermps (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x16,0x3a]      
vpermps (%rdx), %ymm7, %ymm7 

// CHECK: vpermps (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x16,0x0a]      
vpermps (%rdx), %ymm9, %ymm9 

// CHECK: vpermps %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x16,0xff]      
vpermps %ymm7, %ymm7, %ymm7 

// CHECK: vpermps %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x16,0xc9]      
vpermps %ymm9, %ymm9, %ymm9 

// CHECK: vpermq $0, 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermq $0, 485498096, %ymm7 

// CHECK: vpermq $0, 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x00,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpermq $0, 485498096, %ymm9 

// CHECK: vpermq $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x7c,0x82,0xc0,0x00]      
vpermq $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vpermq $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x7c,0x82,0x40,0x00]      
vpermq $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vpermq $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x00,0x4c,0x82,0xc0,0x00]      
vpermq $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vpermq $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x00,0x4c,0x82,0x40,0x00]      
vpermq $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vpermq $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x7c,0x02,0x40,0x00]      
vpermq $0, 64(%rdx,%rax), %ymm7 

// CHECK: vpermq $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x00,0x4c,0x02,0x40,0x00]      
vpermq $0, 64(%rdx,%rax), %ymm9 

// CHECK: vpermq $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x7a,0x40,0x00]      
vpermq $0, 64(%rdx), %ymm7 

// CHECK: vpermq $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x00,0x4a,0x40,0x00]      
vpermq $0, 64(%rdx), %ymm9 

// CHECK: vpermq $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0x3a,0x00]      
vpermq $0, (%rdx), %ymm7 

// CHECK: vpermq $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x63,0xfd,0x00,0x0a,0x00]      
vpermq $0, (%rdx), %ymm9 

// CHECK: vpermq $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe3,0xfd,0x00,0xff,0x00]      
vpermq $0, %ymm7, %ymm7 

// CHECK: vpermq $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x43,0xfd,0x00,0xc9,0x00]      
vpermq $0, %ymm9, %ymm9 

// CHECK: vphaddd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x02,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096, %ymm7, %ymm7 

// CHECK: vphaddd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x02,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddd 485498096, %ymm9, %ymm9 

// CHECK: vphaddd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x02,0x7c,0x82,0xc0]      
vphaddd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphaddd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x02,0x7c,0x82,0x40]      
vphaddd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphaddd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x02,0x4c,0x82,0xc0]      
vphaddd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphaddd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x02,0x4c,0x82,0x40]      
vphaddd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphaddd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x02,0x7c,0x02,0x40]      
vphaddd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vphaddd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x02,0x4c,0x02,0x40]      
vphaddd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vphaddd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x02,0x7a,0x40]      
vphaddd 64(%rdx), %ymm7, %ymm7 

// CHECK: vphaddd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x02,0x4a,0x40]      
vphaddd 64(%rdx), %ymm9, %ymm9 

// CHECK: vphaddd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x02,0x3a]      
vphaddd (%rdx), %ymm7, %ymm7 

// CHECK: vphaddd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x02,0x0a]      
vphaddd (%rdx), %ymm9, %ymm9 

// CHECK: vphaddd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x02,0xff]      
vphaddd %ymm7, %ymm7, %ymm7 

// CHECK: vphaddd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x02,0xc9]      
vphaddd %ymm9, %ymm9, %ymm9 

// CHECK: vphaddsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x03,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096, %ymm7, %ymm7 

// CHECK: vphaddsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x03,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddsw 485498096, %ymm9, %ymm9 

// CHECK: vphaddsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x03,0x7c,0x82,0xc0]      
vphaddsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphaddsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x03,0x7c,0x82,0x40]      
vphaddsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphaddsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x03,0x4c,0x82,0xc0]      
vphaddsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphaddsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x03,0x4c,0x82,0x40]      
vphaddsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphaddsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x03,0x7c,0x02,0x40]      
vphaddsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vphaddsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x03,0x4c,0x02,0x40]      
vphaddsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vphaddsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x03,0x7a,0x40]      
vphaddsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vphaddsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x03,0x4a,0x40]      
vphaddsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vphaddsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x03,0x3a]      
vphaddsw (%rdx), %ymm7, %ymm7 

// CHECK: vphaddsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x03,0x0a]      
vphaddsw (%rdx), %ymm9, %ymm9 

// CHECK: vphaddsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x03,0xff]      
vphaddsw %ymm7, %ymm7, %ymm7 

// CHECK: vphaddsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x03,0xc9]      
vphaddsw %ymm9, %ymm9, %ymm9 

// CHECK: vphaddw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x01,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096, %ymm7, %ymm7 

// CHECK: vphaddw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x01,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphaddw 485498096, %ymm9, %ymm9 

// CHECK: vphaddw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x01,0x7c,0x82,0xc0]      
vphaddw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphaddw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x01,0x7c,0x82,0x40]      
vphaddw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphaddw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x01,0x4c,0x82,0xc0]      
vphaddw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphaddw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x01,0x4c,0x82,0x40]      
vphaddw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphaddw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x01,0x7c,0x02,0x40]      
vphaddw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vphaddw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x01,0x4c,0x02,0x40]      
vphaddw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vphaddw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x01,0x7a,0x40]      
vphaddw 64(%rdx), %ymm7, %ymm7 

// CHECK: vphaddw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x01,0x4a,0x40]      
vphaddw 64(%rdx), %ymm9, %ymm9 

// CHECK: vphaddw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x01,0x3a]      
vphaddw (%rdx), %ymm7, %ymm7 

// CHECK: vphaddw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x01,0x0a]      
vphaddw (%rdx), %ymm9, %ymm9 

// CHECK: vphaddw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x01,0xff]      
vphaddw %ymm7, %ymm7, %ymm7 

// CHECK: vphaddw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x01,0xc9]      
vphaddw %ymm9, %ymm9, %ymm9 

// CHECK: vphsubd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x06,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096, %ymm7, %ymm7 

// CHECK: vphsubd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x06,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubd 485498096, %ymm9, %ymm9 

// CHECK: vphsubd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x06,0x7c,0x82,0xc0]      
vphsubd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphsubd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x06,0x7c,0x82,0x40]      
vphsubd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphsubd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x06,0x4c,0x82,0xc0]      
vphsubd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphsubd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x06,0x4c,0x82,0x40]      
vphsubd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphsubd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x06,0x7c,0x02,0x40]      
vphsubd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vphsubd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x06,0x4c,0x02,0x40]      
vphsubd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vphsubd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x06,0x7a,0x40]      
vphsubd 64(%rdx), %ymm7, %ymm7 

// CHECK: vphsubd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x06,0x4a,0x40]      
vphsubd 64(%rdx), %ymm9, %ymm9 

// CHECK: vphsubd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x06,0x3a]      
vphsubd (%rdx), %ymm7, %ymm7 

// CHECK: vphsubd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x06,0x0a]      
vphsubd (%rdx), %ymm9, %ymm9 

// CHECK: vphsubd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x06,0xff]      
vphsubd %ymm7, %ymm7, %ymm7 

// CHECK: vphsubd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x06,0xc9]      
vphsubd %ymm9, %ymm9, %ymm9 

// CHECK: vphsubsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x07,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096, %ymm7, %ymm7 

// CHECK: vphsubsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x07,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubsw 485498096, %ymm9, %ymm9 

// CHECK: vphsubsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x07,0x7c,0x82,0xc0]      
vphsubsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphsubsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x07,0x7c,0x82,0x40]      
vphsubsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphsubsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x07,0x4c,0x82,0xc0]      
vphsubsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphsubsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x07,0x4c,0x82,0x40]      
vphsubsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphsubsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x07,0x7c,0x02,0x40]      
vphsubsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vphsubsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x07,0x4c,0x02,0x40]      
vphsubsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vphsubsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x07,0x7a,0x40]      
vphsubsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vphsubsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x07,0x4a,0x40]      
vphsubsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vphsubsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x07,0x3a]      
vphsubsw (%rdx), %ymm7, %ymm7 

// CHECK: vphsubsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x07,0x0a]      
vphsubsw (%rdx), %ymm9, %ymm9 

// CHECK: vphsubsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x07,0xff]      
vphsubsw %ymm7, %ymm7, %ymm7 

// CHECK: vphsubsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x07,0xc9]      
vphsubsw %ymm9, %ymm9, %ymm9 

// CHECK: vphsubw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x05,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096, %ymm7, %ymm7 

// CHECK: vphsubw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x05,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vphsubw 485498096, %ymm9, %ymm9 

// CHECK: vphsubw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x05,0x7c,0x82,0xc0]      
vphsubw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphsubw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x05,0x7c,0x82,0x40]      
vphsubw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vphsubw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x05,0x4c,0x82,0xc0]      
vphsubw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphsubw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x05,0x4c,0x82,0x40]      
vphsubw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vphsubw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x05,0x7c,0x02,0x40]      
vphsubw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vphsubw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x05,0x4c,0x02,0x40]      
vphsubw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vphsubw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x05,0x7a,0x40]      
vphsubw 64(%rdx), %ymm7, %ymm7 

// CHECK: vphsubw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x05,0x4a,0x40]      
vphsubw 64(%rdx), %ymm9, %ymm9 

// CHECK: vphsubw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x05,0x3a]      
vphsubw (%rdx), %ymm7, %ymm7 

// CHECK: vphsubw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x05,0x0a]      
vphsubw (%rdx), %ymm9, %ymm9 

// CHECK: vphsubw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x05,0xff]      
vphsubw %ymm7, %ymm7, %ymm7 

// CHECK: vphsubw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x05,0xc9]      
vphsubw %ymm9, %ymm9, %ymm9 

// CHECK: vpmaddubsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x04,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096, %ymm7, %ymm7 

// CHECK: vpmaddubsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x04,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddubsw 485498096, %ymm9, %ymm9 

// CHECK: vpmaddubsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x04,0x7c,0x82,0xc0]      
vpmaddubsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaddubsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x04,0x7c,0x82,0x40]      
vpmaddubsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaddubsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x04,0x4c,0x82,0xc0]      
vpmaddubsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaddubsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x04,0x4c,0x82,0x40]      
vpmaddubsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaddubsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x04,0x7c,0x02,0x40]      
vpmaddubsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaddubsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x04,0x4c,0x02,0x40]      
vpmaddubsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaddubsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x04,0x7a,0x40]      
vpmaddubsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaddubsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x04,0x4a,0x40]      
vpmaddubsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaddubsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x04,0x3a]      
vpmaddubsw (%rdx), %ymm7, %ymm7 

// CHECK: vpmaddubsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x04,0x0a]      
vpmaddubsw (%rdx), %ymm9, %ymm9 

// CHECK: vpmaddubsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x04,0xff]      
vpmaddubsw %ymm7, %ymm7, %ymm7 

// CHECK: vpmaddubsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x04,0xc9]      
vpmaddubsw %ymm9, %ymm9, %ymm9 

// CHECK: vpmaddwd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf5,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096, %ymm7, %ymm7 

// CHECK: vpmaddwd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf5,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaddwd 485498096, %ymm9, %ymm9 

// CHECK: vpmaddwd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf5,0x7c,0x82,0xc0]      
vpmaddwd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaddwd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf5,0x7c,0x82,0x40]      
vpmaddwd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaddwd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf5,0x4c,0x82,0xc0]      
vpmaddwd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaddwd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf5,0x4c,0x82,0x40]      
vpmaddwd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaddwd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf5,0x7c,0x02,0x40]      
vpmaddwd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaddwd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf5,0x4c,0x02,0x40]      
vpmaddwd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaddwd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf5,0x7a,0x40]      
vpmaddwd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaddwd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf5,0x4a,0x40]      
vpmaddwd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaddwd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf5,0x3a]      
vpmaddwd (%rdx), %ymm7, %ymm7 

// CHECK: vpmaddwd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf5,0x0a]      
vpmaddwd (%rdx), %ymm9, %ymm9 

// CHECK: vpmaddwd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf5,0xff]      
vpmaddwd %ymm7, %ymm7, %ymm7 

// CHECK: vpmaddwd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf5,0xc9]      
vpmaddwd %ymm9, %ymm9, %ymm9 

// CHECK: vpmaskmovd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x8c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096, %xmm15, %xmm15 

// CHECK: vpmaskmovd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096, %xmm6, %xmm6 

// CHECK: vpmaskmovd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096, %ymm7, %ymm7 

// CHECK: vpmaskmovd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x8c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd 485498096, %ymm9, %ymm9 

// CHECK: vpmaskmovd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x8c,0x7c,0x82,0xc0]      
vpmaskmovd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaskmovd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x8c,0x7c,0x82,0x40]      
vpmaskmovd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaskmovd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8c,0x74,0x82,0xc0]      
vpmaskmovd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaskmovd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8c,0x74,0x82,0x40]      
vpmaskmovd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaskmovd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8c,0x7c,0x82,0xc0]      
vpmaskmovd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaskmovd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8c,0x7c,0x82,0x40]      
vpmaskmovd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaskmovd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x8c,0x4c,0x82,0xc0]      
vpmaskmovd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaskmovd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x8c,0x4c,0x82,0x40]      
vpmaskmovd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaskmovd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x8c,0x7c,0x02,0x40]      
vpmaskmovd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaskmovd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8c,0x74,0x02,0x40]      
vpmaskmovd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaskmovd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8c,0x7c,0x02,0x40]      
vpmaskmovd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaskmovd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x8c,0x4c,0x02,0x40]      
vpmaskmovd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaskmovd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x8c,0x7a,0x40]      
vpmaskmovd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaskmovd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8c,0x72,0x40]      
vpmaskmovd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaskmovd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8c,0x7a,0x40]      
vpmaskmovd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaskmovd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x8c,0x4a,0x40]      
vpmaskmovd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaskmovd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x8c,0x3a]      
vpmaskmovd (%rdx), %xmm15, %xmm15 

// CHECK: vpmaskmovd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8c,0x32]      
vpmaskmovd (%rdx), %xmm6, %xmm6 

// CHECK: vpmaskmovd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8c,0x3a]      
vpmaskmovd (%rdx), %ymm7, %ymm7 

// CHECK: vpmaskmovd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x8c,0x0a]      
vpmaskmovd (%rdx), %ymm9, %ymm9 

// CHECK: vpmaskmovd %xmm15, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x62,0x01,0x8e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %xmm15, %xmm15, 485498096 

// CHECK: vpmaskmovd %xmm15, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0x01,0x8e,0x7a,0x40]      
vpmaskmovd %xmm15, %xmm15, 64(%rdx) 

// CHECK: vpmaskmovd %xmm15, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0x01,0x8e,0x7c,0x02,0x40]      
vpmaskmovd %xmm15, %xmm15, 64(%rdx,%rax) 

// CHECK: vpmaskmovd %xmm15, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x01,0x8e,0x7c,0x82,0xc0]      
vpmaskmovd %xmm15, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %xmm15, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x01,0x8e,0x7c,0x82,0x40]      
vpmaskmovd %xmm15, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %xmm15, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0x01,0x8e,0x3a]      
vpmaskmovd %xmm15, %xmm15, (%rdx) 

// CHECK: vpmaskmovd %xmm6, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %xmm6, %xmm6, 485498096 

// CHECK: vpmaskmovd %xmm6, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8e,0x72,0x40]      
vpmaskmovd %xmm6, %xmm6, 64(%rdx) 

// CHECK: vpmaskmovd %xmm6, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8e,0x74,0x02,0x40]      
vpmaskmovd %xmm6, %xmm6, 64(%rdx,%rax) 

// CHECK: vpmaskmovd %xmm6, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8e,0x74,0x82,0xc0]      
vpmaskmovd %xmm6, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %xmm6, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8e,0x74,0x82,0x40]      
vpmaskmovd %xmm6, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %xmm6, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x49,0x8e,0x32]      
vpmaskmovd %xmm6, %xmm6, (%rdx) 

// CHECK: vpmaskmovd %ymm7, %ymm7, 485498096 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %ymm7, %ymm7, 485498096 

// CHECK: vpmaskmovd %ymm7, %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8e,0x7a,0x40]      
vpmaskmovd %ymm7, %ymm7, 64(%rdx) 

// CHECK: vpmaskmovd %ymm7, %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8e,0x7c,0x02,0x40]      
vpmaskmovd %ymm7, %ymm7, 64(%rdx,%rax) 

// CHECK: vpmaskmovd %ymm7, %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8e,0x7c,0x82,0xc0]      
vpmaskmovd %ymm7, %ymm7, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %ymm7, %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8e,0x7c,0x82,0x40]      
vpmaskmovd %ymm7, %ymm7, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %ymm7, %ymm7, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0x45,0x8e,0x3a]      
vpmaskmovd %ymm7, %ymm7, (%rdx) 

// CHECK: vpmaskmovd %ymm9, %ymm9, 485498096 
// CHECK: encoding: [0xc4,0x62,0x35,0x8e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovd %ymm9, %ymm9, 485498096 

// CHECK: vpmaskmovd %ymm9, %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0x35,0x8e,0x4a,0x40]      
vpmaskmovd %ymm9, %ymm9, 64(%rdx) 

// CHECK: vpmaskmovd %ymm9, %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0x35,0x8e,0x4c,0x02,0x40]      
vpmaskmovd %ymm9, %ymm9, 64(%rdx,%rax) 

// CHECK: vpmaskmovd %ymm9, %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x35,0x8e,0x4c,0x82,0xc0]      
vpmaskmovd %ymm9, %ymm9, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %ymm9, %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x35,0x8e,0x4c,0x82,0x40]      
vpmaskmovd %ymm9, %ymm9, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovd %ymm9, %ymm9, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0x35,0x8e,0x0a]      
vpmaskmovd %ymm9, %ymm9, (%rdx) 

// CHECK: vpmaskmovq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x8c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096, %xmm15, %xmm15 

// CHECK: vpmaskmovq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096, %xmm6, %xmm6 

// CHECK: vpmaskmovq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096, %ymm7, %ymm7 

// CHECK: vpmaskmovq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq 485498096, %ymm9, %ymm9 

// CHECK: vpmaskmovq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x8c,0x7c,0x82,0xc0]      
vpmaskmovq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaskmovq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x8c,0x7c,0x82,0x40]      
vpmaskmovq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpmaskmovq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8c,0x74,0x82,0xc0]      
vpmaskmovq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaskmovq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8c,0x74,0x82,0x40]      
vpmaskmovq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpmaskmovq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8c,0x7c,0x82,0xc0]      
vpmaskmovq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaskmovq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8c,0x7c,0x82,0x40]      
vpmaskmovq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaskmovq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8c,0x4c,0x82,0xc0]      
vpmaskmovq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaskmovq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8c,0x4c,0x82,0x40]      
vpmaskmovq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaskmovq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x8c,0x7c,0x02,0x40]      
vpmaskmovq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpmaskmovq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8c,0x74,0x02,0x40]      
vpmaskmovq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpmaskmovq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8c,0x7c,0x02,0x40]      
vpmaskmovq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaskmovq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8c,0x4c,0x02,0x40]      
vpmaskmovq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaskmovq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x8c,0x7a,0x40]      
vpmaskmovq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpmaskmovq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8c,0x72,0x40]      
vpmaskmovq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpmaskmovq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8c,0x7a,0x40]      
vpmaskmovq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaskmovq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8c,0x4a,0x40]      
vpmaskmovq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaskmovq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x8c,0x3a]      
vpmaskmovq (%rdx), %xmm15, %xmm15 

// CHECK: vpmaskmovq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8c,0x32]      
vpmaskmovq (%rdx), %xmm6, %xmm6 

// CHECK: vpmaskmovq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8c,0x3a]      
vpmaskmovq (%rdx), %ymm7, %ymm7 

// CHECK: vpmaskmovq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8c,0x0a]      
vpmaskmovq (%rdx), %ymm9, %ymm9 

// CHECK: vpmaskmovq %xmm15, %xmm15, 485498096 
// CHECK: encoding: [0xc4,0x62,0x81,0x8e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %xmm15, %xmm15, 485498096 

// CHECK: vpmaskmovq %xmm15, %xmm15, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0x81,0x8e,0x7a,0x40]      
vpmaskmovq %xmm15, %xmm15, 64(%rdx) 

// CHECK: vpmaskmovq %xmm15, %xmm15, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0x81,0x8e,0x7c,0x02,0x40]      
vpmaskmovq %xmm15, %xmm15, 64(%rdx,%rax) 

// CHECK: vpmaskmovq %xmm15, %xmm15, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x81,0x8e,0x7c,0x82,0xc0]      
vpmaskmovq %xmm15, %xmm15, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %xmm15, %xmm15, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0x81,0x8e,0x7c,0x82,0x40]      
vpmaskmovq %xmm15, %xmm15, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %xmm15, %xmm15, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0x81,0x8e,0x3a]      
vpmaskmovq %xmm15, %xmm15, (%rdx) 

// CHECK: vpmaskmovq %xmm6, %xmm6, 485498096 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %xmm6, %xmm6, 485498096 

// CHECK: vpmaskmovq %xmm6, %xmm6, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8e,0x72,0x40]      
vpmaskmovq %xmm6, %xmm6, 64(%rdx) 

// CHECK: vpmaskmovq %xmm6, %xmm6, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8e,0x74,0x02,0x40]      
vpmaskmovq %xmm6, %xmm6, 64(%rdx,%rax) 

// CHECK: vpmaskmovq %xmm6, %xmm6, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8e,0x74,0x82,0xc0]      
vpmaskmovq %xmm6, %xmm6, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %xmm6, %xmm6, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8e,0x74,0x82,0x40]      
vpmaskmovq %xmm6, %xmm6, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %xmm6, %xmm6, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x8e,0x32]      
vpmaskmovq %xmm6, %xmm6, (%rdx) 

// CHECK: vpmaskmovq %ymm7, %ymm7, 485498096 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %ymm7, %ymm7, 485498096 

// CHECK: vpmaskmovq %ymm7, %ymm7, 64(%rdx) 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8e,0x7a,0x40]      
vpmaskmovq %ymm7, %ymm7, 64(%rdx) 

// CHECK: vpmaskmovq %ymm7, %ymm7, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8e,0x7c,0x02,0x40]      
vpmaskmovq %ymm7, %ymm7, 64(%rdx,%rax) 

// CHECK: vpmaskmovq %ymm7, %ymm7, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8e,0x7c,0x82,0xc0]      
vpmaskmovq %ymm7, %ymm7, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %ymm7, %ymm7, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8e,0x7c,0x82,0x40]      
vpmaskmovq %ymm7, %ymm7, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %ymm7, %ymm7, (%rdx) 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x8e,0x3a]      
vpmaskmovq %ymm7, %ymm7, (%rdx) 

// CHECK: vpmaskmovq %ymm9, %ymm9, 485498096 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaskmovq %ymm9, %ymm9, 485498096 

// CHECK: vpmaskmovq %ymm9, %ymm9, 64(%rdx) 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8e,0x4a,0x40]      
vpmaskmovq %ymm9, %ymm9, 64(%rdx) 

// CHECK: vpmaskmovq %ymm9, %ymm9, 64(%rdx,%rax) 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8e,0x4c,0x02,0x40]      
vpmaskmovq %ymm9, %ymm9, 64(%rdx,%rax) 

// CHECK: vpmaskmovq %ymm9, %ymm9, -64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8e,0x4c,0x82,0xc0]      
vpmaskmovq %ymm9, %ymm9, -64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %ymm9, %ymm9, 64(%rdx,%rax,4) 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8e,0x4c,0x82,0x40]      
vpmaskmovq %ymm9, %ymm9, 64(%rdx,%rax,4) 

// CHECK: vpmaskmovq %ymm9, %ymm9, (%rdx) 
// CHECK: encoding: [0xc4,0x62,0xb5,0x8e,0x0a]      
vpmaskmovq %ymm9, %ymm9, (%rdx) 

// CHECK: vpmaxsb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096, %ymm7, %ymm7 

// CHECK: vpmaxsb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsb 485498096, %ymm9, %ymm9 

// CHECK: vpmaxsb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3c,0x7c,0x82,0xc0]      
vpmaxsb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxsb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3c,0x7c,0x82,0x40]      
vpmaxsb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxsb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3c,0x4c,0x82,0xc0]      
vpmaxsb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxsb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3c,0x4c,0x82,0x40]      
vpmaxsb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxsb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3c,0x7c,0x02,0x40]      
vpmaxsb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaxsb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3c,0x4c,0x02,0x40]      
vpmaxsb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaxsb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3c,0x7a,0x40]      
vpmaxsb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaxsb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3c,0x4a,0x40]      
vpmaxsb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaxsb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3c,0x3a]      
vpmaxsb (%rdx), %ymm7, %ymm7 

// CHECK: vpmaxsb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3c,0x0a]      
vpmaxsb (%rdx), %ymm9, %ymm9 

// CHECK: vpmaxsb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3c,0xff]      
vpmaxsb %ymm7, %ymm7, %ymm7 

// CHECK: vpmaxsb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x3c,0xc9]      
vpmaxsb %ymm9, %ymm9, %ymm9 

// CHECK: vpmaxsd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096, %ymm7, %ymm7 

// CHECK: vpmaxsd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsd 485498096, %ymm9, %ymm9 

// CHECK: vpmaxsd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3d,0x7c,0x82,0xc0]      
vpmaxsd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxsd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3d,0x7c,0x82,0x40]      
vpmaxsd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxsd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3d,0x4c,0x82,0xc0]      
vpmaxsd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxsd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3d,0x4c,0x82,0x40]      
vpmaxsd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxsd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3d,0x7c,0x02,0x40]      
vpmaxsd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaxsd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3d,0x4c,0x02,0x40]      
vpmaxsd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaxsd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3d,0x7a,0x40]      
vpmaxsd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaxsd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3d,0x4a,0x40]      
vpmaxsd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaxsd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3d,0x3a]      
vpmaxsd (%rdx), %ymm7, %ymm7 

// CHECK: vpmaxsd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3d,0x0a]      
vpmaxsd (%rdx), %ymm9, %ymm9 

// CHECK: vpmaxsd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3d,0xff]      
vpmaxsd %ymm7, %ymm7, %ymm7 

// CHECK: vpmaxsd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x3d,0xc9]      
vpmaxsd %ymm9, %ymm9, %ymm9 

// CHECK: vpmaxsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xee,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096, %ymm7, %ymm7 

// CHECK: vpmaxsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xee,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxsw 485498096, %ymm9, %ymm9 

// CHECK: vpmaxsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xee,0x7c,0x82,0xc0]      
vpmaxsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xee,0x7c,0x82,0x40]      
vpmaxsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xee,0x4c,0x82,0xc0]      
vpmaxsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xee,0x4c,0x82,0x40]      
vpmaxsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xee,0x7c,0x02,0x40]      
vpmaxsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaxsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xee,0x4c,0x02,0x40]      
vpmaxsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaxsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xee,0x7a,0x40]      
vpmaxsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaxsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xee,0x4a,0x40]      
vpmaxsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaxsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xee,0x3a]      
vpmaxsw (%rdx), %ymm7, %ymm7 

// CHECK: vpmaxsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xee,0x0a]      
vpmaxsw (%rdx), %ymm9, %ymm9 

// CHECK: vpmaxsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xee,0xff]      
vpmaxsw %ymm7, %ymm7, %ymm7 

// CHECK: vpmaxsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xee,0xc9]      
vpmaxsw %ymm9, %ymm9, %ymm9 

// CHECK: vpmaxub 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xde,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096, %ymm7, %ymm7 

// CHECK: vpmaxub 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xde,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxub 485498096, %ymm9, %ymm9 

// CHECK: vpmaxub -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xde,0x7c,0x82,0xc0]      
vpmaxub -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxub 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xde,0x7c,0x82,0x40]      
vpmaxub 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxub -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xde,0x4c,0x82,0xc0]      
vpmaxub -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxub 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xde,0x4c,0x82,0x40]      
vpmaxub 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxub 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xde,0x7c,0x02,0x40]      
vpmaxub 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaxub 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xde,0x4c,0x02,0x40]      
vpmaxub 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaxub 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xde,0x7a,0x40]      
vpmaxub 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaxub 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xde,0x4a,0x40]      
vpmaxub 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaxub (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xde,0x3a]      
vpmaxub (%rdx), %ymm7, %ymm7 

// CHECK: vpmaxub (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xde,0x0a]      
vpmaxub (%rdx), %ymm9, %ymm9 

// CHECK: vpmaxub %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xde,0xff]      
vpmaxub %ymm7, %ymm7, %ymm7 

// CHECK: vpmaxub %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xde,0xc9]      
vpmaxub %ymm9, %ymm9, %ymm9 

// CHECK: vpmaxud 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096, %ymm7, %ymm7 

// CHECK: vpmaxud 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxud 485498096, %ymm9, %ymm9 

// CHECK: vpmaxud -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3f,0x7c,0x82,0xc0]      
vpmaxud -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxud 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3f,0x7c,0x82,0x40]      
vpmaxud 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxud -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3f,0x4c,0x82,0xc0]      
vpmaxud -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxud 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3f,0x4c,0x82,0x40]      
vpmaxud 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxud 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3f,0x7c,0x02,0x40]      
vpmaxud 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaxud 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3f,0x4c,0x02,0x40]      
vpmaxud 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaxud 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3f,0x7a,0x40]      
vpmaxud 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaxud 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3f,0x4a,0x40]      
vpmaxud 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaxud (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3f,0x3a]      
vpmaxud (%rdx), %ymm7, %ymm7 

// CHECK: vpmaxud (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3f,0x0a]      
vpmaxud (%rdx), %ymm9, %ymm9 

// CHECK: vpmaxud %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3f,0xff]      
vpmaxud %ymm7, %ymm7, %ymm7 

// CHECK: vpmaxud %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x3f,0xc9]      
vpmaxud %ymm9, %ymm9, %ymm9 

// CHECK: vpmaxuw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096, %ymm7, %ymm7 

// CHECK: vpmaxuw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmaxuw 485498096, %ymm9, %ymm9 

// CHECK: vpmaxuw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3e,0x7c,0x82,0xc0]      
vpmaxuw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxuw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3e,0x7c,0x82,0x40]      
vpmaxuw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmaxuw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3e,0x4c,0x82,0xc0]      
vpmaxuw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxuw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3e,0x4c,0x82,0x40]      
vpmaxuw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmaxuw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3e,0x7c,0x02,0x40]      
vpmaxuw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmaxuw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3e,0x4c,0x02,0x40]      
vpmaxuw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmaxuw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3e,0x7a,0x40]      
vpmaxuw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmaxuw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3e,0x4a,0x40]      
vpmaxuw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmaxuw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3e,0x3a]      
vpmaxuw (%rdx), %ymm7, %ymm7 

// CHECK: vpmaxuw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3e,0x0a]      
vpmaxuw (%rdx), %ymm9, %ymm9 

// CHECK: vpmaxuw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3e,0xff]      
vpmaxuw %ymm7, %ymm7, %ymm7 

// CHECK: vpmaxuw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x3e,0xc9]      
vpmaxuw %ymm9, %ymm9, %ymm9 

// CHECK: vpminsb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x38,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096, %ymm7, %ymm7 

// CHECK: vpminsb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x38,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsb 485498096, %ymm9, %ymm9 

// CHECK: vpminsb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x38,0x7c,0x82,0xc0]      
vpminsb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminsb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x38,0x7c,0x82,0x40]      
vpminsb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminsb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x38,0x4c,0x82,0xc0]      
vpminsb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminsb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x38,0x4c,0x82,0x40]      
vpminsb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminsb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x38,0x7c,0x02,0x40]      
vpminsb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpminsb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x38,0x4c,0x02,0x40]      
vpminsb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpminsb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x38,0x7a,0x40]      
vpminsb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpminsb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x38,0x4a,0x40]      
vpminsb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpminsb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x38,0x3a]      
vpminsb (%rdx), %ymm7, %ymm7 

// CHECK: vpminsb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x38,0x0a]      
vpminsb (%rdx), %ymm9, %ymm9 

// CHECK: vpminsb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x38,0xff]      
vpminsb %ymm7, %ymm7, %ymm7 

// CHECK: vpminsb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x38,0xc9]      
vpminsb %ymm9, %ymm9, %ymm9 

// CHECK: vpminsd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x39,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096, %ymm7, %ymm7 

// CHECK: vpminsd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x39,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsd 485498096, %ymm9, %ymm9 

// CHECK: vpminsd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x39,0x7c,0x82,0xc0]      
vpminsd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminsd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x39,0x7c,0x82,0x40]      
vpminsd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminsd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x39,0x4c,0x82,0xc0]      
vpminsd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminsd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x39,0x4c,0x82,0x40]      
vpminsd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminsd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x39,0x7c,0x02,0x40]      
vpminsd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpminsd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x39,0x4c,0x02,0x40]      
vpminsd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpminsd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x39,0x7a,0x40]      
vpminsd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpminsd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x39,0x4a,0x40]      
vpminsd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpminsd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x39,0x3a]      
vpminsd (%rdx), %ymm7, %ymm7 

// CHECK: vpminsd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x39,0x0a]      
vpminsd (%rdx), %ymm9, %ymm9 

// CHECK: vpminsd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x39,0xff]      
vpminsd %ymm7, %ymm7, %ymm7 

// CHECK: vpminsd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x39,0xc9]      
vpminsd %ymm9, %ymm9, %ymm9 

// CHECK: vpminsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xea,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096, %ymm7, %ymm7 

// CHECK: vpminsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xea,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminsw 485498096, %ymm9, %ymm9 

// CHECK: vpminsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xea,0x7c,0x82,0xc0]      
vpminsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xea,0x7c,0x82,0x40]      
vpminsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xea,0x4c,0x82,0xc0]      
vpminsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xea,0x4c,0x82,0x40]      
vpminsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xea,0x7c,0x02,0x40]      
vpminsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpminsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xea,0x4c,0x02,0x40]      
vpminsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpminsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xea,0x7a,0x40]      
vpminsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpminsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xea,0x4a,0x40]      
vpminsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpminsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xea,0x3a]      
vpminsw (%rdx), %ymm7, %ymm7 

// CHECK: vpminsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xea,0x0a]      
vpminsw (%rdx), %ymm9, %ymm9 

// CHECK: vpminsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xea,0xff]      
vpminsw %ymm7, %ymm7, %ymm7 

// CHECK: vpminsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xea,0xc9]      
vpminsw %ymm9, %ymm9, %ymm9 

// CHECK: vpminub 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xda,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096, %ymm7, %ymm7 

// CHECK: vpminub 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xda,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminub 485498096, %ymm9, %ymm9 

// CHECK: vpminub -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xda,0x7c,0x82,0xc0]      
vpminub -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminub 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xda,0x7c,0x82,0x40]      
vpminub 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminub -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xda,0x4c,0x82,0xc0]      
vpminub -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminub 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xda,0x4c,0x82,0x40]      
vpminub 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminub 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xda,0x7c,0x02,0x40]      
vpminub 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpminub 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xda,0x4c,0x02,0x40]      
vpminub 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpminub 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xda,0x7a,0x40]      
vpminub 64(%rdx), %ymm7, %ymm7 

// CHECK: vpminub 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xda,0x4a,0x40]      
vpminub 64(%rdx), %ymm9, %ymm9 

// CHECK: vpminub (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xda,0x3a]      
vpminub (%rdx), %ymm7, %ymm7 

// CHECK: vpminub (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xda,0x0a]      
vpminub (%rdx), %ymm9, %ymm9 

// CHECK: vpminub %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xda,0xff]      
vpminub %ymm7, %ymm7, %ymm7 

// CHECK: vpminub %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xda,0xc9]      
vpminub %ymm9, %ymm9, %ymm9 

// CHECK: vpminud 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096, %ymm7, %ymm7 

// CHECK: vpminud 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminud 485498096, %ymm9, %ymm9 

// CHECK: vpminud -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3b,0x7c,0x82,0xc0]      
vpminud -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminud 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3b,0x7c,0x82,0x40]      
vpminud 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminud -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3b,0x4c,0x82,0xc0]      
vpminud -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminud 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3b,0x4c,0x82,0x40]      
vpminud 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminud 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3b,0x7c,0x02,0x40]      
vpminud 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpminud 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3b,0x4c,0x02,0x40]      
vpminud 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpminud 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3b,0x7a,0x40]      
vpminud 64(%rdx), %ymm7, %ymm7 

// CHECK: vpminud 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3b,0x4a,0x40]      
vpminud 64(%rdx), %ymm9, %ymm9 

// CHECK: vpminud (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3b,0x3a]      
vpminud (%rdx), %ymm7, %ymm7 

// CHECK: vpminud (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3b,0x0a]      
vpminud (%rdx), %ymm9, %ymm9 

// CHECK: vpminud %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3b,0xff]      
vpminud %ymm7, %ymm7, %ymm7 

// CHECK: vpminud %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x3b,0xc9]      
vpminud %ymm9, %ymm9, %ymm9 

// CHECK: vpminuw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096, %ymm7, %ymm7 

// CHECK: vpminuw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpminuw 485498096, %ymm9, %ymm9 

// CHECK: vpminuw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3a,0x7c,0x82,0xc0]      
vpminuw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminuw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3a,0x7c,0x82,0x40]      
vpminuw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpminuw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3a,0x4c,0x82,0xc0]      
vpminuw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminuw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3a,0x4c,0x82,0x40]      
vpminuw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpminuw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3a,0x7c,0x02,0x40]      
vpminuw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpminuw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3a,0x4c,0x02,0x40]      
vpminuw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpminuw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3a,0x7a,0x40]      
vpminuw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpminuw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3a,0x4a,0x40]      
vpminuw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpminuw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3a,0x3a]      
vpminuw (%rdx), %ymm7, %ymm7 

// CHECK: vpminuw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x3a,0x0a]      
vpminuw (%rdx), %ymm9, %ymm9 

// CHECK: vpminuw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x3a,0xff]      
vpminuw %ymm7, %ymm7, %ymm7 

// CHECK: vpminuw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x3a,0xc9]      
vpminuw %ymm9, %ymm9, %ymm9 

// CHECK: vpmovmskb %ymm7, %r13d 
// CHECK: encoding: [0xc5,0x7d,0xd7,0xef]       
vpmovmskb %ymm7, %r13d 

// CHECK: vpmovmskb %ymm9, %r13d 
// CHECK: encoding: [0xc4,0x41,0x7d,0xd7,0xe9]       
vpmovmskb %ymm9, %r13d 

// CHECK: vpmovsxbd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096, %ymm7 

// CHECK: vpmovsxbd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x21,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbd 485498096, %ymm9 

// CHECK: vpmovsxbd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x7c,0x82,0xc0]       
vpmovsxbd -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxbd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x7c,0x82,0x40]       
vpmovsxbd 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxbd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x21,0x4c,0x82,0xc0]       
vpmovsxbd -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxbd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x21,0x4c,0x82,0x40]       
vpmovsxbd 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxbd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x7c,0x02,0x40]       
vpmovsxbd 64(%rdx,%rax), %ymm7 

// CHECK: vpmovsxbd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x21,0x4c,0x02,0x40]       
vpmovsxbd 64(%rdx,%rax), %ymm9 

// CHECK: vpmovsxbd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x7a,0x40]       
vpmovsxbd 64(%rdx), %ymm7 

// CHECK: vpmovsxbd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x21,0x4a,0x40]       
vpmovsxbd 64(%rdx), %ymm9 

// CHECK: vpmovsxbd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0x3a]       
vpmovsxbd (%rdx), %ymm7 

// CHECK: vpmovsxbd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x21,0x0a]       
vpmovsxbd (%rdx), %ymm9 

// CHECK: vpmovsxbd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x21,0xcf]       
vpmovsxbd %xmm15, %ymm9 

// CHECK: vpmovsxbd %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x21,0xfe]       
vpmovsxbd %xmm6, %ymm7 

// CHECK: vpmovsxbq 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096, %ymm7 

// CHECK: vpmovsxbq 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x22,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbq 485498096, %ymm9 

// CHECK: vpmovsxbq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x7c,0x82,0xc0]       
vpmovsxbq -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxbq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x7c,0x82,0x40]       
vpmovsxbq 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxbq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x22,0x4c,0x82,0xc0]       
vpmovsxbq -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxbq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x22,0x4c,0x82,0x40]       
vpmovsxbq 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxbq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x7c,0x02,0x40]       
vpmovsxbq 64(%rdx,%rax), %ymm7 

// CHECK: vpmovsxbq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x22,0x4c,0x02,0x40]       
vpmovsxbq 64(%rdx,%rax), %ymm9 

// CHECK: vpmovsxbq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x7a,0x40]       
vpmovsxbq 64(%rdx), %ymm7 

// CHECK: vpmovsxbq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x22,0x4a,0x40]       
vpmovsxbq 64(%rdx), %ymm9 

// CHECK: vpmovsxbq (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0x3a]       
vpmovsxbq (%rdx), %ymm7 

// CHECK: vpmovsxbq (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x22,0x0a]       
vpmovsxbq (%rdx), %ymm9 

// CHECK: vpmovsxbq %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x22,0xcf]       
vpmovsxbq %xmm15, %ymm9 

// CHECK: vpmovsxbq %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x22,0xfe]       
vpmovsxbq %xmm6, %ymm7 

// CHECK: vpmovsxbw 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096, %ymm7 

// CHECK: vpmovsxbw 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x20,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxbw 485498096, %ymm9 

// CHECK: vpmovsxbw -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x7c,0x82,0xc0]       
vpmovsxbw -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxbw 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x7c,0x82,0x40]       
vpmovsxbw 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxbw -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x20,0x4c,0x82,0xc0]       
vpmovsxbw -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxbw 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x20,0x4c,0x82,0x40]       
vpmovsxbw 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxbw 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x7c,0x02,0x40]       
vpmovsxbw 64(%rdx,%rax), %ymm7 

// CHECK: vpmovsxbw 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x20,0x4c,0x02,0x40]       
vpmovsxbw 64(%rdx,%rax), %ymm9 

// CHECK: vpmovsxbw 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x7a,0x40]       
vpmovsxbw 64(%rdx), %ymm7 

// CHECK: vpmovsxbw 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x20,0x4a,0x40]       
vpmovsxbw 64(%rdx), %ymm9 

// CHECK: vpmovsxbw (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0x3a]       
vpmovsxbw (%rdx), %ymm7 

// CHECK: vpmovsxbw (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x20,0x0a]       
vpmovsxbw (%rdx), %ymm9 

// CHECK: vpmovsxbw %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x20,0xcf]       
vpmovsxbw %xmm15, %ymm9 

// CHECK: vpmovsxbw %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x20,0xfe]       
vpmovsxbw %xmm6, %ymm7 

// CHECK: vpmovsxdq 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096, %ymm7 

// CHECK: vpmovsxdq 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x25,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxdq 485498096, %ymm9 

// CHECK: vpmovsxdq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x7c,0x82,0xc0]       
vpmovsxdq -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxdq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x7c,0x82,0x40]       
vpmovsxdq 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxdq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x25,0x4c,0x82,0xc0]       
vpmovsxdq -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxdq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x25,0x4c,0x82,0x40]       
vpmovsxdq 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxdq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x7c,0x02,0x40]       
vpmovsxdq 64(%rdx,%rax), %ymm7 

// CHECK: vpmovsxdq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x25,0x4c,0x02,0x40]       
vpmovsxdq 64(%rdx,%rax), %ymm9 

// CHECK: vpmovsxdq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x7a,0x40]       
vpmovsxdq 64(%rdx), %ymm7 

// CHECK: vpmovsxdq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x25,0x4a,0x40]       
vpmovsxdq 64(%rdx), %ymm9 

// CHECK: vpmovsxdq (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0x3a]       
vpmovsxdq (%rdx), %ymm7 

// CHECK: vpmovsxdq (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x25,0x0a]       
vpmovsxdq (%rdx), %ymm9 

// CHECK: vpmovsxdq %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x25,0xcf]       
vpmovsxdq %xmm15, %ymm9 

// CHECK: vpmovsxdq %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x25,0xfe]       
vpmovsxdq %xmm6, %ymm7 

// CHECK: vpmovsxwd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096, %ymm7 

// CHECK: vpmovsxwd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x23,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwd 485498096, %ymm9 

// CHECK: vpmovsxwd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x7c,0x82,0xc0]       
vpmovsxwd -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxwd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x7c,0x82,0x40]       
vpmovsxwd 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxwd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x23,0x4c,0x82,0xc0]       
vpmovsxwd -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxwd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x23,0x4c,0x82,0x40]       
vpmovsxwd 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxwd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x7c,0x02,0x40]       
vpmovsxwd 64(%rdx,%rax), %ymm7 

// CHECK: vpmovsxwd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x23,0x4c,0x02,0x40]       
vpmovsxwd 64(%rdx,%rax), %ymm9 

// CHECK: vpmovsxwd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x7a,0x40]       
vpmovsxwd 64(%rdx), %ymm7 

// CHECK: vpmovsxwd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x23,0x4a,0x40]       
vpmovsxwd 64(%rdx), %ymm9 

// CHECK: vpmovsxwd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0x3a]       
vpmovsxwd (%rdx), %ymm7 

// CHECK: vpmovsxwd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x23,0x0a]       
vpmovsxwd (%rdx), %ymm9 

// CHECK: vpmovsxwd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x23,0xcf]       
vpmovsxwd %xmm15, %ymm9 

// CHECK: vpmovsxwd %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x23,0xfe]       
vpmovsxwd %xmm6, %ymm7 

// CHECK: vpmovsxwq 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096, %ymm7 

// CHECK: vpmovsxwq 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x24,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovsxwq 485498096, %ymm9 

// CHECK: vpmovsxwq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x7c,0x82,0xc0]       
vpmovsxwq -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxwq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x7c,0x82,0x40]       
vpmovsxwq 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovsxwq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x24,0x4c,0x82,0xc0]       
vpmovsxwq -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxwq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x24,0x4c,0x82,0x40]       
vpmovsxwq 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovsxwq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x7c,0x02,0x40]       
vpmovsxwq 64(%rdx,%rax), %ymm7 

// CHECK: vpmovsxwq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x24,0x4c,0x02,0x40]       
vpmovsxwq 64(%rdx,%rax), %ymm9 

// CHECK: vpmovsxwq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x7a,0x40]       
vpmovsxwq 64(%rdx), %ymm7 

// CHECK: vpmovsxwq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x24,0x4a,0x40]       
vpmovsxwq 64(%rdx), %ymm9 

// CHECK: vpmovsxwq (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0x3a]       
vpmovsxwq (%rdx), %ymm7 

// CHECK: vpmovsxwq (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x24,0x0a]       
vpmovsxwq (%rdx), %ymm9 

// CHECK: vpmovsxwq %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x24,0xcf]       
vpmovsxwq %xmm15, %ymm9 

// CHECK: vpmovsxwq %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x24,0xfe]       
vpmovsxwq %xmm6, %ymm7 

// CHECK: vpmovzxbd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096, %ymm7 

// CHECK: vpmovzxbd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x31,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbd 485498096, %ymm9 

// CHECK: vpmovzxbd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x7c,0x82,0xc0]       
vpmovzxbd -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxbd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x7c,0x82,0x40]       
vpmovzxbd 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxbd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x31,0x4c,0x82,0xc0]       
vpmovzxbd -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxbd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x31,0x4c,0x82,0x40]       
vpmovzxbd 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxbd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x7c,0x02,0x40]       
vpmovzxbd 64(%rdx,%rax), %ymm7 

// CHECK: vpmovzxbd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x31,0x4c,0x02,0x40]       
vpmovzxbd 64(%rdx,%rax), %ymm9 

// CHECK: vpmovzxbd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x7a,0x40]       
vpmovzxbd 64(%rdx), %ymm7 

// CHECK: vpmovzxbd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x31,0x4a,0x40]       
vpmovzxbd 64(%rdx), %ymm9 

// CHECK: vpmovzxbd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0x3a]       
vpmovzxbd (%rdx), %ymm7 

// CHECK: vpmovzxbd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x31,0x0a]       
vpmovzxbd (%rdx), %ymm9 

// CHECK: vpmovzxbd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x31,0xcf]       
vpmovzxbd %xmm15, %ymm9 

// CHECK: vpmovzxbd %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x31,0xfe]       
vpmovzxbd %xmm6, %ymm7 

// CHECK: vpmovzxbq 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096, %ymm7 

// CHECK: vpmovzxbq 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x32,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbq 485498096, %ymm9 

// CHECK: vpmovzxbq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x7c,0x82,0xc0]       
vpmovzxbq -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxbq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x7c,0x82,0x40]       
vpmovzxbq 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxbq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x32,0x4c,0x82,0xc0]       
vpmovzxbq -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxbq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x32,0x4c,0x82,0x40]       
vpmovzxbq 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxbq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x7c,0x02,0x40]       
vpmovzxbq 64(%rdx,%rax), %ymm7 

// CHECK: vpmovzxbq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x32,0x4c,0x02,0x40]       
vpmovzxbq 64(%rdx,%rax), %ymm9 

// CHECK: vpmovzxbq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x7a,0x40]       
vpmovzxbq 64(%rdx), %ymm7 

// CHECK: vpmovzxbq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x32,0x4a,0x40]       
vpmovzxbq 64(%rdx), %ymm9 

// CHECK: vpmovzxbq (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0x3a]       
vpmovzxbq (%rdx), %ymm7 

// CHECK: vpmovzxbq (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x32,0x0a]       
vpmovzxbq (%rdx), %ymm9 

// CHECK: vpmovzxbq %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x32,0xcf]       
vpmovzxbq %xmm15, %ymm9 

// CHECK: vpmovzxbq %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x32,0xfe]       
vpmovzxbq %xmm6, %ymm7 

// CHECK: vpmovzxbw 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096, %ymm7 

// CHECK: vpmovzxbw 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x30,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxbw 485498096, %ymm9 

// CHECK: vpmovzxbw -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x7c,0x82,0xc0]       
vpmovzxbw -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxbw 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x7c,0x82,0x40]       
vpmovzxbw 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxbw -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x30,0x4c,0x82,0xc0]       
vpmovzxbw -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxbw 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x30,0x4c,0x82,0x40]       
vpmovzxbw 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxbw 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x7c,0x02,0x40]       
vpmovzxbw 64(%rdx,%rax), %ymm7 

// CHECK: vpmovzxbw 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x30,0x4c,0x02,0x40]       
vpmovzxbw 64(%rdx,%rax), %ymm9 

// CHECK: vpmovzxbw 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x7a,0x40]       
vpmovzxbw 64(%rdx), %ymm7 

// CHECK: vpmovzxbw 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x30,0x4a,0x40]       
vpmovzxbw 64(%rdx), %ymm9 

// CHECK: vpmovzxbw (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0x3a]       
vpmovzxbw (%rdx), %ymm7 

// CHECK: vpmovzxbw (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x30,0x0a]       
vpmovzxbw (%rdx), %ymm9 

// CHECK: vpmovzxbw %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x30,0xcf]       
vpmovzxbw %xmm15, %ymm9 

// CHECK: vpmovzxbw %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x30,0xfe]       
vpmovzxbw %xmm6, %ymm7 

// CHECK: vpmovzxdq 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096, %ymm7 

// CHECK: vpmovzxdq 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x35,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxdq 485498096, %ymm9 

// CHECK: vpmovzxdq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x7c,0x82,0xc0]       
vpmovzxdq -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxdq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x7c,0x82,0x40]       
vpmovzxdq 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxdq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x35,0x4c,0x82,0xc0]       
vpmovzxdq -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxdq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x35,0x4c,0x82,0x40]       
vpmovzxdq 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxdq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x7c,0x02,0x40]       
vpmovzxdq 64(%rdx,%rax), %ymm7 

// CHECK: vpmovzxdq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x35,0x4c,0x02,0x40]       
vpmovzxdq 64(%rdx,%rax), %ymm9 

// CHECK: vpmovzxdq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x7a,0x40]       
vpmovzxdq 64(%rdx), %ymm7 

// CHECK: vpmovzxdq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x35,0x4a,0x40]       
vpmovzxdq 64(%rdx), %ymm9 

// CHECK: vpmovzxdq (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0x3a]       
vpmovzxdq (%rdx), %ymm7 

// CHECK: vpmovzxdq (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x35,0x0a]       
vpmovzxdq (%rdx), %ymm9 

// CHECK: vpmovzxdq %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x35,0xcf]       
vpmovzxdq %xmm15, %ymm9 

// CHECK: vpmovzxdq %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x35,0xfe]       
vpmovzxdq %xmm6, %ymm7 

// CHECK: vpmovzxwd 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096, %ymm7 

// CHECK: vpmovzxwd 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x33,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwd 485498096, %ymm9 

// CHECK: vpmovzxwd -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x7c,0x82,0xc0]       
vpmovzxwd -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxwd 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x7c,0x82,0x40]       
vpmovzxwd 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxwd -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x33,0x4c,0x82,0xc0]       
vpmovzxwd -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxwd 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x33,0x4c,0x82,0x40]       
vpmovzxwd 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxwd 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x7c,0x02,0x40]       
vpmovzxwd 64(%rdx,%rax), %ymm7 

// CHECK: vpmovzxwd 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x33,0x4c,0x02,0x40]       
vpmovzxwd 64(%rdx,%rax), %ymm9 

// CHECK: vpmovzxwd 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x7a,0x40]       
vpmovzxwd 64(%rdx), %ymm7 

// CHECK: vpmovzxwd 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x33,0x4a,0x40]       
vpmovzxwd 64(%rdx), %ymm9 

// CHECK: vpmovzxwd (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0x3a]       
vpmovzxwd (%rdx), %ymm7 

// CHECK: vpmovzxwd (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x33,0x0a]       
vpmovzxwd (%rdx), %ymm9 

// CHECK: vpmovzxwd %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x33,0xcf]       
vpmovzxwd %xmm15, %ymm9 

// CHECK: vpmovzxwd %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x33,0xfe]       
vpmovzxwd %xmm6, %ymm7 

// CHECK: vpmovzxwq 485498096, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096, %ymm7 

// CHECK: vpmovzxwq 485498096, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x34,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]       
vpmovzxwq 485498096, %ymm9 

// CHECK: vpmovzxwq -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x7c,0x82,0xc0]       
vpmovzxwq -64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxwq 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x7c,0x82,0x40]       
vpmovzxwq 64(%rdx,%rax,4), %ymm7 

// CHECK: vpmovzxwq -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x34,0x4c,0x82,0xc0]       
vpmovzxwq -64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxwq 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x34,0x4c,0x82,0x40]       
vpmovzxwq 64(%rdx,%rax,4), %ymm9 

// CHECK: vpmovzxwq 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x7c,0x02,0x40]       
vpmovzxwq 64(%rdx,%rax), %ymm7 

// CHECK: vpmovzxwq 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x34,0x4c,0x02,0x40]       
vpmovzxwq 64(%rdx,%rax), %ymm9 

// CHECK: vpmovzxwq 64(%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x7a,0x40]       
vpmovzxwq 64(%rdx), %ymm7 

// CHECK: vpmovzxwq 64(%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x34,0x4a,0x40]       
vpmovzxwq 64(%rdx), %ymm9 

// CHECK: vpmovzxwq (%rdx), %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0x3a]       
vpmovzxwq (%rdx), %ymm7 

// CHECK: vpmovzxwq (%rdx), %ymm9 
// CHECK: encoding: [0xc4,0x62,0x7d,0x34,0x0a]       
vpmovzxwq (%rdx), %ymm9 

// CHECK: vpmovzxwq %xmm15, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x7d,0x34,0xcf]       
vpmovzxwq %xmm15, %ymm9 

// CHECK: vpmovzxwq %xmm6, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x7d,0x34,0xfe]       
vpmovzxwq %xmm6, %ymm7 

// CHECK: vpmuldq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x28,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096, %ymm7, %ymm7 

// CHECK: vpmuldq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x28,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuldq 485498096, %ymm9, %ymm9 

// CHECK: vpmuldq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x28,0x7c,0x82,0xc0]      
vpmuldq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmuldq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x28,0x7c,0x82,0x40]      
vpmuldq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmuldq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x28,0x4c,0x82,0xc0]      
vpmuldq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmuldq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x28,0x4c,0x82,0x40]      
vpmuldq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmuldq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x28,0x7c,0x02,0x40]      
vpmuldq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmuldq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x28,0x4c,0x02,0x40]      
vpmuldq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmuldq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x28,0x7a,0x40]      
vpmuldq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmuldq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x28,0x4a,0x40]      
vpmuldq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmuldq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x28,0x3a]      
vpmuldq (%rdx), %ymm7, %ymm7 

// CHECK: vpmuldq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x28,0x0a]      
vpmuldq (%rdx), %ymm9, %ymm9 

// CHECK: vpmuldq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x28,0xff]      
vpmuldq %ymm7, %ymm7, %ymm7 

// CHECK: vpmuldq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x28,0xc9]      
vpmuldq %ymm9, %ymm9, %ymm9 

// CHECK: vpmulhrsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096, %ymm7, %ymm7 

// CHECK: vpmulhrsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhrsw 485498096, %ymm9, %ymm9 

// CHECK: vpmulhrsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0b,0x7c,0x82,0xc0]      
vpmulhrsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulhrsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0b,0x7c,0x82,0x40]      
vpmulhrsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulhrsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0b,0x4c,0x82,0xc0]      
vpmulhrsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulhrsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0b,0x4c,0x82,0x40]      
vpmulhrsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulhrsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0b,0x7c,0x02,0x40]      
vpmulhrsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmulhrsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0b,0x4c,0x02,0x40]      
vpmulhrsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmulhrsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0b,0x7a,0x40]      
vpmulhrsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmulhrsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0b,0x4a,0x40]      
vpmulhrsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmulhrsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0b,0x3a]      
vpmulhrsw (%rdx), %ymm7, %ymm7 

// CHECK: vpmulhrsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0b,0x0a]      
vpmulhrsw (%rdx), %ymm9, %ymm9 

// CHECK: vpmulhrsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0b,0xff]      
vpmulhrsw %ymm7, %ymm7, %ymm7 

// CHECK: vpmulhrsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x0b,0xc9]      
vpmulhrsw %ymm9, %ymm9, %ymm9 

// CHECK: vpmulhuw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe4,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096, %ymm7, %ymm7 

// CHECK: vpmulhuw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe4,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhuw 485498096, %ymm9, %ymm9 

// CHECK: vpmulhuw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe4,0x7c,0x82,0xc0]      
vpmulhuw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulhuw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe4,0x7c,0x82,0x40]      
vpmulhuw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulhuw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe4,0x4c,0x82,0xc0]      
vpmulhuw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulhuw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe4,0x4c,0x82,0x40]      
vpmulhuw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulhuw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe4,0x7c,0x02,0x40]      
vpmulhuw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmulhuw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe4,0x4c,0x02,0x40]      
vpmulhuw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmulhuw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe4,0x7a,0x40]      
vpmulhuw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmulhuw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe4,0x4a,0x40]      
vpmulhuw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmulhuw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe4,0x3a]      
vpmulhuw (%rdx), %ymm7, %ymm7 

// CHECK: vpmulhuw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe4,0x0a]      
vpmulhuw (%rdx), %ymm9, %ymm9 

// CHECK: vpmulhuw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe4,0xff]      
vpmulhuw %ymm7, %ymm7, %ymm7 

// CHECK: vpmulhuw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe4,0xc9]      
vpmulhuw %ymm9, %ymm9, %ymm9 

// CHECK: vpmulhw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe5,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096, %ymm7, %ymm7 

// CHECK: vpmulhw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe5,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulhw 485498096, %ymm9, %ymm9 

// CHECK: vpmulhw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe5,0x7c,0x82,0xc0]      
vpmulhw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulhw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe5,0x7c,0x82,0x40]      
vpmulhw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulhw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe5,0x4c,0x82,0xc0]      
vpmulhw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulhw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe5,0x4c,0x82,0x40]      
vpmulhw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulhw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe5,0x7c,0x02,0x40]      
vpmulhw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmulhw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe5,0x4c,0x02,0x40]      
vpmulhw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmulhw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe5,0x7a,0x40]      
vpmulhw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmulhw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe5,0x4a,0x40]      
vpmulhw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmulhw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe5,0x3a]      
vpmulhw (%rdx), %ymm7, %ymm7 

// CHECK: vpmulhw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe5,0x0a]      
vpmulhw (%rdx), %ymm9, %ymm9 

// CHECK: vpmulhw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe5,0xff]      
vpmulhw %ymm7, %ymm7, %ymm7 

// CHECK: vpmulhw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe5,0xc9]      
vpmulhw %ymm9, %ymm9, %ymm9 

// CHECK: vpmulld 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x40,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096, %ymm7, %ymm7 

// CHECK: vpmulld 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x40,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmulld 485498096, %ymm9, %ymm9 

// CHECK: vpmulld -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x40,0x7c,0x82,0xc0]      
vpmulld -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulld 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x40,0x7c,0x82,0x40]      
vpmulld 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmulld -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x40,0x4c,0x82,0xc0]      
vpmulld -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulld 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x40,0x4c,0x82,0x40]      
vpmulld 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmulld 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x40,0x7c,0x02,0x40]      
vpmulld 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmulld 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x40,0x4c,0x02,0x40]      
vpmulld 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmulld 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x40,0x7a,0x40]      
vpmulld 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmulld 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x40,0x4a,0x40]      
vpmulld 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmulld (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x40,0x3a]      
vpmulld (%rdx), %ymm7, %ymm7 

// CHECK: vpmulld (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x40,0x0a]      
vpmulld (%rdx), %ymm9, %ymm9 

// CHECK: vpmulld %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x40,0xff]      
vpmulld %ymm7, %ymm7, %ymm7 

// CHECK: vpmulld %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x40,0xc9]      
vpmulld %ymm9, %ymm9, %ymm9 

// CHECK: vpmullw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd5,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096, %ymm7, %ymm7 

// CHECK: vpmullw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd5,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmullw 485498096, %ymm9, %ymm9 

// CHECK: vpmullw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd5,0x7c,0x82,0xc0]      
vpmullw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmullw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd5,0x7c,0x82,0x40]      
vpmullw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmullw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd5,0x4c,0x82,0xc0]      
vpmullw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmullw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd5,0x4c,0x82,0x40]      
vpmullw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmullw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd5,0x7c,0x02,0x40]      
vpmullw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmullw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd5,0x4c,0x02,0x40]      
vpmullw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmullw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd5,0x7a,0x40]      
vpmullw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmullw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd5,0x4a,0x40]      
vpmullw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmullw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd5,0x3a]      
vpmullw (%rdx), %ymm7, %ymm7 

// CHECK: vpmullw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd5,0x0a]      
vpmullw (%rdx), %ymm9, %ymm9 

// CHECK: vpmullw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd5,0xff]      
vpmullw %ymm7, %ymm7, %ymm7 

// CHECK: vpmullw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd5,0xc9]      
vpmullw %ymm9, %ymm9, %ymm9 

// CHECK: vpmuludq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf4,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096, %ymm7, %ymm7 

// CHECK: vpmuludq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf4,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpmuludq 485498096, %ymm9, %ymm9 

// CHECK: vpmuludq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf4,0x7c,0x82,0xc0]      
vpmuludq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmuludq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf4,0x7c,0x82,0x40]      
vpmuludq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpmuludq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf4,0x4c,0x82,0xc0]      
vpmuludq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmuludq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf4,0x4c,0x82,0x40]      
vpmuludq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpmuludq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf4,0x7c,0x02,0x40]      
vpmuludq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpmuludq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf4,0x4c,0x02,0x40]      
vpmuludq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpmuludq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf4,0x7a,0x40]      
vpmuludq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpmuludq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf4,0x4a,0x40]      
vpmuludq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpmuludq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf4,0x3a]      
vpmuludq (%rdx), %ymm7, %ymm7 

// CHECK: vpmuludq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf4,0x0a]      
vpmuludq (%rdx), %ymm9, %ymm9 

// CHECK: vpmuludq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf4,0xff]      
vpmuludq %ymm7, %ymm7, %ymm7 

// CHECK: vpmuludq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf4,0xc9]      
vpmuludq %ymm9, %ymm9, %ymm9 

// CHECK: vpor 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xeb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096, %ymm7, %ymm7 

// CHECK: vpor 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xeb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpor 485498096, %ymm9, %ymm9 

// CHECK: vpor -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xeb,0x7c,0x82,0xc0]      
vpor -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpor 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xeb,0x7c,0x82,0x40]      
vpor 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpor -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xeb,0x4c,0x82,0xc0]      
vpor -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpor 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xeb,0x4c,0x82,0x40]      
vpor 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpor 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xeb,0x7c,0x02,0x40]      
vpor 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpor 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xeb,0x4c,0x02,0x40]      
vpor 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpor 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xeb,0x7a,0x40]      
vpor 64(%rdx), %ymm7, %ymm7 

// CHECK: vpor 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xeb,0x4a,0x40]      
vpor 64(%rdx), %ymm9, %ymm9 

// CHECK: vpor (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xeb,0x3a]      
vpor (%rdx), %ymm7, %ymm7 

// CHECK: vpor (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xeb,0x0a]      
vpor (%rdx), %ymm9, %ymm9 

// CHECK: vpor %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xeb,0xff]      
vpor %ymm7, %ymm7, %ymm7 

// CHECK: vpor %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xeb,0xc9]      
vpor %ymm9, %ymm9, %ymm9 

// CHECK: vpsadbw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf6,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096, %ymm7, %ymm7 

// CHECK: vpsadbw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf6,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsadbw 485498096, %ymm9, %ymm9 

// CHECK: vpsadbw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf6,0x7c,0x82,0xc0]      
vpsadbw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsadbw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf6,0x7c,0x82,0x40]      
vpsadbw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsadbw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf6,0x4c,0x82,0xc0]      
vpsadbw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsadbw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf6,0x4c,0x82,0x40]      
vpsadbw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsadbw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf6,0x7c,0x02,0x40]      
vpsadbw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsadbw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf6,0x4c,0x02,0x40]      
vpsadbw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsadbw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf6,0x7a,0x40]      
vpsadbw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsadbw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf6,0x4a,0x40]      
vpsadbw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsadbw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf6,0x3a]      
vpsadbw (%rdx), %ymm7, %ymm7 

// CHECK: vpsadbw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf6,0x0a]      
vpsadbw (%rdx), %ymm9, %ymm9 

// CHECK: vpsadbw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf6,0xff]      
vpsadbw %ymm7, %ymm7, %ymm7 

// CHECK: vpsadbw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf6,0xc9]      
vpsadbw %ymm9, %ymm9, %ymm9 

// CHECK: vpshufb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x00,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096, %ymm7, %ymm7 

// CHECK: vpshufb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x00,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpshufb 485498096, %ymm9, %ymm9 

// CHECK: vpshufb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x00,0x7c,0x82,0xc0]      
vpshufb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpshufb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x00,0x7c,0x82,0x40]      
vpshufb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpshufb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x00,0x4c,0x82,0xc0]      
vpshufb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpshufb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x00,0x4c,0x82,0x40]      
vpshufb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpshufb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x00,0x7c,0x02,0x40]      
vpshufb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpshufb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x00,0x4c,0x02,0x40]      
vpshufb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpshufb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x00,0x7a,0x40]      
vpshufb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpshufb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x00,0x4a,0x40]      
vpshufb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpshufb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x00,0x3a]      
vpshufb (%rdx), %ymm7, %ymm7 

// CHECK: vpshufb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x00,0x0a]      
vpshufb (%rdx), %ymm9, %ymm9 

// CHECK: vpshufb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x00,0xff]      
vpshufb %ymm7, %ymm7, %ymm7 

// CHECK: vpshufb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x00,0xc9]      
vpshufb %ymm9, %ymm9, %ymm9 

// CHECK: vpshufd $0, 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x70,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096, %ymm7 

// CHECK: vpshufd $0, 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x70,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufd $0, 485498096, %ymm9 

// CHECK: vpshufd $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x70,0x7c,0x82,0xc0,0x00]      
vpshufd $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vpshufd $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x70,0x7c,0x82,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vpshufd $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x70,0x4c,0x82,0xc0,0x00]      
vpshufd $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vpshufd $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x70,0x4c,0x82,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vpshufd $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x70,0x7c,0x02,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax), %ymm7 

// CHECK: vpshufd $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x70,0x4c,0x02,0x40,0x00]      
vpshufd $0, 64(%rdx,%rax), %ymm9 

// CHECK: vpshufd $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x70,0x7a,0x40,0x00]      
vpshufd $0, 64(%rdx), %ymm7 

// CHECK: vpshufd $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x70,0x4a,0x40,0x00]      
vpshufd $0, 64(%rdx), %ymm9 

// CHECK: vpshufd $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x70,0x3a,0x00]      
vpshufd $0, (%rdx), %ymm7 

// CHECK: vpshufd $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7d,0x70,0x0a,0x00]      
vpshufd $0, (%rdx), %ymm9 

// CHECK: vpshufd $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfd,0x70,0xff,0x00]      
vpshufd $0, %ymm7, %ymm7 

// CHECK: vpshufd $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7d,0x70,0xc9,0x00]      
vpshufd $0, %ymm9, %ymm9 

// CHECK: vpshufhw $0, 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x70,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096, %ymm7 

// CHECK: vpshufhw $0, 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x70,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshufhw $0, 485498096, %ymm9 

// CHECK: vpshufhw $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x70,0x7c,0x82,0xc0,0x00]      
vpshufhw $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vpshufhw $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x70,0x7c,0x82,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vpshufhw $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x70,0x4c,0x82,0xc0,0x00]      
vpshufhw $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vpshufhw $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x70,0x4c,0x82,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vpshufhw $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x70,0x7c,0x02,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax), %ymm7 

// CHECK: vpshufhw $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x70,0x4c,0x02,0x40,0x00]      
vpshufhw $0, 64(%rdx,%rax), %ymm9 

// CHECK: vpshufhw $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x70,0x7a,0x40,0x00]      
vpshufhw $0, 64(%rdx), %ymm7 

// CHECK: vpshufhw $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x70,0x4a,0x40,0x00]      
vpshufhw $0, 64(%rdx), %ymm9 

// CHECK: vpshufhw $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x70,0x3a,0x00]      
vpshufhw $0, (%rdx), %ymm7 

// CHECK: vpshufhw $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7e,0x70,0x0a,0x00]      
vpshufhw $0, (%rdx), %ymm9 

// CHECK: vpshufhw $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xfe,0x70,0xff,0x00]      
vpshufhw $0, %ymm7, %ymm7 

// CHECK: vpshufhw $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7e,0x70,0xc9,0x00]      
vpshufhw $0, %ymm9, %ymm9 

// CHECK: vpshuflw $0, 485498096, %ymm7 
// CHECK: encoding: [0xc5,0xff,0x70,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096, %ymm7 

// CHECK: vpshuflw $0, 485498096, %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x70,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]      
vpshuflw $0, 485498096, %ymm9 

// CHECK: vpshuflw $0, -64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x70,0x7c,0x82,0xc0,0x00]      
vpshuflw $0, -64(%rdx,%rax,4), %ymm7 

// CHECK: vpshuflw $0, 64(%rdx,%rax,4), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x70,0x7c,0x82,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax,4), %ymm7 

// CHECK: vpshuflw $0, -64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x70,0x4c,0x82,0xc0,0x00]      
vpshuflw $0, -64(%rdx,%rax,4), %ymm9 

// CHECK: vpshuflw $0, 64(%rdx,%rax,4), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x70,0x4c,0x82,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax,4), %ymm9 

// CHECK: vpshuflw $0, 64(%rdx,%rax), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x70,0x7c,0x02,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax), %ymm7 

// CHECK: vpshuflw $0, 64(%rdx,%rax), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x70,0x4c,0x02,0x40,0x00]      
vpshuflw $0, 64(%rdx,%rax), %ymm9 

// CHECK: vpshuflw $0, 64(%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x70,0x7a,0x40,0x00]      
vpshuflw $0, 64(%rdx), %ymm7 

// CHECK: vpshuflw $0, 64(%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x70,0x4a,0x40,0x00]      
vpshuflw $0, 64(%rdx), %ymm9 

// CHECK: vpshuflw $0, (%rdx), %ymm7 
// CHECK: encoding: [0xc5,0xff,0x70,0x3a,0x00]      
vpshuflw $0, (%rdx), %ymm7 

// CHECK: vpshuflw $0, (%rdx), %ymm9 
// CHECK: encoding: [0xc5,0x7f,0x70,0x0a,0x00]      
vpshuflw $0, (%rdx), %ymm9 

// CHECK: vpshuflw $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xff,0x70,0xff,0x00]      
vpshuflw $0, %ymm7, %ymm7 

// CHECK: vpshuflw $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x7f,0x70,0xc9,0x00]      
vpshuflw $0, %ymm9, %ymm9 

// CHECK: vpsignb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x08,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096, %ymm7, %ymm7 

// CHECK: vpsignb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x08,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignb 485498096, %ymm9, %ymm9 

// CHECK: vpsignb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x08,0x7c,0x82,0xc0]      
vpsignb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsignb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x08,0x7c,0x82,0x40]      
vpsignb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsignb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x08,0x4c,0x82,0xc0]      
vpsignb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsignb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x08,0x4c,0x82,0x40]      
vpsignb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsignb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x08,0x7c,0x02,0x40]      
vpsignb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsignb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x08,0x4c,0x02,0x40]      
vpsignb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsignb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x08,0x7a,0x40]      
vpsignb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsignb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x08,0x4a,0x40]      
vpsignb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsignb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x08,0x3a]      
vpsignb (%rdx), %ymm7, %ymm7 

// CHECK: vpsignb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x08,0x0a]      
vpsignb (%rdx), %ymm9, %ymm9 

// CHECK: vpsignb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x08,0xff]      
vpsignb %ymm7, %ymm7, %ymm7 

// CHECK: vpsignb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x08,0xc9]      
vpsignb %ymm9, %ymm9, %ymm9 

// CHECK: vpsignd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096, %ymm7, %ymm7 

// CHECK: vpsignd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignd 485498096, %ymm9, %ymm9 

// CHECK: vpsignd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0a,0x7c,0x82,0xc0]      
vpsignd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsignd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0a,0x7c,0x82,0x40]      
vpsignd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsignd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0a,0x4c,0x82,0xc0]      
vpsignd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsignd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0a,0x4c,0x82,0x40]      
vpsignd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsignd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0a,0x7c,0x02,0x40]      
vpsignd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsignd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0a,0x4c,0x02,0x40]      
vpsignd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsignd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0a,0x7a,0x40]      
vpsignd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsignd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0a,0x4a,0x40]      
vpsignd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsignd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0a,0x3a]      
vpsignd (%rdx), %ymm7, %ymm7 

// CHECK: vpsignd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x0a,0x0a]      
vpsignd (%rdx), %ymm9, %ymm9 

// CHECK: vpsignd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x0a,0xff]      
vpsignd %ymm7, %ymm7, %ymm7 

// CHECK: vpsignd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x0a,0xc9]      
vpsignd %ymm9, %ymm9, %ymm9 

// CHECK: vpsignw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x09,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096, %ymm7, %ymm7 

// CHECK: vpsignw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x09,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsignw 485498096, %ymm9, %ymm9 

// CHECK: vpsignw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x09,0x7c,0x82,0xc0]      
vpsignw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsignw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x09,0x7c,0x82,0x40]      
vpsignw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsignw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x09,0x4c,0x82,0xc0]      
vpsignw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsignw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x09,0x4c,0x82,0x40]      
vpsignw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsignw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x09,0x7c,0x02,0x40]      
vpsignw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsignw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x09,0x4c,0x02,0x40]      
vpsignw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsignw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x09,0x7a,0x40]      
vpsignw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsignw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x09,0x4a,0x40]      
vpsignw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsignw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x09,0x3a]      
vpsignw (%rdx), %ymm7, %ymm7 

// CHECK: vpsignw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x09,0x0a]      
vpsignw (%rdx), %ymm9, %ymm9 

// CHECK: vpsignw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x09,0xff]      
vpsignw %ymm7, %ymm7, %ymm7 

// CHECK: vpsignw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x09,0xc9]      
vpsignw %ymm9, %ymm9, %ymm9 

// CHECK: vpslld $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x72,0xf7,0x00]      
vpslld $0, %ymm7, %ymm7 

// CHECK: vpslld $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x72,0xf1,0x00]      
vpslld $0, %ymm9, %ymm9 

// CHECK: vpslld 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096, %ymm7, %ymm7 

// CHECK: vpslld 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf2,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpslld 485498096, %ymm9, %ymm9 

// CHECK: vpslld -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf2,0x7c,0x82,0xc0]      
vpslld -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpslld 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf2,0x7c,0x82,0x40]      
vpslld 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpslld -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf2,0x4c,0x82,0xc0]      
vpslld -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpslld 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf2,0x4c,0x82,0x40]      
vpslld 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpslld 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf2,0x7c,0x02,0x40]      
vpslld 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpslld 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf2,0x4c,0x02,0x40]      
vpslld 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpslld 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf2,0x7a,0x40]      
vpslld 64(%rdx), %ymm7, %ymm7 

// CHECK: vpslld 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf2,0x4a,0x40]      
vpslld 64(%rdx), %ymm9, %ymm9 

// CHECK: vpslldq $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x73,0xff,0x00]      
vpslldq $0, %ymm7, %ymm7 

// CHECK: vpslldq $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x73,0xf9,0x00]      
vpslldq $0, %ymm9, %ymm9 

// CHECK: vpslld (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf2,0x3a]      
vpslld (%rdx), %ymm7, %ymm7 

// CHECK: vpslld (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf2,0x0a]      
vpslld (%rdx), %ymm9, %ymm9 

// CHECK: vpslld %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf2,0xcf]      
vpslld %xmm15, %ymm9, %ymm9 

// CHECK: vpslld %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf2,0xfe]      
vpslld %xmm6, %ymm7, %ymm7 

// CHECK: vpsllq $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x73,0xf7,0x00]      
vpsllq $0, %ymm7, %ymm7 

// CHECK: vpsllq $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x73,0xf1,0x00]      
vpsllq $0, %ymm9, %ymm9 

// CHECK: vpsllq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096, %ymm7, %ymm7 

// CHECK: vpsllq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf3,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllq 485498096, %ymm9, %ymm9 

// CHECK: vpsllq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf3,0x7c,0x82,0xc0]      
vpsllq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf3,0x7c,0x82,0x40]      
vpsllq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf3,0x4c,0x82,0xc0]      
vpsllq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf3,0x4c,0x82,0x40]      
vpsllq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf3,0x7c,0x02,0x40]      
vpsllq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsllq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf3,0x4c,0x02,0x40]      
vpsllq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsllq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf3,0x7a,0x40]      
vpsllq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsllq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf3,0x4a,0x40]      
vpsllq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsllq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf3,0x3a]      
vpsllq (%rdx), %ymm7, %ymm7 

// CHECK: vpsllq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf3,0x0a]      
vpsllq (%rdx), %ymm9, %ymm9 

// CHECK: vpsllq %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf3,0xcf]      
vpsllq %xmm15, %ymm9, %ymm9 

// CHECK: vpsllq %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf3,0xfe]      
vpsllq %xmm6, %ymm7, %ymm7 

// CHECK: vpsllvd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x47,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096, %xmm15, %xmm15 

// CHECK: vpsllvd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x47,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096, %xmm6, %xmm6 

// CHECK: vpsllvd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x47,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096, %ymm7, %ymm7 

// CHECK: vpsllvd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x47,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvd 485498096, %ymm9, %ymm9 

// CHECK: vpsllvd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x47,0x7c,0x82,0xc0]      
vpsllvd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllvd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x47,0x7c,0x82,0x40]      
vpsllvd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllvd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x47,0x74,0x82,0xc0]      
vpsllvd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllvd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x47,0x74,0x82,0x40]      
vpsllvd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllvd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x47,0x7c,0x82,0xc0]      
vpsllvd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllvd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x47,0x7c,0x82,0x40]      
vpsllvd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllvd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x47,0x4c,0x82,0xc0]      
vpsllvd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllvd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x47,0x4c,0x82,0x40]      
vpsllvd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllvd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x47,0x7c,0x02,0x40]      
vpsllvd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsllvd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x47,0x74,0x02,0x40]      
vpsllvd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsllvd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x47,0x7c,0x02,0x40]      
vpsllvd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsllvd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x47,0x4c,0x02,0x40]      
vpsllvd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsllvd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x47,0x7a,0x40]      
vpsllvd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsllvd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x47,0x72,0x40]      
vpsllvd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsllvd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x47,0x7a,0x40]      
vpsllvd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsllvd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x47,0x4a,0x40]      
vpsllvd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsllvd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x47,0x3a]      
vpsllvd (%rdx), %xmm15, %xmm15 

// CHECK: vpsllvd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x47,0x32]      
vpsllvd (%rdx), %xmm6, %xmm6 

// CHECK: vpsllvd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x47,0x3a]      
vpsllvd (%rdx), %ymm7, %ymm7 

// CHECK: vpsllvd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x47,0x0a]      
vpsllvd (%rdx), %ymm9, %ymm9 

// CHECK: vpsllvd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x47,0xff]      
vpsllvd %xmm15, %xmm15, %xmm15 

// CHECK: vpsllvd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x47,0xf6]      
vpsllvd %xmm6, %xmm6, %xmm6 

// CHECK: vpsllvd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x47,0xff]      
vpsllvd %ymm7, %ymm7, %ymm7 

// CHECK: vpsllvd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x47,0xc9]      
vpsllvd %ymm9, %ymm9, %ymm9 

// CHECK: vpsllvq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x47,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096, %xmm15, %xmm15 

// CHECK: vpsllvq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x47,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096, %xmm6, %xmm6 

// CHECK: vpsllvq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x47,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096, %ymm7, %ymm7 

// CHECK: vpsllvq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x47,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllvq 485498096, %ymm9, %ymm9 

// CHECK: vpsllvq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x47,0x7c,0x82,0xc0]      
vpsllvq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllvq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x47,0x7c,0x82,0x40]      
vpsllvq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsllvq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x47,0x74,0x82,0xc0]      
vpsllvq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllvq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x47,0x74,0x82,0x40]      
vpsllvq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsllvq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x47,0x7c,0x82,0xc0]      
vpsllvq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllvq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x47,0x7c,0x82,0x40]      
vpsllvq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllvq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x47,0x4c,0x82,0xc0]      
vpsllvq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllvq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x47,0x4c,0x82,0x40]      
vpsllvq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllvq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x47,0x7c,0x02,0x40]      
vpsllvq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsllvq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x47,0x74,0x02,0x40]      
vpsllvq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsllvq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x47,0x7c,0x02,0x40]      
vpsllvq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsllvq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x47,0x4c,0x02,0x40]      
vpsllvq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsllvq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x47,0x7a,0x40]      
vpsllvq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsllvq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x47,0x72,0x40]      
vpsllvq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsllvq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x47,0x7a,0x40]      
vpsllvq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsllvq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x47,0x4a,0x40]      
vpsllvq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsllvq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x47,0x3a]      
vpsllvq (%rdx), %xmm15, %xmm15 

// CHECK: vpsllvq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x47,0x32]      
vpsllvq (%rdx), %xmm6, %xmm6 

// CHECK: vpsllvq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x47,0x3a]      
vpsllvq (%rdx), %ymm7, %ymm7 

// CHECK: vpsllvq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x47,0x0a]      
vpsllvq (%rdx), %ymm9, %ymm9 

// CHECK: vpsllvq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x47,0xff]      
vpsllvq %xmm15, %xmm15, %xmm15 

// CHECK: vpsllvq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x47,0xf6]      
vpsllvq %xmm6, %xmm6, %xmm6 

// CHECK: vpsllvq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x47,0xff]      
vpsllvq %ymm7, %ymm7, %ymm7 

// CHECK: vpsllvq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x47,0xc9]      
vpsllvq %ymm9, %ymm9, %ymm9 

// CHECK: vpsllw $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x71,0xf7,0x00]      
vpsllw $0, %ymm7, %ymm7 

// CHECK: vpsllw $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x71,0xf1,0x00]      
vpsllw $0, %ymm9, %ymm9 

// CHECK: vpsllw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096, %ymm7, %ymm7 

// CHECK: vpsllw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsllw 485498096, %ymm9, %ymm9 

// CHECK: vpsllw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf1,0x7c,0x82,0xc0]      
vpsllw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf1,0x7c,0x82,0x40]      
vpsllw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsllw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf1,0x4c,0x82,0xc0]      
vpsllw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf1,0x4c,0x82,0x40]      
vpsllw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsllw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf1,0x7c,0x02,0x40]      
vpsllw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsllw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf1,0x4c,0x02,0x40]      
vpsllw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsllw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf1,0x7a,0x40]      
vpsllw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsllw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf1,0x4a,0x40]      
vpsllw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsllw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf1,0x3a]      
vpsllw (%rdx), %ymm7, %ymm7 

// CHECK: vpsllw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf1,0x0a]      
vpsllw (%rdx), %ymm9, %ymm9 

// CHECK: vpsllw %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf1,0xcf]      
vpsllw %xmm15, %ymm9, %ymm9 

// CHECK: vpsllw %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf1,0xfe]      
vpsllw %xmm6, %ymm7, %ymm7 

// CHECK: vpsrad $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x72,0xe7,0x00]      
vpsrad $0, %ymm7, %ymm7 

// CHECK: vpsrad $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x72,0xe1,0x00]      
vpsrad $0, %ymm9, %ymm9 

// CHECK: vpsrad 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096, %ymm7, %ymm7 

// CHECK: vpsrad 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe2,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrad 485498096, %ymm9, %ymm9 

// CHECK: vpsrad -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe2,0x7c,0x82,0xc0]      
vpsrad -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrad 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe2,0x7c,0x82,0x40]      
vpsrad 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrad -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe2,0x4c,0x82,0xc0]      
vpsrad -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrad 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe2,0x4c,0x82,0x40]      
vpsrad 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrad 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe2,0x7c,0x02,0x40]      
vpsrad 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsrad 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe2,0x4c,0x02,0x40]      
vpsrad 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsrad 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe2,0x7a,0x40]      
vpsrad 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsrad 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe2,0x4a,0x40]      
vpsrad 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsrad (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe2,0x3a]      
vpsrad (%rdx), %ymm7, %ymm7 

// CHECK: vpsrad (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe2,0x0a]      
vpsrad (%rdx), %ymm9, %ymm9 

// CHECK: vpsrad %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe2,0xcf]      
vpsrad %xmm15, %ymm9, %ymm9 

// CHECK: vpsrad %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe2,0xfe]      
vpsrad %xmm6, %ymm7, %ymm7 

// CHECK: vpsravd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x46,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096, %xmm15, %xmm15 

// CHECK: vpsravd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x46,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096, %xmm6, %xmm6 

// CHECK: vpsravd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x46,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096, %ymm7, %ymm7 

// CHECK: vpsravd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x46,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsravd 485498096, %ymm9, %ymm9 

// CHECK: vpsravd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x46,0x7c,0x82,0xc0]      
vpsravd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsravd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x46,0x7c,0x82,0x40]      
vpsravd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsravd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x46,0x74,0x82,0xc0]      
vpsravd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsravd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x46,0x74,0x82,0x40]      
vpsravd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsravd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x46,0x7c,0x82,0xc0]      
vpsravd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsravd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x46,0x7c,0x82,0x40]      
vpsravd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsravd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x46,0x4c,0x82,0xc0]      
vpsravd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsravd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x46,0x4c,0x82,0x40]      
vpsravd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsravd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x46,0x7c,0x02,0x40]      
vpsravd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsravd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x46,0x74,0x02,0x40]      
vpsravd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsravd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x46,0x7c,0x02,0x40]      
vpsravd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsravd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x46,0x4c,0x02,0x40]      
vpsravd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsravd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x46,0x7a,0x40]      
vpsravd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsravd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x46,0x72,0x40]      
vpsravd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsravd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x46,0x7a,0x40]      
vpsravd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsravd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x46,0x4a,0x40]      
vpsravd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsravd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x46,0x3a]      
vpsravd (%rdx), %xmm15, %xmm15 

// CHECK: vpsravd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x46,0x32]      
vpsravd (%rdx), %xmm6, %xmm6 

// CHECK: vpsravd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x46,0x3a]      
vpsravd (%rdx), %ymm7, %ymm7 

// CHECK: vpsravd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x46,0x0a]      
vpsravd (%rdx), %ymm9, %ymm9 

// CHECK: vpsravd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x46,0xff]      
vpsravd %xmm15, %xmm15, %xmm15 

// CHECK: vpsravd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x46,0xf6]      
vpsravd %xmm6, %xmm6, %xmm6 

// CHECK: vpsravd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x46,0xff]      
vpsravd %ymm7, %ymm7, %ymm7 

// CHECK: vpsravd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x46,0xc9]      
vpsravd %ymm9, %ymm9, %ymm9 

// CHECK: vpsraw $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x71,0xe7,0x00]      
vpsraw $0, %ymm7, %ymm7 

// CHECK: vpsraw $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x71,0xe1,0x00]      
vpsraw $0, %ymm9, %ymm9 

// CHECK: vpsraw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096, %ymm7, %ymm7 

// CHECK: vpsraw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsraw 485498096, %ymm9, %ymm9 

// CHECK: vpsraw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe1,0x7c,0x82,0xc0]      
vpsraw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsraw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe1,0x7c,0x82,0x40]      
vpsraw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsraw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe1,0x4c,0x82,0xc0]      
vpsraw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsraw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe1,0x4c,0x82,0x40]      
vpsraw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsraw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe1,0x7c,0x02,0x40]      
vpsraw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsraw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe1,0x4c,0x02,0x40]      
vpsraw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsraw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe1,0x7a,0x40]      
vpsraw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsraw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe1,0x4a,0x40]      
vpsraw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsraw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe1,0x3a]      
vpsraw (%rdx), %ymm7, %ymm7 

// CHECK: vpsraw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe1,0x0a]      
vpsraw (%rdx), %ymm9, %ymm9 

// CHECK: vpsraw %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe1,0xcf]      
vpsraw %xmm15, %ymm9, %ymm9 

// CHECK: vpsraw %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe1,0xfe]      
vpsraw %xmm6, %ymm7, %ymm7 

// CHECK: vpsrld $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x72,0xd7,0x00]      
vpsrld $0, %ymm7, %ymm7 

// CHECK: vpsrld $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x72,0xd1,0x00]      
vpsrld $0, %ymm9, %ymm9 

// CHECK: vpsrld 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd2,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096, %ymm7, %ymm7 

// CHECK: vpsrld 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd2,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrld 485498096, %ymm9, %ymm9 

// CHECK: vpsrld -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd2,0x7c,0x82,0xc0]      
vpsrld -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrld 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd2,0x7c,0x82,0x40]      
vpsrld 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrld -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd2,0x4c,0x82,0xc0]      
vpsrld -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrld 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd2,0x4c,0x82,0x40]      
vpsrld 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrld 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd2,0x7c,0x02,0x40]      
vpsrld 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsrld 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd2,0x4c,0x02,0x40]      
vpsrld 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsrld 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd2,0x7a,0x40]      
vpsrld 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsrld 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd2,0x4a,0x40]      
vpsrld 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsrldq $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x73,0xdf,0x00]      
vpsrldq $0, %ymm7, %ymm7 

// CHECK: vpsrldq $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x73,0xd9,0x00]      
vpsrldq $0, %ymm9, %ymm9 

// CHECK: vpsrld (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd2,0x3a]      
vpsrld (%rdx), %ymm7, %ymm7 

// CHECK: vpsrld (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd2,0x0a]      
vpsrld (%rdx), %ymm9, %ymm9 

// CHECK: vpsrld %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd2,0xcf]      
vpsrld %xmm15, %ymm9, %ymm9 

// CHECK: vpsrld %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd2,0xfe]      
vpsrld %xmm6, %ymm7, %ymm7 

// CHECK: vpsrlq $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x73,0xd7,0x00]      
vpsrlq $0, %ymm7, %ymm7 

// CHECK: vpsrlq $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x73,0xd1,0x00]      
vpsrlq $0, %ymm9, %ymm9 

// CHECK: vpsrlq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd3,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096, %ymm7, %ymm7 

// CHECK: vpsrlq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd3,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlq 485498096, %ymm9, %ymm9 

// CHECK: vpsrlq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd3,0x7c,0x82,0xc0]      
vpsrlq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd3,0x7c,0x82,0x40]      
vpsrlq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd3,0x4c,0x82,0xc0]      
vpsrlq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd3,0x4c,0x82,0x40]      
vpsrlq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd3,0x7c,0x02,0x40]      
vpsrlq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsrlq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd3,0x4c,0x02,0x40]      
vpsrlq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsrlq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd3,0x7a,0x40]      
vpsrlq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsrlq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd3,0x4a,0x40]      
vpsrlq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsrlq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd3,0x3a]      
vpsrlq (%rdx), %ymm7, %ymm7 

// CHECK: vpsrlq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd3,0x0a]      
vpsrlq (%rdx), %ymm9, %ymm9 

// CHECK: vpsrlq %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd3,0xcf]      
vpsrlq %xmm15, %ymm9, %ymm9 

// CHECK: vpsrlq %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd3,0xfe]      
vpsrlq %xmm6, %ymm7, %ymm7 

// CHECK: vpsrlvd 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x45,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096, %xmm15, %xmm15 

// CHECK: vpsrlvd 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x45,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096, %xmm6, %xmm6 

// CHECK: vpsrlvd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x45,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096, %ymm7, %ymm7 

// CHECK: vpsrlvd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x45,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvd 485498096, %ymm9, %ymm9 

// CHECK: vpsrlvd -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x45,0x7c,0x82,0xc0]      
vpsrlvd -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlvd 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x45,0x7c,0x82,0x40]      
vpsrlvd 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlvd -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x45,0x74,0x82,0xc0]      
vpsrlvd -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlvd 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x45,0x74,0x82,0x40]      
vpsrlvd 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlvd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x45,0x7c,0x82,0xc0]      
vpsrlvd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlvd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x45,0x7c,0x82,0x40]      
vpsrlvd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlvd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x45,0x4c,0x82,0xc0]      
vpsrlvd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlvd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x45,0x4c,0x82,0x40]      
vpsrlvd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlvd 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x45,0x7c,0x02,0x40]      
vpsrlvd 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsrlvd 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x45,0x74,0x02,0x40]      
vpsrlvd 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsrlvd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x45,0x7c,0x02,0x40]      
vpsrlvd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsrlvd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x45,0x4c,0x02,0x40]      
vpsrlvd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsrlvd 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x45,0x7a,0x40]      
vpsrlvd 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsrlvd 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x45,0x72,0x40]      
vpsrlvd 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsrlvd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x45,0x7a,0x40]      
vpsrlvd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsrlvd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x45,0x4a,0x40]      
vpsrlvd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsrlvd (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x01,0x45,0x3a]      
vpsrlvd (%rdx), %xmm15, %xmm15 

// CHECK: vpsrlvd (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x45,0x32]      
vpsrlvd (%rdx), %xmm6, %xmm6 

// CHECK: vpsrlvd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x45,0x3a]      
vpsrlvd (%rdx), %ymm7, %ymm7 

// CHECK: vpsrlvd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0x35,0x45,0x0a]      
vpsrlvd (%rdx), %ymm9, %ymm9 

// CHECK: vpsrlvd %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x01,0x45,0xff]      
vpsrlvd %xmm15, %xmm15, %xmm15 

// CHECK: vpsrlvd %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0x49,0x45,0xf6]      
vpsrlvd %xmm6, %xmm6, %xmm6 

// CHECK: vpsrlvd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0x45,0x45,0xff]      
vpsrlvd %ymm7, %ymm7, %ymm7 

// CHECK: vpsrlvd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0x35,0x45,0xc9]      
vpsrlvd %ymm9, %ymm9, %ymm9 

// CHECK: vpsrlvq 485498096, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x45,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096, %xmm15, %xmm15 

// CHECK: vpsrlvq 485498096, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x45,0x34,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096, %xmm6, %xmm6 

// CHECK: vpsrlvq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x45,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096, %ymm7, %ymm7 

// CHECK: vpsrlvq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x45,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlvq 485498096, %ymm9, %ymm9 

// CHECK: vpsrlvq -64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x45,0x7c,0x82,0xc0]      
vpsrlvq -64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlvq 64(%rdx,%rax,4), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x45,0x7c,0x82,0x40]      
vpsrlvq 64(%rdx,%rax,4), %xmm15, %xmm15 

// CHECK: vpsrlvq -64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x45,0x74,0x82,0xc0]      
vpsrlvq -64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlvq 64(%rdx,%rax,4), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x45,0x74,0x82,0x40]      
vpsrlvq 64(%rdx,%rax,4), %xmm6, %xmm6 

// CHECK: vpsrlvq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x45,0x7c,0x82,0xc0]      
vpsrlvq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlvq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x45,0x7c,0x82,0x40]      
vpsrlvq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlvq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x45,0x4c,0x82,0xc0]      
vpsrlvq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlvq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x45,0x4c,0x82,0x40]      
vpsrlvq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlvq 64(%rdx,%rax), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x45,0x7c,0x02,0x40]      
vpsrlvq 64(%rdx,%rax), %xmm15, %xmm15 

// CHECK: vpsrlvq 64(%rdx,%rax), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x45,0x74,0x02,0x40]      
vpsrlvq 64(%rdx,%rax), %xmm6, %xmm6 

// CHECK: vpsrlvq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x45,0x7c,0x02,0x40]      
vpsrlvq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsrlvq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x45,0x4c,0x02,0x40]      
vpsrlvq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsrlvq 64(%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x45,0x7a,0x40]      
vpsrlvq 64(%rdx), %xmm15, %xmm15 

// CHECK: vpsrlvq 64(%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x45,0x72,0x40]      
vpsrlvq 64(%rdx), %xmm6, %xmm6 

// CHECK: vpsrlvq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x45,0x7a,0x40]      
vpsrlvq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsrlvq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x45,0x4a,0x40]      
vpsrlvq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsrlvq (%rdx), %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x62,0x81,0x45,0x3a]      
vpsrlvq (%rdx), %xmm15, %xmm15 

// CHECK: vpsrlvq (%rdx), %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x45,0x32]      
vpsrlvq (%rdx), %xmm6, %xmm6 

// CHECK: vpsrlvq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x45,0x3a]      
vpsrlvq (%rdx), %ymm7, %ymm7 

// CHECK: vpsrlvq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x62,0xb5,0x45,0x0a]      
vpsrlvq (%rdx), %ymm9, %ymm9 

// CHECK: vpsrlvq %xmm15, %xmm15, %xmm15 
// CHECK: encoding: [0xc4,0x42,0x81,0x45,0xff]      
vpsrlvq %xmm15, %xmm15, %xmm15 

// CHECK: vpsrlvq %xmm6, %xmm6, %xmm6 
// CHECK: encoding: [0xc4,0xe2,0xc9,0x45,0xf6]      
vpsrlvq %xmm6, %xmm6, %xmm6 

// CHECK: vpsrlvq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc4,0xe2,0xc5,0x45,0xff]      
vpsrlvq %ymm7, %ymm7, %ymm7 

// CHECK: vpsrlvq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x42,0xb5,0x45,0xc9]      
vpsrlvq %ymm9, %ymm9, %ymm9 

// CHECK: vpsrlw $0, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x71,0xd7,0x00]      
vpsrlw $0, %ymm7, %ymm7 

// CHECK: vpsrlw $0, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0xc1,0x35,0x71,0xd1,0x00]      
vpsrlw $0, %ymm9, %ymm9 

// CHECK: vpsrlw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd1,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096, %ymm7, %ymm7 

// CHECK: vpsrlw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd1,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsrlw 485498096, %ymm9, %ymm9 

// CHECK: vpsrlw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd1,0x7c,0x82,0xc0]      
vpsrlw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd1,0x7c,0x82,0x40]      
vpsrlw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsrlw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd1,0x4c,0x82,0xc0]      
vpsrlw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd1,0x4c,0x82,0x40]      
vpsrlw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsrlw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd1,0x7c,0x02,0x40]      
vpsrlw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsrlw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd1,0x4c,0x02,0x40]      
vpsrlw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsrlw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd1,0x7a,0x40]      
vpsrlw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsrlw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd1,0x4a,0x40]      
vpsrlw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsrlw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd1,0x3a]      
vpsrlw (%rdx), %ymm7, %ymm7 

// CHECK: vpsrlw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd1,0x0a]      
vpsrlw (%rdx), %ymm9, %ymm9 

// CHECK: vpsrlw %xmm15, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd1,0xcf]      
vpsrlw %xmm15, %ymm9, %ymm9 

// CHECK: vpsrlw %xmm6, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd1,0xfe]      
vpsrlw %xmm6, %ymm7, %ymm7 

// CHECK: vpsubb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096, %ymm7, %ymm7 

// CHECK: vpsubb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubb 485498096, %ymm9, %ymm9 

// CHECK: vpsubb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf8,0x7c,0x82,0xc0]      
vpsubb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf8,0x7c,0x82,0x40]      
vpsubb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf8,0x4c,0x82,0xc0]      
vpsubb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf8,0x4c,0x82,0x40]      
vpsubb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf8,0x7c,0x02,0x40]      
vpsubb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf8,0x4c,0x02,0x40]      
vpsubb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf8,0x7a,0x40]      
vpsubb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf8,0x4a,0x40]      
vpsubb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf8,0x3a]      
vpsubb (%rdx), %ymm7, %ymm7 

// CHECK: vpsubb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf8,0x0a]      
vpsubb (%rdx), %ymm9, %ymm9 

// CHECK: vpsubb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf8,0xff]      
vpsubb %ymm7, %ymm7, %ymm7 

// CHECK: vpsubb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf8,0xc9]      
vpsubb %ymm9, %ymm9, %ymm9 

// CHECK: vpsubd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfa,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096, %ymm7, %ymm7 

// CHECK: vpsubd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfa,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubd 485498096, %ymm9, %ymm9 

// CHECK: vpsubd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfa,0x7c,0x82,0xc0]      
vpsubd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfa,0x7c,0x82,0x40]      
vpsubd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfa,0x4c,0x82,0xc0]      
vpsubd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfa,0x4c,0x82,0x40]      
vpsubd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfa,0x7c,0x02,0x40]      
vpsubd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfa,0x4c,0x02,0x40]      
vpsubd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfa,0x7a,0x40]      
vpsubd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfa,0x4a,0x40]      
vpsubd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfa,0x3a]      
vpsubd (%rdx), %ymm7, %ymm7 

// CHECK: vpsubd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfa,0x0a]      
vpsubd (%rdx), %ymm9, %ymm9 

// CHECK: vpsubd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfa,0xff]      
vpsubd %ymm7, %ymm7, %ymm7 

// CHECK: vpsubd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xfa,0xc9]      
vpsubd %ymm9, %ymm9, %ymm9 

// CHECK: vpsubq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096, %ymm7, %ymm7 

// CHECK: vpsubq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubq 485498096, %ymm9, %ymm9 

// CHECK: vpsubq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfb,0x7c,0x82,0xc0]      
vpsubq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfb,0x7c,0x82,0x40]      
vpsubq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfb,0x4c,0x82,0xc0]      
vpsubq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfb,0x4c,0x82,0x40]      
vpsubq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfb,0x7c,0x02,0x40]      
vpsubq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfb,0x4c,0x02,0x40]      
vpsubq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfb,0x7a,0x40]      
vpsubq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfb,0x4a,0x40]      
vpsubq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfb,0x3a]      
vpsubq (%rdx), %ymm7, %ymm7 

// CHECK: vpsubq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xfb,0x0a]      
vpsubq (%rdx), %ymm9, %ymm9 

// CHECK: vpsubq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xfb,0xff]      
vpsubq %ymm7, %ymm7, %ymm7 

// CHECK: vpsubq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xfb,0xc9]      
vpsubq %ymm9, %ymm9, %ymm9 

// CHECK: vpsubsb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096, %ymm7, %ymm7 

// CHECK: vpsubsb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsb 485498096, %ymm9, %ymm9 

// CHECK: vpsubsb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe8,0x7c,0x82,0xc0]      
vpsubsb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubsb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe8,0x7c,0x82,0x40]      
vpsubsb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubsb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe8,0x4c,0x82,0xc0]      
vpsubsb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubsb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe8,0x4c,0x82,0x40]      
vpsubsb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubsb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe8,0x7c,0x02,0x40]      
vpsubsb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubsb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe8,0x4c,0x02,0x40]      
vpsubsb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubsb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe8,0x7a,0x40]      
vpsubsb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubsb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe8,0x4a,0x40]      
vpsubsb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubsb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe8,0x3a]      
vpsubsb (%rdx), %ymm7, %ymm7 

// CHECK: vpsubsb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe8,0x0a]      
vpsubsb (%rdx), %ymm9, %ymm9 

// CHECK: vpsubsb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe8,0xff]      
vpsubsb %ymm7, %ymm7, %ymm7 

// CHECK: vpsubsb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe8,0xc9]      
vpsubsb %ymm9, %ymm9, %ymm9 

// CHECK: vpsubsw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096, %ymm7, %ymm7 

// CHECK: vpsubsw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubsw 485498096, %ymm9, %ymm9 

// CHECK: vpsubsw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe9,0x7c,0x82,0xc0]      
vpsubsw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubsw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe9,0x7c,0x82,0x40]      
vpsubsw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubsw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe9,0x4c,0x82,0xc0]      
vpsubsw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubsw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe9,0x4c,0x82,0x40]      
vpsubsw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubsw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe9,0x7c,0x02,0x40]      
vpsubsw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubsw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe9,0x4c,0x02,0x40]      
vpsubsw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubsw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe9,0x7a,0x40]      
vpsubsw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubsw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe9,0x4a,0x40]      
vpsubsw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubsw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe9,0x3a]      
vpsubsw (%rdx), %ymm7, %ymm7 

// CHECK: vpsubsw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xe9,0x0a]      
vpsubsw (%rdx), %ymm9, %ymm9 

// CHECK: vpsubsw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xe9,0xff]      
vpsubsw %ymm7, %ymm7, %ymm7 

// CHECK: vpsubsw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xe9,0xc9]      
vpsubsw %ymm9, %ymm9, %ymm9 

// CHECK: vpsubusb 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd8,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096, %ymm7, %ymm7 

// CHECK: vpsubusb 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd8,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusb 485498096, %ymm9, %ymm9 

// CHECK: vpsubusb -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd8,0x7c,0x82,0xc0]      
vpsubusb -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubusb 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd8,0x7c,0x82,0x40]      
vpsubusb 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubusb -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd8,0x4c,0x82,0xc0]      
vpsubusb -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubusb 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd8,0x4c,0x82,0x40]      
vpsubusb 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubusb 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd8,0x7c,0x02,0x40]      
vpsubusb 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubusb 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd8,0x4c,0x02,0x40]      
vpsubusb 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubusb 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd8,0x7a,0x40]      
vpsubusb 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubusb 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd8,0x4a,0x40]      
vpsubusb 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubusb (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd8,0x3a]      
vpsubusb (%rdx), %ymm7, %ymm7 

// CHECK: vpsubusb (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd8,0x0a]      
vpsubusb (%rdx), %ymm9, %ymm9 

// CHECK: vpsubusb %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd8,0xff]      
vpsubusb %ymm7, %ymm7, %ymm7 

// CHECK: vpsubusb %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd8,0xc9]      
vpsubusb %ymm9, %ymm9, %ymm9 

// CHECK: vpsubusw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096, %ymm7, %ymm7 

// CHECK: vpsubusw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubusw 485498096, %ymm9, %ymm9 

// CHECK: vpsubusw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd9,0x7c,0x82,0xc0]      
vpsubusw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubusw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd9,0x7c,0x82,0x40]      
vpsubusw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubusw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd9,0x4c,0x82,0xc0]      
vpsubusw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubusw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd9,0x4c,0x82,0x40]      
vpsubusw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubusw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd9,0x7c,0x02,0x40]      
vpsubusw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubusw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd9,0x4c,0x02,0x40]      
vpsubusw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubusw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd9,0x7a,0x40]      
vpsubusw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubusw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd9,0x4a,0x40]      
vpsubusw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubusw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd9,0x3a]      
vpsubusw (%rdx), %ymm7, %ymm7 

// CHECK: vpsubusw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xd9,0x0a]      
vpsubusw (%rdx), %ymm9, %ymm9 

// CHECK: vpsubusw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xd9,0xff]      
vpsubusw %ymm7, %ymm7, %ymm7 

// CHECK: vpsubusw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xd9,0xc9]      
vpsubusw %ymm9, %ymm9, %ymm9 

// CHECK: vpsubw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096, %ymm7, %ymm7 

// CHECK: vpsubw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpsubw 485498096, %ymm9, %ymm9 

// CHECK: vpsubw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf9,0x7c,0x82,0xc0]      
vpsubw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf9,0x7c,0x82,0x40]      
vpsubw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpsubw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf9,0x4c,0x82,0xc0]      
vpsubw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf9,0x4c,0x82,0x40]      
vpsubw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpsubw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf9,0x7c,0x02,0x40]      
vpsubw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpsubw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf9,0x4c,0x02,0x40]      
vpsubw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpsubw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf9,0x7a,0x40]      
vpsubw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpsubw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf9,0x4a,0x40]      
vpsubw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpsubw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf9,0x3a]      
vpsubw (%rdx), %ymm7, %ymm7 

// CHECK: vpsubw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xf9,0x0a]      
vpsubw (%rdx), %ymm9, %ymm9 

// CHECK: vpsubw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xf9,0xff]      
vpsubw %ymm7, %ymm7, %ymm7 

// CHECK: vpsubw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xf9,0xc9]      
vpsubw %ymm9, %ymm9, %ymm9 

// CHECK: vpunpckhbw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x68,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096, %ymm7, %ymm7 

// CHECK: vpunpckhbw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x68,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhbw 485498096, %ymm9, %ymm9 

// CHECK: vpunpckhbw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x68,0x7c,0x82,0xc0]      
vpunpckhbw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhbw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x68,0x7c,0x82,0x40]      
vpunpckhbw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhbw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x68,0x4c,0x82,0xc0]      
vpunpckhbw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhbw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x68,0x4c,0x82,0x40]      
vpunpckhbw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhbw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x68,0x7c,0x02,0x40]      
vpunpckhbw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpckhbw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x68,0x4c,0x02,0x40]      
vpunpckhbw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpckhbw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x68,0x7a,0x40]      
vpunpckhbw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhbw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x68,0x4a,0x40]      
vpunpckhbw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhbw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x68,0x3a]      
vpunpckhbw (%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhbw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x68,0x0a]      
vpunpckhbw (%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhbw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x68,0xff]      
vpunpckhbw %ymm7, %ymm7, %ymm7 

// CHECK: vpunpckhbw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x68,0xc9]      
vpunpckhbw %ymm9, %ymm9, %ymm9 

// CHECK: vpunpckhdq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096, %ymm7, %ymm7 

// CHECK: vpunpckhdq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhdq 485498096, %ymm9, %ymm9 

// CHECK: vpunpckhdq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6a,0x7c,0x82,0xc0]      
vpunpckhdq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhdq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6a,0x7c,0x82,0x40]      
vpunpckhdq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhdq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6a,0x4c,0x82,0xc0]      
vpunpckhdq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhdq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6a,0x4c,0x82,0x40]      
vpunpckhdq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhdq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6a,0x7c,0x02,0x40]      
vpunpckhdq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpckhdq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6a,0x4c,0x02,0x40]      
vpunpckhdq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpckhdq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6a,0x7a,0x40]      
vpunpckhdq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhdq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6a,0x4a,0x40]      
vpunpckhdq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhdq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6a,0x3a]      
vpunpckhdq (%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhdq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6a,0x0a]      
vpunpckhdq (%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhdq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6a,0xff]      
vpunpckhdq %ymm7, %ymm7, %ymm7 

// CHECK: vpunpckhdq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x6a,0xc9]      
vpunpckhdq %ymm9, %ymm9, %ymm9 

// CHECK: vpunpckhqdq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096, %ymm7, %ymm7 

// CHECK: vpunpckhqdq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhqdq 485498096, %ymm9, %ymm9 

// CHECK: vpunpckhqdq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6d,0x7c,0x82,0xc0]      
vpunpckhqdq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhqdq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6d,0x7c,0x82,0x40]      
vpunpckhqdq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhqdq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6d,0x4c,0x82,0xc0]      
vpunpckhqdq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhqdq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6d,0x4c,0x82,0x40]      
vpunpckhqdq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhqdq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6d,0x7c,0x02,0x40]      
vpunpckhqdq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpckhqdq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6d,0x4c,0x02,0x40]      
vpunpckhqdq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpckhqdq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6d,0x7a,0x40]      
vpunpckhqdq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhqdq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6d,0x4a,0x40]      
vpunpckhqdq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhqdq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6d,0x3a]      
vpunpckhqdq (%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhqdq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6d,0x0a]      
vpunpckhqdq (%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhqdq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6d,0xff]      
vpunpckhqdq %ymm7, %ymm7, %ymm7 

// CHECK: vpunpckhqdq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x6d,0xc9]      
vpunpckhqdq %ymm9, %ymm9, %ymm9 

// CHECK: vpunpckhwd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x69,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096, %ymm7, %ymm7 

// CHECK: vpunpckhwd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x69,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckhwd 485498096, %ymm9, %ymm9 

// CHECK: vpunpckhwd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x69,0x7c,0x82,0xc0]      
vpunpckhwd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhwd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x69,0x7c,0x82,0x40]      
vpunpckhwd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckhwd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x69,0x4c,0x82,0xc0]      
vpunpckhwd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhwd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x69,0x4c,0x82,0x40]      
vpunpckhwd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckhwd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x69,0x7c,0x02,0x40]      
vpunpckhwd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpckhwd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x69,0x4c,0x02,0x40]      
vpunpckhwd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpckhwd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x69,0x7a,0x40]      
vpunpckhwd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhwd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x69,0x4a,0x40]      
vpunpckhwd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhwd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x69,0x3a]      
vpunpckhwd (%rdx), %ymm7, %ymm7 

// CHECK: vpunpckhwd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x69,0x0a]      
vpunpckhwd (%rdx), %ymm9, %ymm9 

// CHECK: vpunpckhwd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x69,0xff]      
vpunpckhwd %ymm7, %ymm7, %ymm7 

// CHECK: vpunpckhwd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x69,0xc9]      
vpunpckhwd %ymm9, %ymm9, %ymm9 

// CHECK: vpunpcklbw 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x60,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096, %ymm7, %ymm7 

// CHECK: vpunpcklbw 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x60,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklbw 485498096, %ymm9, %ymm9 

// CHECK: vpunpcklbw -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x60,0x7c,0x82,0xc0]      
vpunpcklbw -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpcklbw 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x60,0x7c,0x82,0x40]      
vpunpcklbw 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpcklbw -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x60,0x4c,0x82,0xc0]      
vpunpcklbw -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpcklbw 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x60,0x4c,0x82,0x40]      
vpunpcklbw 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpcklbw 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x60,0x7c,0x02,0x40]      
vpunpcklbw 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpcklbw 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x60,0x4c,0x02,0x40]      
vpunpcklbw 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpcklbw 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x60,0x7a,0x40]      
vpunpcklbw 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpcklbw 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x60,0x4a,0x40]      
vpunpcklbw 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpcklbw (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x60,0x3a]      
vpunpcklbw (%rdx), %ymm7, %ymm7 

// CHECK: vpunpcklbw (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x60,0x0a]      
vpunpcklbw (%rdx), %ymm9, %ymm9 

// CHECK: vpunpcklbw %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x60,0xff]      
vpunpcklbw %ymm7, %ymm7, %ymm7 

// CHECK: vpunpcklbw %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x60,0xc9]      
vpunpcklbw %ymm9, %ymm9, %ymm9 

// CHECK: vpunpckldq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x62,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096, %ymm7, %ymm7 

// CHECK: vpunpckldq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x62,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpckldq 485498096, %ymm9, %ymm9 

// CHECK: vpunpckldq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x62,0x7c,0x82,0xc0]      
vpunpckldq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckldq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x62,0x7c,0x82,0x40]      
vpunpckldq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpckldq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x62,0x4c,0x82,0xc0]      
vpunpckldq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckldq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x62,0x4c,0x82,0x40]      
vpunpckldq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpckldq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x62,0x7c,0x02,0x40]      
vpunpckldq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpckldq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x62,0x4c,0x02,0x40]      
vpunpckldq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpckldq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x62,0x7a,0x40]      
vpunpckldq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpckldq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x62,0x4a,0x40]      
vpunpckldq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpckldq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x62,0x3a]      
vpunpckldq (%rdx), %ymm7, %ymm7 

// CHECK: vpunpckldq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x62,0x0a]      
vpunpckldq (%rdx), %ymm9, %ymm9 

// CHECK: vpunpckldq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x62,0xff]      
vpunpckldq %ymm7, %ymm7, %ymm7 

// CHECK: vpunpckldq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x62,0xc9]      
vpunpckldq %ymm9, %ymm9, %ymm9 

// CHECK: vpunpcklqdq 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096, %ymm7, %ymm7 

// CHECK: vpunpcklqdq 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklqdq 485498096, %ymm9, %ymm9 

// CHECK: vpunpcklqdq -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6c,0x7c,0x82,0xc0]      
vpunpcklqdq -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpcklqdq 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6c,0x7c,0x82,0x40]      
vpunpcklqdq 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpcklqdq -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6c,0x4c,0x82,0xc0]      
vpunpcklqdq -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpcklqdq 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6c,0x4c,0x82,0x40]      
vpunpcklqdq 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpcklqdq 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6c,0x7c,0x02,0x40]      
vpunpcklqdq 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpcklqdq 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6c,0x4c,0x02,0x40]      
vpunpcklqdq 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpcklqdq 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6c,0x7a,0x40]      
vpunpcklqdq 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpcklqdq 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6c,0x4a,0x40]      
vpunpcklqdq 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpcklqdq (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6c,0x3a]      
vpunpcklqdq (%rdx), %ymm7, %ymm7 

// CHECK: vpunpcklqdq (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x6c,0x0a]      
vpunpcklqdq (%rdx), %ymm9, %ymm9 

// CHECK: vpunpcklqdq %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x6c,0xff]      
vpunpcklqdq %ymm7, %ymm7, %ymm7 

// CHECK: vpunpcklqdq %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x6c,0xc9]      
vpunpcklqdq %ymm9, %ymm9, %ymm9 

// CHECK: vpunpcklwd 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x61,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096, %ymm7, %ymm7 

// CHECK: vpunpcklwd 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x61,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpunpcklwd 485498096, %ymm9, %ymm9 

// CHECK: vpunpcklwd -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x61,0x7c,0x82,0xc0]      
vpunpcklwd -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpcklwd 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x61,0x7c,0x82,0x40]      
vpunpcklwd 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpunpcklwd -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x61,0x4c,0x82,0xc0]      
vpunpcklwd -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpcklwd 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x61,0x4c,0x82,0x40]      
vpunpcklwd 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpunpcklwd 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x61,0x7c,0x02,0x40]      
vpunpcklwd 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpunpcklwd 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x61,0x4c,0x02,0x40]      
vpunpcklwd 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpunpcklwd 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x61,0x7a,0x40]      
vpunpcklwd 64(%rdx), %ymm7, %ymm7 

// CHECK: vpunpcklwd 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x61,0x4a,0x40]      
vpunpcklwd 64(%rdx), %ymm9, %ymm9 

// CHECK: vpunpcklwd (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x61,0x3a]      
vpunpcklwd (%rdx), %ymm7, %ymm7 

// CHECK: vpunpcklwd (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0x61,0x0a]      
vpunpcklwd (%rdx), %ymm9, %ymm9 

// CHECK: vpunpcklwd %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0x61,0xff]      
vpunpcklwd %ymm7, %ymm7, %ymm7 

// CHECK: vpunpcklwd %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0x61,0xc9]      
vpunpcklwd %ymm9, %ymm9, %ymm9 

// CHECK: vpxor 485498096, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xef,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096, %ymm7, %ymm7 

// CHECK: vpxor 485498096, %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xef,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]      
vpxor 485498096, %ymm9, %ymm9 

// CHECK: vpxor -64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xef,0x7c,0x82,0xc0]      
vpxor -64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpxor 64(%rdx,%rax,4), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xef,0x7c,0x82,0x40]      
vpxor 64(%rdx,%rax,4), %ymm7, %ymm7 

// CHECK: vpxor -64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xef,0x4c,0x82,0xc0]      
vpxor -64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpxor 64(%rdx,%rax,4), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xef,0x4c,0x82,0x40]      
vpxor 64(%rdx,%rax,4), %ymm9, %ymm9 

// CHECK: vpxor 64(%rdx,%rax), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xef,0x7c,0x02,0x40]      
vpxor 64(%rdx,%rax), %ymm7, %ymm7 

// CHECK: vpxor 64(%rdx,%rax), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xef,0x4c,0x02,0x40]      
vpxor 64(%rdx,%rax), %ymm9, %ymm9 

// CHECK: vpxor 64(%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xef,0x7a,0x40]      
vpxor 64(%rdx), %ymm7, %ymm7 

// CHECK: vpxor 64(%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xef,0x4a,0x40]      
vpxor 64(%rdx), %ymm9, %ymm9 

// CHECK: vpxor (%rdx), %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xef,0x3a]      
vpxor (%rdx), %ymm7, %ymm7 

// CHECK: vpxor (%rdx), %ymm9, %ymm9 
// CHECK: encoding: [0xc5,0x35,0xef,0x0a]      
vpxor (%rdx), %ymm9, %ymm9 

// CHECK: vpxor %ymm7, %ymm7, %ymm7 
// CHECK: encoding: [0xc5,0xc5,0xef,0xff]      
vpxor %ymm7, %ymm7, %ymm7 

// CHECK: vpxor %ymm9, %ymm9, %ymm9 
// CHECK: encoding: [0xc4,0x41,0x35,0xef,0xc9]      
vpxor %ymm9, %ymm9, %ymm9 

