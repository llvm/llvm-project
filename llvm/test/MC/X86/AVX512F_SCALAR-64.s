// RUN: llvm-mc -triple x86_64-unknown-unknown -mcpu=skx --show-encoding %s | FileCheck %s

// CHECK: vaddsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm15, %xmm15

// CHECK: vaddsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vaddsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm1, %xmm1

// CHECK: vaddsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x58,0xbc,0x82,0x00,0x02,0x00,0x00]
vaddsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vaddsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x58,0xbc,0x82,0x00,0xfe,0xff,0xff]
vaddsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vaddsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x58,0x7c,0x82,0x40]
vaddsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vaddsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x58,0x7c,0x82,0xc0]
vaddsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vaddsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x58,0x7c,0x82,0x40]
vaddsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x58,0x7c,0x82,0xc0]
vaddsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x58,0x8c,0x82,0x00,0x02,0x00,0x00]
vaddsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vaddsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x58,0x8c,0x82,0x00,0xfe,0xff,0xff]
vaddsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vaddsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x4c,0x82,0x40]
vaddsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x4c,0x82,0xc0]
vaddsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x4c,0x82,0x40]
vaddsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x4c,0x82,0xc0]
vaddsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x58,0xbc,0x02,0x00,0x02,0x00,0x00]
vaddsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vaddsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x58,0x7c,0x02,0x40]
vaddsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vaddsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x58,0x7c,0x02,0x40]
vaddsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x58,0x8c,0x02,0x00,0x02,0x00,0x00]
vaddsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vaddsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x4c,0x02,0x40]
vaddsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x4c,0x02,0x40]
vaddsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x58,0xba,0x00,0x02,0x00,0x00]
vaddsd 512(%rdx), %xmm15, %xmm15

// CHECK: vaddsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x58,0x7a,0x40]
vaddsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vaddsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x58,0x7a,0x40]
vaddsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x58,0x8a,0x00,0x02,0x00,0x00]
vaddsd 512(%rdx), %xmm1, %xmm1

// CHECK: vaddsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x4a,0x40]
vaddsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x4a,0x40]
vaddsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x58,0xff]
vaddsd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x3a,0x58,0xff]
vaddsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xba,0x58,0xff]
vaddsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x58,0xc9]
vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x58,0xc9]
vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x58,0xc9]
vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x58,0x3a]
vaddsd (%rdx), %xmm15, %xmm15

// CHECK: vaddsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x58,0x3a]
vaddsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vaddsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x58,0x3a]
vaddsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x58,0x0a]
vaddsd (%rdx), %xmm1, %xmm1

// CHECK: vaddsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x0a]
vaddsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x0a]
vaddsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x58,0xff]
vaddsd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x58,0xff]
vaddsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x58,0xff]
vaddsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x58,0xc9]
vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x58,0xc9]
vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x58,0xc9]
vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x58,0xff]
vaddsd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x5a,0x58,0xff]
vaddsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xda,0x58,0xff]
vaddsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x58,0xc9]
vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x58,0xc9]
vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x58,0xc9]
vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x58,0xff]
vaddsd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x7a,0x58,0xff]
vaddsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xfa,0x58,0xff]
vaddsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x58,0xc9]
vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x58,0xc9]
vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x58,0xc9]
vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x58,0xff]
vaddsd %xmm15, %xmm15, %xmm15

// CHECK: vaddsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x58,0xff]
vaddsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x58,0xff]
vaddsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x58,0xc9]
vaddsd %xmm1, %xmm1, %xmm1

// CHECK: vaddsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0xc9]
vaddsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0xc9]
vaddsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x58,0xbc,0x82,0x00,0x01,0x00,0x00]
vaddss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vaddss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x58,0xbc,0x82,0x00,0xff,0xff,0xff]
vaddss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vaddss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x58,0x7c,0x82,0x40]
vaddss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vaddss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x58,0x7c,0x82,0xc0]
vaddss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vaddss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x58,0x7c,0x82,0x40]
vaddss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x58,0x7c,0x82,0xc0]
vaddss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x58,0x8c,0x82,0x00,0x01,0x00,0x00]
vaddss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vaddss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x58,0x8c,0x82,0x00,0xff,0xff,0xff]
vaddss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vaddss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x4c,0x82,0x40]
vaddss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x4c,0x82,0xc0]
vaddss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x4c,0x82,0x40]
vaddss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x4c,0x82,0xc0]
vaddss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x58,0xbc,0x02,0x00,0x01,0x00,0x00]
vaddss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vaddss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x58,0x7c,0x02,0x40]
vaddss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vaddss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x58,0x7c,0x02,0x40]
vaddss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x58,0x8c,0x02,0x00,0x01,0x00,0x00]
vaddss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vaddss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x4c,0x02,0x40]
vaddss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vaddss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x4c,0x02,0x40]
vaddss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x58,0xba,0x00,0x01,0x00,0x00]
vaddss 256(%rdx), %xmm15, %xmm15

// CHECK: vaddss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x58,0x7a,0x40]
vaddss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vaddss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x58,0x7a,0x40]
vaddss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x58,0x8a,0x00,0x01,0x00,0x00]
vaddss 256(%rdx), %xmm1, %xmm1

// CHECK: vaddss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x4a,0x40]
vaddss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vaddss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x4a,0x40]
vaddss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm15, %xmm15

// CHECK: vaddss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vaddss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x58,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm1, %xmm1

// CHECK: vaddss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vaddss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x38,0x58,0xff]
vaddss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x3a,0x58,0xff]
vaddss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xba,0x58,0xff]
vaddss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x58,0xc9]
vaddss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x58,0xc9]
vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x58,0xc9]
vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x58,0x3a]
vaddss (%rdx), %xmm15, %xmm15

// CHECK: vaddss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x58,0x3a]
vaddss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vaddss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x58,0x3a]
vaddss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x58,0x0a]
vaddss (%rdx), %xmm1, %xmm1

// CHECK: vaddss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x0a]
vaddss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vaddss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x0a]
vaddss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x58,0xff]
vaddss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x58,0xff]
vaddss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x58,0xff]
vaddss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x58,0xc9]
vaddss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x58,0xc9]
vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x58,0xc9]
vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x58,0x58,0xff]
vaddss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x5a,0x58,0xff]
vaddss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xda,0x58,0xff]
vaddss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x58,0xc9]
vaddss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x58,0xc9]
vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x58,0xc9]
vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x78,0x58,0xff]
vaddss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vaddss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x7a,0x58,0xff]
vaddss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xfa,0x58,0xff]
vaddss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x58,0xc9]
vaddss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x58,0xc9]
vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x58,0xc9]
vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x58,0xff]
vaddss %xmm15, %xmm15, %xmm15

// CHECK: vaddss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x58,0xff]
vaddss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vaddss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x58,0xff]
vaddss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vaddss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x58,0xc9]
vaddss %xmm1, %xmm1, %xmm1

// CHECK: vaddss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0xc9]
vaddss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0xc9]
vaddss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcmpeqsd 485498096, %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x87,0x08,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096, %xmm15, %k2

// CHECK: vcmpeqsd 485498096, %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x87,0x0a,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096, %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd 485498096, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096, %xmm1, %k2

// CHECK: vcmpeqsd 485498096, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096, %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd 512(%rdx,%rax,4), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x87,0x08,0xc2,0x54,0x82,0x40,0x00]
vcmpeqsd 512(%rdx,%rax,4), %xmm15, %k2

// CHECK: vcmpeqsd -512(%rdx,%rax,4), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x87,0x08,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqsd -512(%rdx,%rax,4), %xmm15, %k2

// CHECK: vcmpeqsd 512(%rdx,%rax,4), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x87,0x0a,0xc2,0x54,0x82,0x40,0x00]
vcmpeqsd 512(%rdx,%rax,4), %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd -512(%rdx,%rax,4), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x87,0x0a,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqsd -512(%rdx,%rax,4), %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd 512(%rdx,%rax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x54,0x82,0x40,0x00]
vcmpeqsd 512(%rdx,%rax,4), %xmm1, %k2

// CHECK: vcmpeqsd -512(%rdx,%rax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqsd -512(%rdx,%rax,4), %xmm1, %k2

// CHECK: vcmpeqsd 512(%rdx,%rax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x54,0x82,0x40,0x00]
vcmpeqsd 512(%rdx,%rax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd -512(%rdx,%rax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqsd -512(%rdx,%rax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd 512(%rdx,%rax), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x87,0x08,0xc2,0x54,0x02,0x40,0x00]
vcmpeqsd 512(%rdx,%rax), %xmm15, %k2

// CHECK: vcmpeqsd 512(%rdx,%rax), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x87,0x0a,0xc2,0x54,0x02,0x40,0x00]
vcmpeqsd 512(%rdx,%rax), %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd 512(%rdx,%rax), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x54,0x02,0x40,0x00]
vcmpeqsd 512(%rdx,%rax), %xmm1, %k2

// CHECK: vcmpeqsd 512(%rdx,%rax), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x54,0x02,0x40,0x00]
vcmpeqsd 512(%rdx,%rax), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd 512(%rdx), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x87,0x08,0xc2,0x52,0x40,0x00]
vcmpeqsd 512(%rdx), %xmm15, %k2

// CHECK: vcmpeqsd 512(%rdx), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x87,0x0a,0xc2,0x52,0x40,0x00]
vcmpeqsd 512(%rdx), %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd 512(%rdx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x52,0x40,0x00]
vcmpeqsd 512(%rdx), %xmm1, %k2

// CHECK: vcmpeqsd 512(%rdx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x52,0x40,0x00]
vcmpeqsd 512(%rdx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd (%rdx), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x87,0x08,0xc2,0x12,0x00]
vcmpeqsd (%rdx), %xmm15, %k2

// CHECK: vcmpeqsd (%rdx), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x87,0x0a,0xc2,0x12,0x00]
vcmpeqsd (%rdx), %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd (%rdx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x12,0x00]
vcmpeqsd (%rdx), %xmm1, %k2

// CHECK: vcmpeqsd (%rdx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x12,0x00]
vcmpeqsd (%rdx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd {sae}, %xmm15, %xmm15, %k2
// CHECK: encoding: [0x62,0xd1,0x87,0x18,0xc2,0xd7,0x00]
vcmpeqsd {sae}, %xmm15, %xmm15, %k2

// CHECK: vcmpeqsd {sae}, %xmm15, %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xd1,0x87,0x1a,0xc2,0xd7,0x00]
vcmpeqsd {sae}, %xmm15, %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd {sae}, %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0xc2,0xd1,0x00]
vcmpeqsd {sae}, %xmm1, %xmm1, %k2

// CHECK: vcmpeqsd {sae}, %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0xc2,0xd1,0x00]
vcmpeqsd {sae}, %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd %xmm15, %xmm15, %k2
// CHECK: encoding: [0x62,0xd1,0x87,0x08,0xc2,0xd7,0x00]
vcmpeqsd %xmm15, %xmm15, %k2

// CHECK: vcmpeqsd %xmm15, %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xd1,0x87,0x0a,0xc2,0xd7,0x00]
vcmpeqsd %xmm15, %xmm15, %k2 {%k2}

// CHECK: vcmpeqsd %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0xd1,0x00]
vcmpeqsd %xmm1, %xmm1, %k2

// CHECK: vcmpeqsd %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0xd1,0x00]
vcmpeqsd %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 256(%rdx,%rax,4), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x06,0x08,0xc2,0x54,0x82,0x40,0x00]
vcmpeqss 256(%rdx,%rax,4), %xmm15, %k2

// CHECK: vcmpeqss -256(%rdx,%rax,4), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x06,0x08,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqss -256(%rdx,%rax,4), %xmm15, %k2

// CHECK: vcmpeqss 256(%rdx,%rax,4), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x06,0x0a,0xc2,0x54,0x82,0x40,0x00]
vcmpeqss 256(%rdx,%rax,4), %xmm15, %k2 {%k2}

// CHECK: vcmpeqss -256(%rdx,%rax,4), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x06,0x0a,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqss -256(%rdx,%rax,4), %xmm15, %k2 {%k2}

// CHECK: vcmpeqss 256(%rdx,%rax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x54,0x82,0x40,0x00]
vcmpeqss 256(%rdx,%rax,4), %xmm1, %k2

// CHECK: vcmpeqss -256(%rdx,%rax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqss -256(%rdx,%rax,4), %xmm1, %k2

// CHECK: vcmpeqss 256(%rdx,%rax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x54,0x82,0x40,0x00]
vcmpeqss 256(%rdx,%rax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss -256(%rdx,%rax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x54,0x82,0xc0,0x00]
vcmpeqss -256(%rdx,%rax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 256(%rdx,%rax), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x06,0x08,0xc2,0x54,0x02,0x40,0x00]
vcmpeqss 256(%rdx,%rax), %xmm15, %k2

// CHECK: vcmpeqss 256(%rdx,%rax), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x06,0x0a,0xc2,0x54,0x02,0x40,0x00]
vcmpeqss 256(%rdx,%rax), %xmm15, %k2 {%k2}

// CHECK: vcmpeqss 256(%rdx,%rax), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x54,0x02,0x40,0x00]
vcmpeqss 256(%rdx,%rax), %xmm1, %k2

// CHECK: vcmpeqss 256(%rdx,%rax), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x54,0x02,0x40,0x00]
vcmpeqss 256(%rdx,%rax), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 256(%rdx), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x06,0x08,0xc2,0x52,0x40,0x00]
vcmpeqss 256(%rdx), %xmm15, %k2

// CHECK: vcmpeqss 256(%rdx), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x06,0x0a,0xc2,0x52,0x40,0x00]
vcmpeqss 256(%rdx), %xmm15, %k2 {%k2}

// CHECK: vcmpeqss 256(%rdx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x52,0x40,0x00]
vcmpeqss 256(%rdx), %xmm1, %k2

// CHECK: vcmpeqss 256(%rdx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x52,0x40,0x00]
vcmpeqss 256(%rdx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 485498096, %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x06,0x08,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096, %xmm15, %k2

// CHECK: vcmpeqss 485498096, %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x06,0x0a,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096, %xmm15, %k2 {%k2}

// CHECK: vcmpeqss 485498096, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096, %xmm1, %k2

// CHECK: vcmpeqss 485498096, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x14,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096, %xmm1, %k2 {%k2}

// CHECK: vcmpeqss (%rdx), %xmm15, %k2
// CHECK: encoding: [0x62,0xf1,0x06,0x08,0xc2,0x12,0x00]
vcmpeqss (%rdx), %xmm15, %k2

// CHECK: vcmpeqss (%rdx), %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x06,0x0a,0xc2,0x12,0x00]
vcmpeqss (%rdx), %xmm15, %k2 {%k2}

// CHECK: vcmpeqss (%rdx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x12,0x00]
vcmpeqss (%rdx), %xmm1, %k2

// CHECK: vcmpeqss (%rdx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x12,0x00]
vcmpeqss (%rdx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss {sae}, %xmm15, %xmm15, %k2
// CHECK: encoding: [0x62,0xd1,0x06,0x18,0xc2,0xd7,0x00]
vcmpeqss {sae}, %xmm15, %xmm15, %k2

// CHECK: vcmpeqss {sae}, %xmm15, %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xd1,0x06,0x1a,0xc2,0xd7,0x00]
vcmpeqss {sae}, %xmm15, %xmm15, %k2 {%k2}

// CHECK: vcmpeqss {sae}, %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0xc2,0xd1,0x00]
vcmpeqss {sae}, %xmm1, %xmm1, %k2

// CHECK: vcmpeqss {sae}, %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0xc2,0xd1,0x00]
vcmpeqss {sae}, %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcmpeqss %xmm15, %xmm15, %k2
// CHECK: encoding: [0x62,0xd1,0x06,0x08,0xc2,0xd7,0x00]
vcmpeqss %xmm15, %xmm15, %k2

// CHECK: vcmpeqss %xmm15, %xmm15, %k2 {%k2}
// CHECK: encoding: [0x62,0xd1,0x06,0x0a,0xc2,0xd7,0x00]
vcmpeqss %xmm15, %xmm15, %k2 {%k2}

// CHECK: vcmpeqss %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0xd1,0x00]
vcmpeqss %xmm1, %xmm1, %k2

// CHECK: vcmpeqss %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0xd1,0x00]
vcmpeqss %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcomisd 485498096, %xmm15
// CHECK: encoding: [0xc5,0x79,0x2f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcomisd 485498096, %xmm15

// CHECK: vcomisd 485498096, %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcomisd 485498096, %xmm1

// CHECK: vcomisd 512(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2f,0xbc,0x82,0x00,0x02,0x00,0x00]
vcomisd 512(%rdx,%rax,4), %xmm15

// CHECK: vcomisd -512(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2f,0xbc,0x82,0x00,0xfe,0xff,0xff]
vcomisd -512(%rdx,%rax,4), %xmm15

// CHECK: vcomisd 512(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2f,0x8c,0x82,0x00,0x02,0x00,0x00]
vcomisd 512(%rdx,%rax,4), %xmm1

// CHECK: vcomisd -512(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2f,0x8c,0x82,0x00,0xfe,0xff,0xff]
vcomisd -512(%rdx,%rax,4), %xmm1

// CHECK: vcomisd 512(%rdx,%rax), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2f,0xbc,0x02,0x00,0x02,0x00,0x00]
vcomisd 512(%rdx,%rax), %xmm15

// CHECK: vcomisd 512(%rdx,%rax), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2f,0x8c,0x02,0x00,0x02,0x00,0x00]
vcomisd 512(%rdx,%rax), %xmm1

// CHECK: vcomisd 512(%rdx), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2f,0xba,0x00,0x02,0x00,0x00]
vcomisd 512(%rdx), %xmm15

// CHECK: vcomisd 512(%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2f,0x8a,0x00,0x02,0x00,0x00]
vcomisd 512(%rdx), %xmm1

// CHECK: vcomisd (%rdx), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2f,0x3a]
vcomisd (%rdx), %xmm15

// CHECK: vcomisd (%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2f,0x0a]
vcomisd (%rdx), %xmm1

// CHECK: vcomisd {sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0xfd,0x18,0x2f,0xff]
vcomisd {sae}, %xmm15, %xmm15

// CHECK: vcomisd {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x18,0x2f,0xc9]
vcomisd {sae}, %xmm1, %xmm1

// CHECK: vcomisd %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x79,0x2f,0xff]
vcomisd %xmm15, %xmm15

// CHECK: vcomisd %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2f,0xc9]
vcomisd %xmm1, %xmm1

// CHECK: vcomiss 256(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2f,0xbc,0x82,0x00,0x01,0x00,0x00]
vcomiss 256(%rdx,%rax,4), %xmm15

// CHECK: vcomiss -256(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2f,0xbc,0x82,0x00,0xff,0xff,0xff]
vcomiss -256(%rdx,%rax,4), %xmm15

// CHECK: vcomiss 256(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2f,0x8c,0x82,0x00,0x01,0x00,0x00]
vcomiss 256(%rdx,%rax,4), %xmm1

// CHECK: vcomiss -256(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2f,0x8c,0x82,0x00,0xff,0xff,0xff]
vcomiss -256(%rdx,%rax,4), %xmm1

// CHECK: vcomiss 256(%rdx,%rax), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2f,0xbc,0x02,0x00,0x01,0x00,0x00]
vcomiss 256(%rdx,%rax), %xmm15

// CHECK: vcomiss 256(%rdx,%rax), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2f,0x8c,0x02,0x00,0x01,0x00,0x00]
vcomiss 256(%rdx,%rax), %xmm1

// CHECK: vcomiss 256(%rdx), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2f,0xba,0x00,0x01,0x00,0x00]
vcomiss 256(%rdx), %xmm15

// CHECK: vcomiss 256(%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2f,0x8a,0x00,0x01,0x00,0x00]
vcomiss 256(%rdx), %xmm1

// CHECK: vcomiss 485498096, %xmm15
// CHECK: encoding: [0xc5,0x78,0x2f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcomiss 485498096, %xmm15

// CHECK: vcomiss 485498096, %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcomiss 485498096, %xmm1

// CHECK: vcomiss (%rdx), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2f,0x3a]
vcomiss (%rdx), %xmm15

// CHECK: vcomiss (%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2f,0x0a]
vcomiss (%rdx), %xmm1

// CHECK: vcomiss {sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x7c,0x18,0x2f,0xff]
vcomiss {sae}, %xmm15, %xmm15

// CHECK: vcomiss {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x18,0x2f,0xc9]
vcomiss {sae}, %xmm1, %xmm1

// CHECK: vcomiss %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x78,0x2f,0xff]
vcomiss %xmm15, %xmm15

// CHECK: vcomiss %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2f,0xc9]
vcomiss %xmm1, %xmm1

// CHECK: vcvtsd2si 485498096, %r13d
// CHECK: encoding: [0xc5,0x7b,0x2d,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2si 485498096, %r13d

// CHECK: vcvtsd2si 485498096, %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2si 485498096, %r15

// CHECK: vcvtsd2si 512(%rdx), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2d,0xaa,0x00,0x02,0x00,0x00]
vcvtsd2si 512(%rdx), %r13d

// CHECK: vcvtsd2si 512(%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0xba,0x00,0x02,0x00,0x00]
vcvtsd2si 512(%rdx), %r15

// CHECK: vcvtsd2si 512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2d,0xac,0x82,0x00,0x02,0x00,0x00]
vcvtsd2si 512(%rdx,%rax,4), %r13d

// CHECK: vcvtsd2si -512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2d,0xac,0x82,0x00,0xfe,0xff,0xff]
vcvtsd2si -512(%rdx,%rax,4), %r13d

// CHECK: vcvtsd2si 512(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0xbc,0x82,0x00,0x02,0x00,0x00]
vcvtsd2si 512(%rdx,%rax,4), %r15

// CHECK: vcvtsd2si -512(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0xbc,0x82,0x00,0xfe,0xff,0xff]
vcvtsd2si -512(%rdx,%rax,4), %r15

// CHECK: vcvtsd2si 512(%rdx,%rax), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2d,0xac,0x02,0x00,0x02,0x00,0x00]
vcvtsd2si 512(%rdx,%rax), %r13d

// CHECK: vcvtsd2si 512(%rdx,%rax), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0xbc,0x02,0x00,0x02,0x00,0x00]
vcvtsd2si 512(%rdx,%rax), %r15

// CHECK: vcvtsd2si {rd-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x38,0x2d,0xef]
vcvtsd2si {rd-sae}, %xmm15, %r13d

// CHECK: vcvtsd2si {rd-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x38,0x2d,0xff]
vcvtsd2si {rd-sae}, %xmm15, %r15

// CHECK: vcvtsd2si {rd-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x38,0x2d,0xe9]
vcvtsd2si {rd-sae}, %xmm1, %r13d

// CHECK: vcvtsd2si {rd-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x38,0x2d,0xf9]
vcvtsd2si {rd-sae}, %xmm1, %r15

// CHECK: vcvtsd2si (%rdx), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2d,0x2a]
vcvtsd2si (%rdx), %r13d

// CHECK: vcvtsd2si (%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0x3a]
vcvtsd2si (%rdx), %r15

// CHECK: vcvtsd2si {rn-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x18,0x2d,0xef]
vcvtsd2si {rn-sae}, %xmm15, %r13d

// CHECK: vcvtsd2si {rn-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x18,0x2d,0xff]
vcvtsd2si {rn-sae}, %xmm15, %r15

// CHECK: vcvtsd2si {rn-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x18,0x2d,0xe9]
vcvtsd2si {rn-sae}, %xmm1, %r13d

// CHECK: vcvtsd2si {rn-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x18,0x2d,0xf9]
vcvtsd2si {rn-sae}, %xmm1, %r15

// CHECK: vcvtsd2si {ru-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x58,0x2d,0xef]
vcvtsd2si {ru-sae}, %xmm15, %r13d

// CHECK: vcvtsd2si {ru-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x58,0x2d,0xff]
vcvtsd2si {ru-sae}, %xmm15, %r15

// CHECK: vcvtsd2si {ru-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x58,0x2d,0xe9]
vcvtsd2si {ru-sae}, %xmm1, %r13d

// CHECK: vcvtsd2si {ru-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x58,0x2d,0xf9]
vcvtsd2si {ru-sae}, %xmm1, %r15

// CHECK: vcvtsd2si {rz-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x78,0x2d,0xef]
vcvtsd2si {rz-sae}, %xmm15, %r13d

// CHECK: vcvtsd2si {rz-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x78,0x2d,0xff]
vcvtsd2si {rz-sae}, %xmm15, %r15

// CHECK: vcvtsd2si {rz-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x78,0x2d,0xe9]
vcvtsd2si {rz-sae}, %xmm1, %r13d

// CHECK: vcvtsd2si {rz-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x78,0x2d,0xf9]
vcvtsd2si {rz-sae}, %xmm1, %r15

// CHECK: vcvtsd2si %xmm15, %r13d
// CHECK: encoding: [0xc4,0x41,0x7b,0x2d,0xef]
vcvtsd2si %xmm15, %r13d

// CHECK: vcvtsd2si %xmm15, %r15
// CHECK: encoding: [0xc4,0x41,0xfb,0x2d,0xff]
vcvtsd2si %xmm15, %r15

// CHECK: vcvtsd2si %xmm1, %r13d
// CHECK: encoding: [0xc5,0x7b,0x2d,0xe9]
vcvtsd2si %xmm1, %r13d

// CHECK: vcvtsd2si %xmm1, %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2d,0xf9]
vcvtsd2si %xmm1, %r15

// CHECK: vcvtsd2ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm15, %xmm15

// CHECK: vcvtsd2ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm1, %xmm1

// CHECK: vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5a,0xbc,0x82,0x00,0x02,0x00,0x00]
vcvtsd2ss 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsd2ss -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5a,0xbc,0x82,0x00,0xfe,0xff,0xff]
vcvtsd2ss -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsd2ss 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5a,0x7c,0x82,0x40]
vcvtsd2ss 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5a,0x7c,0x82,0xc0]
vcvtsd2ss -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5a,0x7c,0x82,0x40]
vcvtsd2ss 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5a,0x7c,0x82,0xc0]
vcvtsd2ss -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5a,0x8c,0x82,0x00,0x02,0x00,0x00]
vcvtsd2ss 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsd2ss -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5a,0x8c,0x82,0x00,0xfe,0xff,0xff]
vcvtsd2ss -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsd2ss 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x4c,0x82,0x40]
vcvtsd2ss 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x4c,0x82,0xc0]
vcvtsd2ss -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x4c,0x82,0x40]
vcvtsd2ss 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x4c,0x82,0xc0]
vcvtsd2ss -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5a,0xbc,0x02,0x00,0x02,0x00,0x00]
vcvtsd2ss 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtsd2ss 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5a,0x7c,0x02,0x40]
vcvtsd2ss 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5a,0x7c,0x02,0x40]
vcvtsd2ss 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5a,0x8c,0x02,0x00,0x02,0x00,0x00]
vcvtsd2ss 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtsd2ss 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x4c,0x02,0x40]
vcvtsd2ss 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x4c,0x02,0x40]
vcvtsd2ss 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5a,0xba,0x00,0x02,0x00,0x00]
vcvtsd2ss 512(%rdx), %xmm15, %xmm15

// CHECK: vcvtsd2ss 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5a,0x7a,0x40]
vcvtsd2ss 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5a,0x7a,0x40]
vcvtsd2ss 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5a,0x8a,0x00,0x02,0x00,0x00]
vcvtsd2ss 512(%rdx), %xmm1, %xmm1

// CHECK: vcvtsd2ss 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x4a,0x40]
vcvtsd2ss 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x4a,0x40]
vcvtsd2ss 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x5a,0xff]
vcvtsd2ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vcvtsd2ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x3a,0x5a,0xff]
vcvtsd2ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xba,0x5a,0xff]
vcvtsd2ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x5a,0xc9]
vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x5a,0xc9]
vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x5a,0xc9]
vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5a,0x3a]
vcvtsd2ss (%rdx), %xmm15, %xmm15

// CHECK: vcvtsd2ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5a,0x3a]
vcvtsd2ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5a,0x3a]
vcvtsd2ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5a,0x0a]
vcvtsd2ss (%rdx), %xmm1, %xmm1

// CHECK: vcvtsd2ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x0a]
vcvtsd2ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x0a]
vcvtsd2ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x5a,0xff]
vcvtsd2ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vcvtsd2ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x5a,0xff]
vcvtsd2ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x5a,0xff]
vcvtsd2ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5a,0xc9]
vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5a,0xc9]
vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5a,0xc9]
vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x5a,0xff]
vcvtsd2ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vcvtsd2ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x5a,0x5a,0xff]
vcvtsd2ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xda,0x5a,0xff]
vcvtsd2ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x5a,0xc9]
vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x5a,0xc9]
vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x5a,0xc9]
vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x5a,0xff]
vcvtsd2ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vcvtsd2ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x7a,0x5a,0xff]
vcvtsd2ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xfa,0x5a,0xff]
vcvtsd2ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x5a,0xc9]
vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x5a,0xc9]
vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x5a,0xc9]
vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x5a,0xff]
vcvtsd2ss %xmm15, %xmm15, %xmm15

// CHECK: vcvtsd2ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x5a,0xff]
vcvtsd2ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vcvtsd2ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x5a,0xff]
vcvtsd2ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtsd2ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5a,0xc9]
vcvtsd2ss %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0xc9]
vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0xc9]
vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2usi 485498096, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x79,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2usi 485498096, %r13d

// CHECK: vcvtsd2usi 485498096, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x79,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsd2usi 485498096, %r15

// CHECK: vcvtsd2usi 512(%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x79,0x6a,0x40]
vcvtsd2usi 512(%rdx), %r13d

// CHECK: vcvtsd2usi 512(%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x79,0x7a,0x40]
vcvtsd2usi 512(%rdx), %r15

// CHECK: vcvtsd2usi 512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x79,0x6c,0x82,0x40]
vcvtsd2usi 512(%rdx,%rax,4), %r13d

// CHECK: vcvtsd2usi -512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x79,0x6c,0x82,0xc0]
vcvtsd2usi -512(%rdx,%rax,4), %r13d

// CHECK: vcvtsd2usi 512(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x79,0x7c,0x82,0x40]
vcvtsd2usi 512(%rdx,%rax,4), %r15

// CHECK: vcvtsd2usi -512(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x79,0x7c,0x82,0xc0]
vcvtsd2usi -512(%rdx,%rax,4), %r15

// CHECK: vcvtsd2usi 512(%rdx,%rax), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x79,0x6c,0x02,0x40]
vcvtsd2usi 512(%rdx,%rax), %r13d

// CHECK: vcvtsd2usi 512(%rdx,%rax), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x79,0x7c,0x02,0x40]
vcvtsd2usi 512(%rdx,%rax), %r15

// CHECK: vcvtsd2usi {rd-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x38,0x79,0xef]
vcvtsd2usi {rd-sae}, %xmm15, %r13d

// CHECK: vcvtsd2usi {rd-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x38,0x79,0xff]
vcvtsd2usi {rd-sae}, %xmm15, %r15

// CHECK: vcvtsd2usi {rd-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x38,0x79,0xe9]
vcvtsd2usi {rd-sae}, %xmm1, %r13d

// CHECK: vcvtsd2usi {rd-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x38,0x79,0xf9]
vcvtsd2usi {rd-sae}, %xmm1, %r15

// CHECK: vcvtsd2usi (%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x79,0x2a]
vcvtsd2usi (%rdx), %r13d

// CHECK: vcvtsd2usi (%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x79,0x3a]
vcvtsd2usi (%rdx), %r15

// CHECK: vcvtsd2usi {rn-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x18,0x79,0xef]
vcvtsd2usi {rn-sae}, %xmm15, %r13d

// CHECK: vcvtsd2usi {rn-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x18,0x79,0xff]
vcvtsd2usi {rn-sae}, %xmm15, %r15

// CHECK: vcvtsd2usi {rn-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x18,0x79,0xe9]
vcvtsd2usi {rn-sae}, %xmm1, %r13d

// CHECK: vcvtsd2usi {rn-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x18,0x79,0xf9]
vcvtsd2usi {rn-sae}, %xmm1, %r15

// CHECK: vcvtsd2usi {ru-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x58,0x79,0xef]
vcvtsd2usi {ru-sae}, %xmm15, %r13d

// CHECK: vcvtsd2usi {ru-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x58,0x79,0xff]
vcvtsd2usi {ru-sae}, %xmm15, %r15

// CHECK: vcvtsd2usi {ru-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x58,0x79,0xe9]
vcvtsd2usi {ru-sae}, %xmm1, %r13d

// CHECK: vcvtsd2usi {ru-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x58,0x79,0xf9]
vcvtsd2usi {ru-sae}, %xmm1, %r15

// CHECK: vcvtsd2usi {rz-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x78,0x79,0xef]
vcvtsd2usi {rz-sae}, %xmm15, %r13d

// CHECK: vcvtsd2usi {rz-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x78,0x79,0xff]
vcvtsd2usi {rz-sae}, %xmm15, %r15

// CHECK: vcvtsd2usi {rz-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x78,0x79,0xe9]
vcvtsd2usi {rz-sae}, %xmm1, %r13d

// CHECK: vcvtsd2usi {rz-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x78,0x79,0xf9]
vcvtsd2usi {rz-sae}, %xmm1, %r15

// CHECK: vcvtsd2usi %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x08,0x79,0xef]
vcvtsd2usi %xmm15, %r13d

// CHECK: vcvtsd2usi %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x08,0x79,0xff]
vcvtsd2usi %xmm15, %r15

// CHECK: vcvtsd2usi %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x79,0xe9]
vcvtsd2usi %xmm1, %r13d

// CHECK: vcvtsd2usi %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x79,0xf9]
vcvtsd2usi %xmm1, %r15

// CHECK: vcvtsi2sdl 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x2a,0xbc,0x82,0x00,0x01,0x00,0x00]
vcvtsi2sdl 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2sdl -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x2a,0xbc,0x82,0x00,0xff,0xff,0xff]
vcvtsi2sdl -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2sdl 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x2a,0x8c,0x82,0x00,0x01,0x00,0x00]
vcvtsi2sdl 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2sdl -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x2a,0x8c,0x82,0x00,0xff,0xff,0xff]
vcvtsi2sdl -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2sdl 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x2a,0xbc,0x02,0x00,0x01,0x00,0x00]
vcvtsi2sdl 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtsi2sdl 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x2a,0x8c,0x02,0x00,0x01,0x00,0x00]
vcvtsi2sdl 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtsi2sdl 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x2a,0xba,0x00,0x01,0x00,0x00]
vcvtsi2sdl 256(%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2sdl 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x2a,0x8a,0x00,0x01,0x00,0x00]
vcvtsi2sdl 256(%rdx), %xmm1, %xmm1

// CHECK: vcvtsi2sdl 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2sdl 485498096, %xmm15, %xmm15

// CHECK: vcvtsi2sdl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x2a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2sdl 485498096, %xmm1, %xmm1

// CHECK: vcvtsi2sdl %r13d, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x2a,0xfd]
vcvtsi2sdl %r13d, %xmm15, %xmm15

// CHECK: vcvtsi2sdl %r13d, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xc1,0x73,0x2a,0xcd]
vcvtsi2sdl %r13d, %xmm1, %xmm1

// CHECK: vcvtsi2sdl (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x2a,0x3a]
vcvtsi2sdl (%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2sdl (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x2a,0x0a]
vcvtsi2sdl (%rdx), %xmm1, %xmm1

// CHECK: vcvtsi2sdq 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2sdq 485498096, %xmm15, %xmm15

// CHECK: vcvtsi2sdq 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf3,0x2a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2sdq 485498096, %xmm1, %xmm1

// CHECK: vcvtsi2sdq 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0xbc,0x82,0x00,0x02,0x00,0x00]
vcvtsi2sdq 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2sdq -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0xbc,0x82,0x00,0xfe,0xff,0xff]
vcvtsi2sdq -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2sdq 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf3,0x2a,0x8c,0x82,0x00,0x02,0x00,0x00]
vcvtsi2sdq 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2sdq -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf3,0x2a,0x8c,0x82,0x00,0xfe,0xff,0xff]
vcvtsi2sdq -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2sdq 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0xbc,0x02,0x00,0x02,0x00,0x00]
vcvtsi2sdq 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtsi2sdq 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf3,0x2a,0x8c,0x02,0x00,0x02,0x00,0x00]
vcvtsi2sdq 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtsi2sdq 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0xba,0x00,0x02,0x00,0x00]
vcvtsi2sdq 512(%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2sdq 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf3,0x2a,0x8a,0x00,0x02,0x00,0x00]
vcvtsi2sdq 512(%rdx), %xmm1, %xmm1

// CHECK: vcvtsi2sdq %r15, {rd-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x2a,0xff]
vcvtsi2sdq %r15, {rd-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2sdq %r15, {rd-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x38,0x2a,0xcf]
vcvtsi2sdq %r15, {rd-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2sdq %r15, {rn-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x2a,0xff]
vcvtsi2sdq %r15, {rn-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2sdq %r15, {rn-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x18,0x2a,0xcf]
vcvtsi2sdq %r15, {rn-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2sdq %r15, {ru-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x2a,0xff]
vcvtsi2sdq %r15, {ru-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2sdq %r15, {ru-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x58,0x2a,0xcf]
vcvtsi2sdq %r15, {ru-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2sdq %r15, {rz-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x2a,0xff]
vcvtsi2sdq %r15, {rz-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2sdq %r15, {rz-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x78,0x2a,0xcf]
vcvtsi2sdq %r15, {rz-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2sdq %r15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x83,0x2a,0xff]
vcvtsi2sdq %r15, %xmm15, %xmm15

// CHECK: vcvtsi2sdq %r15, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xc1,0xf3,0x2a,0xcf]
vcvtsi2sdq %r15, %xmm1, %xmm1

// CHECK: vcvtsi2sdq (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x83,0x2a,0x3a]
vcvtsi2sdq (%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2sdq (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf3,0x2a,0x0a]
vcvtsi2sdq (%rdx), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x2a,0xbc,0x82,0x00,0x01,0x00,0x00]
vcvtsi2ssl 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2ssl -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x2a,0xbc,0x82,0x00,0xff,0xff,0xff]
vcvtsi2ssl -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2ssl 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x2a,0x8c,0x82,0x00,0x01,0x00,0x00]
vcvtsi2ssl 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2ssl -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x2a,0x8c,0x82,0x00,0xff,0xff,0xff]
vcvtsi2ssl -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x2a,0xbc,0x02,0x00,0x01,0x00,0x00]
vcvtsi2ssl 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtsi2ssl 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x2a,0x8c,0x02,0x00,0x01,0x00,0x00]
vcvtsi2ssl 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x2a,0xba,0x00,0x01,0x00,0x00]
vcvtsi2ssl 256(%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2ssl 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x2a,0x8a,0x00,0x01,0x00,0x00]
vcvtsi2ssl 256(%rdx), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2ssl 485498096, %xmm15, %xmm15

// CHECK: vcvtsi2ssl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x2a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2ssl 485498096, %xmm1, %xmm1

// CHECK: vcvtsi2ssl %r13d, {rd-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x38,0x2a,0xfd]
vcvtsi2ssl %r13d, {rd-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssl %r13d, {rd-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x38,0x2a,0xcd]
vcvtsi2ssl %r13d, {rd-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssl %r13d, {rn-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x2a,0xfd]
vcvtsi2ssl %r13d, {rn-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssl %r13d, {rn-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x18,0x2a,0xcd]
vcvtsi2ssl %r13d, {rn-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssl %r13d, {ru-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x58,0x2a,0xfd]
vcvtsi2ssl %r13d, {ru-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssl %r13d, {ru-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x58,0x2a,0xcd]
vcvtsi2ssl %r13d, {ru-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssl %r13d, {rz-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x78,0x2a,0xfd]
vcvtsi2ssl %r13d, {rz-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssl %r13d, {rz-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x78,0x2a,0xcd]
vcvtsi2ssl %r13d, {rz-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssl %r13d, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x2a,0xfd]
vcvtsi2ssl %r13d, %xmm15, %xmm15

// CHECK: vcvtsi2ssl %r13d, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xc1,0x72,0x2a,0xcd]
vcvtsi2ssl %r13d, %xmm1, %xmm1

// CHECK: vcvtsi2ssl (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x2a,0x3a]
vcvtsi2ssl (%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2ssl (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x2a,0x0a]
vcvtsi2ssl (%rdx), %xmm1, %xmm1

// CHECK: vcvtsi2ssq 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2ssq 485498096, %xmm15, %xmm15

// CHECK: vcvtsi2ssq 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf2,0x2a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtsi2ssq 485498096, %xmm1, %xmm1

// CHECK: vcvtsi2ssq 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0xbc,0x82,0x00,0x02,0x00,0x00]
vcvtsi2ssq 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2ssq -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0xbc,0x82,0x00,0xfe,0xff,0xff]
vcvtsi2ssq -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtsi2ssq 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf2,0x2a,0x8c,0x82,0x00,0x02,0x00,0x00]
vcvtsi2ssq 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2ssq -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf2,0x2a,0x8c,0x82,0x00,0xfe,0xff,0xff]
vcvtsi2ssq -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtsi2ssq 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0xbc,0x02,0x00,0x02,0x00,0x00]
vcvtsi2ssq 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtsi2ssq 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf2,0x2a,0x8c,0x02,0x00,0x02,0x00,0x00]
vcvtsi2ssq 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtsi2ssq 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0xba,0x00,0x02,0x00,0x00]
vcvtsi2ssq 512(%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2ssq 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf2,0x2a,0x8a,0x00,0x02,0x00,0x00]
vcvtsi2ssq 512(%rdx), %xmm1, %xmm1

// CHECK: vcvtsi2ssq %r15, {rd-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x38,0x2a,0xff]
vcvtsi2ssq %r15, {rd-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssq %r15, {rd-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x38,0x2a,0xcf]
vcvtsi2ssq %r15, {rd-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssq %r15, {rn-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x18,0x2a,0xff]
vcvtsi2ssq %r15, {rn-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssq %r15, {rn-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x18,0x2a,0xcf]
vcvtsi2ssq %r15, {rn-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssq %r15, {ru-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x58,0x2a,0xff]
vcvtsi2ssq %r15, {ru-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssq %r15, {ru-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x58,0x2a,0xcf]
vcvtsi2ssq %r15, {ru-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssq %r15, {rz-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x78,0x2a,0xff]
vcvtsi2ssq %r15, {rz-sae}, %xmm15, %xmm15

// CHECK: vcvtsi2ssq %r15, {rz-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x78,0x2a,0xcf]
vcvtsi2ssq %r15, {rz-sae}, %xmm1, %xmm1

// CHECK: vcvtsi2ssq %r15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x82,0x2a,0xff]
vcvtsi2ssq %r15, %xmm15, %xmm15

// CHECK: vcvtsi2ssq %r15, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xc1,0xf2,0x2a,0xcf]
vcvtsi2ssq %r15, %xmm1, %xmm1

// CHECK: vcvtsi2ssq (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x61,0x82,0x2a,0x3a]
vcvtsi2ssq (%rdx), %xmm15, %xmm15

// CHECK: vcvtsi2ssq (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe1,0xf2,0x2a,0x0a]
vcvtsi2ssq (%rdx), %xmm1, %xmm1

// CHECK: vcvtss2sd 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5a,0xbc,0x82,0x00,0x01,0x00,0x00]
vcvtss2sd 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtss2sd -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5a,0xbc,0x82,0x00,0xff,0xff,0xff]
vcvtss2sd -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtss2sd 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5a,0x7c,0x82,0x40]
vcvtss2sd 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5a,0x7c,0x82,0xc0]
vcvtss2sd -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5a,0x7c,0x82,0x40]
vcvtss2sd 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5a,0x7c,0x82,0xc0]
vcvtss2sd -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5a,0x8c,0x82,0x00,0x01,0x00,0x00]
vcvtss2sd 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtss2sd -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5a,0x8c,0x82,0x00,0xff,0xff,0xff]
vcvtss2sd -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtss2sd 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x4c,0x82,0x40]
vcvtss2sd 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x4c,0x82,0xc0]
vcvtss2sd -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x4c,0x82,0x40]
vcvtss2sd 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x4c,0x82,0xc0]
vcvtss2sd -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5a,0xbc,0x02,0x00,0x01,0x00,0x00]
vcvtss2sd 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtss2sd 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5a,0x7c,0x02,0x40]
vcvtss2sd 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5a,0x7c,0x02,0x40]
vcvtss2sd 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5a,0x8c,0x02,0x00,0x01,0x00,0x00]
vcvtss2sd 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtss2sd 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x4c,0x02,0x40]
vcvtss2sd 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x4c,0x02,0x40]
vcvtss2sd 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5a,0xba,0x00,0x01,0x00,0x00]
vcvtss2sd 256(%rdx), %xmm15, %xmm15

// CHECK: vcvtss2sd 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5a,0x7a,0x40]
vcvtss2sd 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5a,0x7a,0x40]
vcvtss2sd 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5a,0x8a,0x00,0x01,0x00,0x00]
vcvtss2sd 256(%rdx), %xmm1, %xmm1

// CHECK: vcvtss2sd 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x4a,0x40]
vcvtss2sd 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x4a,0x40]
vcvtss2sd 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm15, %xmm15

// CHECK: vcvtss2sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm1, %xmm1

// CHECK: vcvtss2sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5a,0x3a]
vcvtss2sd (%rdx), %xmm15, %xmm15

// CHECK: vcvtss2sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5a,0x3a]
vcvtss2sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5a,0x3a]
vcvtss2sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5a,0x0a]
vcvtss2sd (%rdx), %xmm1, %xmm1

// CHECK: vcvtss2sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x0a]
vcvtss2sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x0a]
vcvtss2sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x5a,0xff]
vcvtss2sd {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vcvtss2sd {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x5a,0xff]
vcvtss2sd {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x5a,0xff]
vcvtss2sd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5a,0xc9]
vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5a,0xc9]
vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5a,0xc9]
vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x5a,0xff]
vcvtss2sd %xmm15, %xmm15, %xmm15

// CHECK: vcvtss2sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x5a,0xff]
vcvtss2sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vcvtss2sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x5a,0xff]
vcvtss2sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vcvtss2sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5a,0xc9]
vcvtss2sd %xmm1, %xmm1, %xmm1

// CHECK: vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0xc9]
vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0xc9]
vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2si 256(%rdx), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2d,0xaa,0x00,0x01,0x00,0x00]
vcvtss2si 256(%rdx), %r13d

// CHECK: vcvtss2si 256(%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0xba,0x00,0x01,0x00,0x00]
vcvtss2si 256(%rdx), %r15

// CHECK: vcvtss2si 256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2d,0xac,0x82,0x00,0x01,0x00,0x00]
vcvtss2si 256(%rdx,%rax,4), %r13d

// CHECK: vcvtss2si -256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2d,0xac,0x82,0x00,0xff,0xff,0xff]
vcvtss2si -256(%rdx,%rax,4), %r13d

// CHECK: vcvtss2si 256(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0xbc,0x82,0x00,0x01,0x00,0x00]
vcvtss2si 256(%rdx,%rax,4), %r15

// CHECK: vcvtss2si -256(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0xbc,0x82,0x00,0xff,0xff,0xff]
vcvtss2si -256(%rdx,%rax,4), %r15

// CHECK: vcvtss2si 256(%rdx,%rax), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2d,0xac,0x02,0x00,0x01,0x00,0x00]
vcvtss2si 256(%rdx,%rax), %r13d

// CHECK: vcvtss2si 256(%rdx,%rax), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0xbc,0x02,0x00,0x01,0x00,0x00]
vcvtss2si 256(%rdx,%rax), %r15

// CHECK: vcvtss2si 485498096, %r13d
// CHECK: encoding: [0xc5,0x7a,0x2d,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2si 485498096, %r13d

// CHECK: vcvtss2si 485498096, %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2si 485498096, %r15

// CHECK: vcvtss2si {rd-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x38,0x2d,0xef]
vcvtss2si {rd-sae}, %xmm15, %r13d

// CHECK: vcvtss2si {rd-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x38,0x2d,0xff]
vcvtss2si {rd-sae}, %xmm15, %r15

// CHECK: vcvtss2si {rd-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x38,0x2d,0xe9]
vcvtss2si {rd-sae}, %xmm1, %r13d

// CHECK: vcvtss2si {rd-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x38,0x2d,0xf9]
vcvtss2si {rd-sae}, %xmm1, %r15

// CHECK: vcvtss2si (%rdx), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2d,0x2a]
vcvtss2si (%rdx), %r13d

// CHECK: vcvtss2si (%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x3a]
vcvtss2si (%rdx), %r15

// CHECK: vcvtss2si {rn-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x18,0x2d,0xef]
vcvtss2si {rn-sae}, %xmm15, %r13d

// CHECK: vcvtss2si {rn-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x18,0x2d,0xff]
vcvtss2si {rn-sae}, %xmm15, %r15

// CHECK: vcvtss2si {rn-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x18,0x2d,0xe9]
vcvtss2si {rn-sae}, %xmm1, %r13d

// CHECK: vcvtss2si {rn-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x18,0x2d,0xf9]
vcvtss2si {rn-sae}, %xmm1, %r15

// CHECK: vcvtss2si {ru-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x58,0x2d,0xef]
vcvtss2si {ru-sae}, %xmm15, %r13d

// CHECK: vcvtss2si {ru-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x58,0x2d,0xff]
vcvtss2si {ru-sae}, %xmm15, %r15

// CHECK: vcvtss2si {ru-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x58,0x2d,0xe9]
vcvtss2si {ru-sae}, %xmm1, %r13d

// CHECK: vcvtss2si {ru-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x58,0x2d,0xf9]
vcvtss2si {ru-sae}, %xmm1, %r15

// CHECK: vcvtss2si {rz-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x78,0x2d,0xef]
vcvtss2si {rz-sae}, %xmm15, %r13d

// CHECK: vcvtss2si {rz-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x78,0x2d,0xff]
vcvtss2si {rz-sae}, %xmm15, %r15

// CHECK: vcvtss2si {rz-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x78,0x2d,0xe9]
vcvtss2si {rz-sae}, %xmm1, %r13d

// CHECK: vcvtss2si {rz-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x78,0x2d,0xf9]
vcvtss2si {rz-sae}, %xmm1, %r15

// CHECK: vcvtss2si %xmm15, %r13d
// CHECK: encoding: [0xc4,0x41,0x7a,0x2d,0xef]
vcvtss2si %xmm15, %r13d

// CHECK: vcvtss2si %xmm15, %r15
// CHECK: encoding: [0xc4,0x41,0xfa,0x2d,0xff]
vcvtss2si %xmm15, %r15

// CHECK: vcvtss2si %xmm1, %r13d
// CHECK: encoding: [0xc5,0x7a,0x2d,0xe9]
vcvtss2si %xmm1, %r13d

// CHECK: vcvtss2si %xmm1, %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0xf9]
vcvtss2si %xmm1, %r15

// CHECK: vcvtss2usi 256(%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x79,0x6a,0x40]
vcvtss2usi 256(%rdx), %r13d

// CHECK: vcvtss2usi 256(%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x79,0x7a,0x40]
vcvtss2usi 256(%rdx), %r15

// CHECK: vcvtss2usi 256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x79,0x6c,0x82,0x40]
vcvtss2usi 256(%rdx,%rax,4), %r13d

// CHECK: vcvtss2usi -256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x79,0x6c,0x82,0xc0]
vcvtss2usi -256(%rdx,%rax,4), %r13d

// CHECK: vcvtss2usi 256(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x79,0x7c,0x82,0x40]
vcvtss2usi 256(%rdx,%rax,4), %r15

// CHECK: vcvtss2usi -256(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x79,0x7c,0x82,0xc0]
vcvtss2usi -256(%rdx,%rax,4), %r15

// CHECK: vcvtss2usi 256(%rdx,%rax), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x79,0x6c,0x02,0x40]
vcvtss2usi 256(%rdx,%rax), %r13d

// CHECK: vcvtss2usi 256(%rdx,%rax), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x79,0x7c,0x02,0x40]
vcvtss2usi 256(%rdx,%rax), %r15

// CHECK: vcvtss2usi 485498096, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x79,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2usi 485498096, %r13d

// CHECK: vcvtss2usi 485498096, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x79,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtss2usi 485498096, %r15

// CHECK: vcvtss2usi {rd-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x38,0x79,0xef]
vcvtss2usi {rd-sae}, %xmm15, %r13d

// CHECK: vcvtss2usi {rd-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x38,0x79,0xff]
vcvtss2usi {rd-sae}, %xmm15, %r15

// CHECK: vcvtss2usi {rd-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x38,0x79,0xe9]
vcvtss2usi {rd-sae}, %xmm1, %r13d

// CHECK: vcvtss2usi {rd-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x38,0x79,0xf9]
vcvtss2usi {rd-sae}, %xmm1, %r15

// CHECK: vcvtss2usi (%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x79,0x2a]
vcvtss2usi (%rdx), %r13d

// CHECK: vcvtss2usi (%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x79,0x3a]
vcvtss2usi (%rdx), %r15

// CHECK: vcvtss2usi {rn-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x18,0x79,0xef]
vcvtss2usi {rn-sae}, %xmm15, %r13d

// CHECK: vcvtss2usi {rn-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x18,0x79,0xff]
vcvtss2usi {rn-sae}, %xmm15, %r15

// CHECK: vcvtss2usi {rn-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x18,0x79,0xe9]
vcvtss2usi {rn-sae}, %xmm1, %r13d

// CHECK: vcvtss2usi {rn-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x18,0x79,0xf9]
vcvtss2usi {rn-sae}, %xmm1, %r15

// CHECK: vcvtss2usi {ru-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x58,0x79,0xef]
vcvtss2usi {ru-sae}, %xmm15, %r13d

// CHECK: vcvtss2usi {ru-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x58,0x79,0xff]
vcvtss2usi {ru-sae}, %xmm15, %r15

// CHECK: vcvtss2usi {ru-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x58,0x79,0xe9]
vcvtss2usi {ru-sae}, %xmm1, %r13d

// CHECK: vcvtss2usi {ru-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x58,0x79,0xf9]
vcvtss2usi {ru-sae}, %xmm1, %r15

// CHECK: vcvtss2usi {rz-sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x78,0x79,0xef]
vcvtss2usi {rz-sae}, %xmm15, %r13d

// CHECK: vcvtss2usi {rz-sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x78,0x79,0xff]
vcvtss2usi {rz-sae}, %xmm15, %r15

// CHECK: vcvtss2usi {rz-sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x78,0x79,0xe9]
vcvtss2usi {rz-sae}, %xmm1, %r13d

// CHECK: vcvtss2usi {rz-sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x78,0x79,0xf9]
vcvtss2usi {rz-sae}, %xmm1, %r15

// CHECK: vcvtss2usi %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x08,0x79,0xef]
vcvtss2usi %xmm15, %r13d

// CHECK: vcvtss2usi %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x08,0x79,0xff]
vcvtss2usi %xmm15, %r15

// CHECK: vcvtss2usi %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x79,0xe9]
vcvtss2usi %xmm1, %r13d

// CHECK: vcvtss2usi %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x79,0xf9]
vcvtss2usi %xmm1, %r15

// CHECK: vcvttsd2si 485498096, %r13d
// CHECK: encoding: [0xc5,0x7b,0x2c,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttsd2si 485498096, %r13d

// CHECK: vcvttsd2si 485498096, %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttsd2si 485498096, %r15

// CHECK: vcvttsd2si 512(%rdx), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2c,0xaa,0x00,0x02,0x00,0x00]
vcvttsd2si 512(%rdx), %r13d

// CHECK: vcvttsd2si 512(%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0xba,0x00,0x02,0x00,0x00]
vcvttsd2si 512(%rdx), %r15

// CHECK: vcvttsd2si 512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2c,0xac,0x82,0x00,0x02,0x00,0x00]
vcvttsd2si 512(%rdx,%rax,4), %r13d

// CHECK: vcvttsd2si -512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2c,0xac,0x82,0x00,0xfe,0xff,0xff]
vcvttsd2si -512(%rdx,%rax,4), %r13d

// CHECK: vcvttsd2si 512(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0xbc,0x82,0x00,0x02,0x00,0x00]
vcvttsd2si 512(%rdx,%rax,4), %r15

// CHECK: vcvttsd2si -512(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0xbc,0x82,0x00,0xfe,0xff,0xff]
vcvttsd2si -512(%rdx,%rax,4), %r15

// CHECK: vcvttsd2si 512(%rdx,%rax), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2c,0xac,0x02,0x00,0x02,0x00,0x00]
vcvttsd2si 512(%rdx,%rax), %r13d

// CHECK: vcvttsd2si 512(%rdx,%rax), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0xbc,0x02,0x00,0x02,0x00,0x00]
vcvttsd2si 512(%rdx,%rax), %r15

// CHECK: vcvttsd2si (%rdx), %r13d
// CHECK: encoding: [0xc5,0x7b,0x2c,0x2a]
vcvttsd2si (%rdx), %r13d

// CHECK: vcvttsd2si (%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0x3a]
vcvttsd2si (%rdx), %r15

// CHECK: vcvttsd2si {sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x18,0x2c,0xef]
vcvttsd2si {sae}, %xmm15, %r13d

// CHECK: vcvttsd2si {sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x18,0x2c,0xff]
vcvttsd2si {sae}, %xmm15, %r15

// CHECK: vcvttsd2si {sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x18,0x2c,0xe9]
vcvttsd2si {sae}, %xmm1, %r13d

// CHECK: vcvttsd2si {sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x18,0x2c,0xf9]
vcvttsd2si {sae}, %xmm1, %r15

// CHECK: vcvttsd2si %xmm15, %r13d
// CHECK: encoding: [0xc4,0x41,0x7b,0x2c,0xef]
vcvttsd2si %xmm15, %r13d

// CHECK: vcvttsd2si %xmm15, %r15
// CHECK: encoding: [0xc4,0x41,0xfb,0x2c,0xff]
vcvttsd2si %xmm15, %r15

// CHECK: vcvttsd2si %xmm1, %r13d
// CHECK: encoding: [0xc5,0x7b,0x2c,0xe9]
vcvttsd2si %xmm1, %r13d

// CHECK: vcvttsd2si %xmm1, %r15
// CHECK: encoding: [0xc4,0x61,0xfb,0x2c,0xf9]
vcvttsd2si %xmm1, %r15

// CHECK: vcvttsd2usi 485498096, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x78,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttsd2usi 485498096, %r13d

// CHECK: vcvttsd2usi 485498096, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x78,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttsd2usi 485498096, %r15

// CHECK: vcvttsd2usi 512(%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x78,0x6a,0x40]
vcvttsd2usi 512(%rdx), %r13d

// CHECK: vcvttsd2usi 512(%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x78,0x7a,0x40]
vcvttsd2usi 512(%rdx), %r15

// CHECK: vcvttsd2usi 512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x78,0x6c,0x82,0x40]
vcvttsd2usi 512(%rdx,%rax,4), %r13d

// CHECK: vcvttsd2usi -512(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x78,0x6c,0x82,0xc0]
vcvttsd2usi -512(%rdx,%rax,4), %r13d

// CHECK: vcvttsd2usi 512(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x78,0x7c,0x82,0x40]
vcvttsd2usi 512(%rdx,%rax,4), %r15

// CHECK: vcvttsd2usi -512(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x78,0x7c,0x82,0xc0]
vcvttsd2usi -512(%rdx,%rax,4), %r15

// CHECK: vcvttsd2usi 512(%rdx,%rax), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x78,0x6c,0x02,0x40]
vcvttsd2usi 512(%rdx,%rax), %r13d

// CHECK: vcvttsd2usi 512(%rdx,%rax), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x78,0x7c,0x02,0x40]
vcvttsd2usi 512(%rdx,%rax), %r15

// CHECK: vcvttsd2usi (%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x78,0x2a]
vcvttsd2usi (%rdx), %r13d

// CHECK: vcvttsd2usi (%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x78,0x3a]
vcvttsd2usi (%rdx), %r15

// CHECK: vcvttsd2usi {sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x18,0x78,0xef]
vcvttsd2usi {sae}, %xmm15, %r13d

// CHECK: vcvttsd2usi {sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x18,0x78,0xff]
vcvttsd2usi {sae}, %xmm15, %r15

// CHECK: vcvttsd2usi {sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x18,0x78,0xe9]
vcvttsd2usi {sae}, %xmm1, %r13d

// CHECK: vcvttsd2usi {sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x18,0x78,0xf9]
vcvttsd2usi {sae}, %xmm1, %r15

// CHECK: vcvttsd2usi %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7f,0x08,0x78,0xef]
vcvttsd2usi %xmm15, %r13d

// CHECK: vcvttsd2usi %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xff,0x08,0x78,0xff]
vcvttsd2usi %xmm15, %r15

// CHECK: vcvttsd2usi %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7f,0x08,0x78,0xe9]
vcvttsd2usi %xmm1, %r13d

// CHECK: vcvttsd2usi %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xff,0x08,0x78,0xf9]
vcvttsd2usi %xmm1, %r15

// CHECK: vcvttss2si 256(%rdx), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2c,0xaa,0x00,0x01,0x00,0x00]
vcvttss2si 256(%rdx), %r13d

// CHECK: vcvttss2si 256(%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0xba,0x00,0x01,0x00,0x00]
vcvttss2si 256(%rdx), %r15

// CHECK: vcvttss2si 256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2c,0xac,0x82,0x00,0x01,0x00,0x00]
vcvttss2si 256(%rdx,%rax,4), %r13d

// CHECK: vcvttss2si -256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2c,0xac,0x82,0x00,0xff,0xff,0xff]
vcvttss2si -256(%rdx,%rax,4), %r13d

// CHECK: vcvttss2si 256(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0xbc,0x82,0x00,0x01,0x00,0x00]
vcvttss2si 256(%rdx,%rax,4), %r15

// CHECK: vcvttss2si -256(%rdx,%rax,4), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0xbc,0x82,0x00,0xff,0xff,0xff]
vcvttss2si -256(%rdx,%rax,4), %r15

// CHECK: vcvttss2si 256(%rdx,%rax), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2c,0xac,0x02,0x00,0x01,0x00,0x00]
vcvttss2si 256(%rdx,%rax), %r13d

// CHECK: vcvttss2si 256(%rdx,%rax), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0xbc,0x02,0x00,0x01,0x00,0x00]
vcvttss2si 256(%rdx,%rax), %r15

// CHECK: vcvttss2si 485498096, %r13d
// CHECK: encoding: [0xc5,0x7a,0x2c,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttss2si 485498096, %r13d

// CHECK: vcvttss2si 485498096, %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttss2si 485498096, %r15

// CHECK: vcvttss2si (%rdx), %r13d
// CHECK: encoding: [0xc5,0x7a,0x2c,0x2a]
vcvttss2si (%rdx), %r13d

// CHECK: vcvttss2si (%rdx), %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0x3a]
vcvttss2si (%rdx), %r15

// CHECK: vcvttss2si {sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x18,0x2c,0xef]
vcvttss2si {sae}, %xmm15, %r13d

// CHECK: vcvttss2si {sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x18,0x2c,0xff]
vcvttss2si {sae}, %xmm15, %r15

// CHECK: vcvttss2si {sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x18,0x2c,0xe9]
vcvttss2si {sae}, %xmm1, %r13d

// CHECK: vcvttss2si {sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x18,0x2c,0xf9]
vcvttss2si {sae}, %xmm1, %r15

// CHECK: vcvttss2si %xmm15, %r13d
// CHECK: encoding: [0xc4,0x41,0x7a,0x2c,0xef]
vcvttss2si %xmm15, %r13d

// CHECK: vcvttss2si %xmm15, %r15
// CHECK: encoding: [0xc4,0x41,0xfa,0x2c,0xff]
vcvttss2si %xmm15, %r15

// CHECK: vcvttss2si %xmm1, %r13d
// CHECK: encoding: [0xc5,0x7a,0x2c,0xe9]
vcvttss2si %xmm1, %r13d

// CHECK: vcvttss2si %xmm1, %r15
// CHECK: encoding: [0xc4,0x61,0xfa,0x2c,0xf9]
vcvttss2si %xmm1, %r15

// CHECK: vcvttss2usi 256(%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x78,0x6a,0x40]
vcvttss2usi 256(%rdx), %r13d

// CHECK: vcvttss2usi 256(%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x78,0x7a,0x40]
vcvttss2usi 256(%rdx), %r15

// CHECK: vcvttss2usi 256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x78,0x6c,0x82,0x40]
vcvttss2usi 256(%rdx,%rax,4), %r13d

// CHECK: vcvttss2usi -256(%rdx,%rax,4), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x78,0x6c,0x82,0xc0]
vcvttss2usi -256(%rdx,%rax,4), %r13d

// CHECK: vcvttss2usi 256(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x78,0x7c,0x82,0x40]
vcvttss2usi 256(%rdx,%rax,4), %r15

// CHECK: vcvttss2usi -256(%rdx,%rax,4), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x78,0x7c,0x82,0xc0]
vcvttss2usi -256(%rdx,%rax,4), %r15

// CHECK: vcvttss2usi 256(%rdx,%rax), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x78,0x6c,0x02,0x40]
vcvttss2usi 256(%rdx,%rax), %r13d

// CHECK: vcvttss2usi 256(%rdx,%rax), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x78,0x7c,0x02,0x40]
vcvttss2usi 256(%rdx,%rax), %r15

// CHECK: vcvttss2usi 485498096, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x78,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttss2usi 485498096, %r13d

// CHECK: vcvttss2usi 485498096, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x78,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvttss2usi 485498096, %r15

// CHECK: vcvttss2usi (%rdx), %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x78,0x2a]
vcvttss2usi (%rdx), %r13d

// CHECK: vcvttss2usi (%rdx), %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x78,0x3a]
vcvttss2usi (%rdx), %r15

// CHECK: vcvttss2usi {sae}, %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x18,0x78,0xef]
vcvttss2usi {sae}, %xmm15, %r13d

// CHECK: vcvttss2usi {sae}, %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x18,0x78,0xff]
vcvttss2usi {sae}, %xmm15, %r15

// CHECK: vcvttss2usi {sae}, %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x18,0x78,0xe9]
vcvttss2usi {sae}, %xmm1, %r13d

// CHECK: vcvttss2usi {sae}, %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x18,0x78,0xf9]
vcvttss2usi {sae}, %xmm1, %r15

// CHECK: vcvttss2usi %xmm15, %r13d
// CHECK: encoding: [0x62,0x51,0x7e,0x08,0x78,0xef]
vcvttss2usi %xmm15, %r13d

// CHECK: vcvttss2usi %xmm15, %r15
// CHECK: encoding: [0x62,0x51,0xfe,0x08,0x78,0xff]
vcvttss2usi %xmm15, %r15

// CHECK: vcvttss2usi %xmm1, %r13d
// CHECK: encoding: [0x62,0x71,0x7e,0x08,0x78,0xe9]
vcvttss2usi %xmm1, %r13d

// CHECK: vcvttss2usi %xmm1, %r15
// CHECK: encoding: [0x62,0x71,0xfe,0x08,0x78,0xf9]
vcvttss2usi %xmm1, %r15

// CHECK: vcvtusi2sdl 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x07,0x08,0x7b,0x7c,0x82,0x40]
vcvtusi2sdl 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2sdl -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x07,0x08,0x7b,0x7c,0x82,0xc0]
vcvtusi2sdl -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2sdl 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x4c,0x82,0x40]
vcvtusi2sdl 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2sdl -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x4c,0x82,0xc0]
vcvtusi2sdl -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2sdl 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x07,0x08,0x7b,0x7c,0x02,0x40]
vcvtusi2sdl 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtusi2sdl 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x4c,0x02,0x40]
vcvtusi2sdl 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtusi2sdl 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x07,0x08,0x7b,0x7a,0x40]
vcvtusi2sdl 256(%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2sdl 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x4a,0x40]
vcvtusi2sdl 256(%rdx), %xmm1, %xmm1

// CHECK: vcvtusi2sdl 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x07,0x08,0x7b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2sdl 485498096, %xmm15, %xmm15

// CHECK: vcvtusi2sdl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2sdl 485498096, %xmm1, %xmm1

// CHECK: vcvtusi2sdl %r13d, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x07,0x08,0x7b,0xfd]
vcvtusi2sdl %r13d, %xmm15, %xmm15

// CHECK: vcvtusi2sdl %r13d, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x77,0x08,0x7b,0xcd]
vcvtusi2sdl %r13d, %xmm1, %xmm1

// CHECK: vcvtusi2sdl (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x07,0x08,0x7b,0x3a]
vcvtusi2sdl (%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2sdl (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x0a]
vcvtusi2sdl (%rdx), %xmm1, %xmm1

// CHECK: vcvtusi2sdq 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x87,0x08,0x7b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2sdq 485498096, %xmm15, %xmm15

// CHECK: vcvtusi2sdq 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x7b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2sdq 485498096, %xmm1, %xmm1

// CHECK: vcvtusi2sdq 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x87,0x08,0x7b,0x7c,0x82,0x40]
vcvtusi2sdq 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2sdq -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x87,0x08,0x7b,0x7c,0x82,0xc0]
vcvtusi2sdq -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2sdq 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x7b,0x4c,0x82,0x40]
vcvtusi2sdq 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2sdq -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x7b,0x4c,0x82,0xc0]
vcvtusi2sdq -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2sdq 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x87,0x08,0x7b,0x7c,0x02,0x40]
vcvtusi2sdq 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtusi2sdq 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x7b,0x4c,0x02,0x40]
vcvtusi2sdq 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtusi2sdq 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x87,0x08,0x7b,0x7a,0x40]
vcvtusi2sdq 512(%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2sdq 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x7b,0x4a,0x40]
vcvtusi2sdq 512(%rdx), %xmm1, %xmm1

// CHECK: vcvtusi2sdq %r15, {rd-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x7b,0xff]
vcvtusi2sdq %r15, {rd-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2sdq %r15, {rd-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x38,0x7b,0xcf]
vcvtusi2sdq %r15, {rd-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2sdq %r15, {rn-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x7b,0xff]
vcvtusi2sdq %r15, {rn-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2sdq %r15, {rn-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x18,0x7b,0xcf]
vcvtusi2sdq %r15, {rn-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2sdq %r15, {ru-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x7b,0xff]
vcvtusi2sdq %r15, {ru-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2sdq %r15, {ru-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x58,0x7b,0xcf]
vcvtusi2sdq %r15, {ru-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2sdq %r15, {rz-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x7b,0xff]
vcvtusi2sdq %r15, {rz-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2sdq %r15, {rz-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x78,0x7b,0xcf]
vcvtusi2sdq %r15, {rz-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2sdq %r15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x08,0x7b,0xff]
vcvtusi2sdq %r15, %xmm15, %xmm15

// CHECK: vcvtusi2sdq %r15, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf7,0x08,0x7b,0xcf]
vcvtusi2sdq %r15, %xmm1, %xmm1

// CHECK: vcvtusi2sdq (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x87,0x08,0x7b,0x3a]
vcvtusi2sdq (%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2sdq (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x7b,0x0a]
vcvtusi2sdq (%rdx), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x06,0x08,0x7b,0x7c,0x82,0x40]
vcvtusi2ssl 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2ssl -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x06,0x08,0x7b,0x7c,0x82,0xc0]
vcvtusi2ssl -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2ssl 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x4c,0x82,0x40]
vcvtusi2ssl 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2ssl -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x4c,0x82,0xc0]
vcvtusi2ssl -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x06,0x08,0x7b,0x7c,0x02,0x40]
vcvtusi2ssl 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtusi2ssl 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x4c,0x02,0x40]
vcvtusi2ssl 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x06,0x08,0x7b,0x7a,0x40]
vcvtusi2ssl 256(%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2ssl 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x4a,0x40]
vcvtusi2ssl 256(%rdx), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x06,0x08,0x7b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2ssl 485498096, %xmm15, %xmm15

// CHECK: vcvtusi2ssl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2ssl 485498096, %xmm1, %xmm1

// CHECK: vcvtusi2ssl %r13d, {rd-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x38,0x7b,0xfd]
vcvtusi2ssl %r13d, {rd-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssl %r13d, {rd-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x38,0x7b,0xcd]
vcvtusi2ssl %r13d, {rd-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssl %r13d, {rn-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x7b,0xfd]
vcvtusi2ssl %r13d, {rn-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssl %r13d, {rn-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x18,0x7b,0xcd]
vcvtusi2ssl %r13d, {rn-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssl %r13d, {ru-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x58,0x7b,0xfd]
vcvtusi2ssl %r13d, {ru-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssl %r13d, {ru-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x58,0x7b,0xcd]
vcvtusi2ssl %r13d, {ru-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssl %r13d, {rz-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x78,0x7b,0xfd]
vcvtusi2ssl %r13d, {rz-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssl %r13d, {rz-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x78,0x7b,0xcd]
vcvtusi2ssl %r13d, {rz-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssl %r13d, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x08,0x7b,0xfd]
vcvtusi2ssl %r13d, %xmm15, %xmm15

// CHECK: vcvtusi2ssl %r13d, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0x76,0x08,0x7b,0xcd]
vcvtusi2ssl %r13d, %xmm1, %xmm1

// CHECK: vcvtusi2ssl (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x06,0x08,0x7b,0x3a]
vcvtusi2ssl (%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2ssl (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x0a]
vcvtusi2ssl (%rdx), %xmm1, %xmm1

// CHECK: vcvtusi2ssq 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x86,0x08,0x7b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2ssq 485498096, %xmm15, %xmm15

// CHECK: vcvtusi2ssq 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf6,0x08,0x7b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vcvtusi2ssq 485498096, %xmm1, %xmm1

// CHECK: vcvtusi2ssq 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x86,0x08,0x7b,0x7c,0x82,0x40]
vcvtusi2ssq 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2ssq -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x86,0x08,0x7b,0x7c,0x82,0xc0]
vcvtusi2ssq -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vcvtusi2ssq 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf6,0x08,0x7b,0x4c,0x82,0x40]
vcvtusi2ssq 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2ssq -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf6,0x08,0x7b,0x4c,0x82,0xc0]
vcvtusi2ssq -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vcvtusi2ssq 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x86,0x08,0x7b,0x7c,0x02,0x40]
vcvtusi2ssq 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vcvtusi2ssq 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf6,0x08,0x7b,0x4c,0x02,0x40]
vcvtusi2ssq 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vcvtusi2ssq 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x86,0x08,0x7b,0x7a,0x40]
vcvtusi2ssq 512(%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2ssq 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf6,0x08,0x7b,0x4a,0x40]
vcvtusi2ssq 512(%rdx), %xmm1, %xmm1

// CHECK: vcvtusi2ssq %r15, {rd-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x38,0x7b,0xff]
vcvtusi2ssq %r15, {rd-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssq %r15, {rd-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x38,0x7b,0xcf]
vcvtusi2ssq %r15, {rd-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssq %r15, {rn-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x18,0x7b,0xff]
vcvtusi2ssq %r15, {rn-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssq %r15, {rn-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x18,0x7b,0xcf]
vcvtusi2ssq %r15, {rn-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssq %r15, {ru-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x58,0x7b,0xff]
vcvtusi2ssq %r15, {ru-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssq %r15, {ru-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x58,0x7b,0xcf]
vcvtusi2ssq %r15, {ru-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssq %r15, {rz-sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x78,0x7b,0xff]
vcvtusi2ssq %r15, {rz-sae}, %xmm15, %xmm15

// CHECK: vcvtusi2ssq %r15, {rz-sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x78,0x7b,0xcf]
vcvtusi2ssq %r15, {rz-sae}, %xmm1, %xmm1

// CHECK: vcvtusi2ssq %r15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x86,0x08,0x7b,0xff]
vcvtusi2ssq %r15, %xmm15, %xmm15

// CHECK: vcvtusi2ssq %r15, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xd1,0xf6,0x08,0x7b,0xcf]
vcvtusi2ssq %r15, %xmm1, %xmm1

// CHECK: vcvtusi2ssq (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x71,0x86,0x08,0x7b,0x3a]
vcvtusi2ssq (%rdx), %xmm15, %xmm15

// CHECK: vcvtusi2ssq (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf6,0x08,0x7b,0x0a]
vcvtusi2ssq (%rdx), %xmm1, %xmm1

// CHECK: vdivsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm15, %xmm15

// CHECK: vdivsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vdivsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm1, %xmm1

// CHECK: vdivsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5e,0xbc,0x82,0x00,0x02,0x00,0x00]
vdivsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vdivsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5e,0xbc,0x82,0x00,0xfe,0xff,0xff]
vdivsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vdivsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5e,0x7c,0x82,0x40]
vdivsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vdivsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5e,0x7c,0x82,0xc0]
vdivsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vdivsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5e,0x7c,0x82,0x40]
vdivsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5e,0x7c,0x82,0xc0]
vdivsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5e,0x8c,0x82,0x00,0x02,0x00,0x00]
vdivsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vdivsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5e,0x8c,0x82,0x00,0xfe,0xff,0xff]
vdivsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vdivsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x4c,0x82,0x40]
vdivsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x4c,0x82,0xc0]
vdivsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x4c,0x82,0x40]
vdivsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x4c,0x82,0xc0]
vdivsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5e,0xbc,0x02,0x00,0x02,0x00,0x00]
vdivsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vdivsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5e,0x7c,0x02,0x40]
vdivsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vdivsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5e,0x7c,0x02,0x40]
vdivsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5e,0x8c,0x02,0x00,0x02,0x00,0x00]
vdivsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vdivsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x4c,0x02,0x40]
vdivsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x4c,0x02,0x40]
vdivsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5e,0xba,0x00,0x02,0x00,0x00]
vdivsd 512(%rdx), %xmm15, %xmm15

// CHECK: vdivsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5e,0x7a,0x40]
vdivsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vdivsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5e,0x7a,0x40]
vdivsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5e,0x8a,0x00,0x02,0x00,0x00]
vdivsd 512(%rdx), %xmm1, %xmm1

// CHECK: vdivsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x4a,0x40]
vdivsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x4a,0x40]
vdivsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x5e,0xff]
vdivsd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x3a,0x5e,0xff]
vdivsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xba,0x5e,0xff]
vdivsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x5e,0xc9]
vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x5e,0xc9]
vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x5e,0xc9]
vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5e,0x3a]
vdivsd (%rdx), %xmm15, %xmm15

// CHECK: vdivsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5e,0x3a]
vdivsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vdivsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5e,0x3a]
vdivsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5e,0x0a]
vdivsd (%rdx), %xmm1, %xmm1

// CHECK: vdivsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x0a]
vdivsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x0a]
vdivsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x5e,0xff]
vdivsd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x5e,0xff]
vdivsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x5e,0xff]
vdivsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5e,0xc9]
vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5e,0xc9]
vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5e,0xc9]
vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x5e,0xff]
vdivsd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x5a,0x5e,0xff]
vdivsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xda,0x5e,0xff]
vdivsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x5e,0xc9]
vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x5e,0xc9]
vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x5e,0xc9]
vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x5e,0xff]
vdivsd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x7a,0x5e,0xff]
vdivsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xfa,0x5e,0xff]
vdivsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x5e,0xc9]
vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x5e,0xc9]
vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x5e,0xc9]
vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x5e,0xff]
vdivsd %xmm15, %xmm15, %xmm15

// CHECK: vdivsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x5e,0xff]
vdivsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x5e,0xff]
vdivsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5e,0xc9]
vdivsd %xmm1, %xmm1, %xmm1

// CHECK: vdivsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0xc9]
vdivsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0xc9]
vdivsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5e,0xbc,0x82,0x00,0x01,0x00,0x00]
vdivss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vdivss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5e,0xbc,0x82,0x00,0xff,0xff,0xff]
vdivss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vdivss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5e,0x7c,0x82,0x40]
vdivss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vdivss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5e,0x7c,0x82,0xc0]
vdivss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vdivss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5e,0x7c,0x82,0x40]
vdivss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5e,0x7c,0x82,0xc0]
vdivss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5e,0x8c,0x82,0x00,0x01,0x00,0x00]
vdivss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vdivss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5e,0x8c,0x82,0x00,0xff,0xff,0xff]
vdivss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vdivss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x4c,0x82,0x40]
vdivss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x4c,0x82,0xc0]
vdivss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x4c,0x82,0x40]
vdivss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x4c,0x82,0xc0]
vdivss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5e,0xbc,0x02,0x00,0x01,0x00,0x00]
vdivss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vdivss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5e,0x7c,0x02,0x40]
vdivss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vdivss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5e,0x7c,0x02,0x40]
vdivss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5e,0x8c,0x02,0x00,0x01,0x00,0x00]
vdivss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vdivss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x4c,0x02,0x40]
vdivss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vdivss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x4c,0x02,0x40]
vdivss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5e,0xba,0x00,0x01,0x00,0x00]
vdivss 256(%rdx), %xmm15, %xmm15

// CHECK: vdivss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5e,0x7a,0x40]
vdivss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vdivss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5e,0x7a,0x40]
vdivss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5e,0x8a,0x00,0x01,0x00,0x00]
vdivss 256(%rdx), %xmm1, %xmm1

// CHECK: vdivss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x4a,0x40]
vdivss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vdivss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x4a,0x40]
vdivss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm15, %xmm15

// CHECK: vdivss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vdivss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm1, %xmm1

// CHECK: vdivss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vdivss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x38,0x5e,0xff]
vdivss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x3a,0x5e,0xff]
vdivss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xba,0x5e,0xff]
vdivss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x5e,0xc9]
vdivss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x5e,0xc9]
vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x5e,0xc9]
vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5e,0x3a]
vdivss (%rdx), %xmm15, %xmm15

// CHECK: vdivss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5e,0x3a]
vdivss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vdivss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5e,0x3a]
vdivss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5e,0x0a]
vdivss (%rdx), %xmm1, %xmm1

// CHECK: vdivss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x0a]
vdivss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vdivss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x0a]
vdivss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x5e,0xff]
vdivss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x5e,0xff]
vdivss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x5e,0xff]
vdivss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5e,0xc9]
vdivss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5e,0xc9]
vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5e,0xc9]
vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x58,0x5e,0xff]
vdivss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x5a,0x5e,0xff]
vdivss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xda,0x5e,0xff]
vdivss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x5e,0xc9]
vdivss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x5e,0xc9]
vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x5e,0xc9]
vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x78,0x5e,0xff]
vdivss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vdivss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x7a,0x5e,0xff]
vdivss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xfa,0x5e,0xff]
vdivss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x5e,0xc9]
vdivss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x5e,0xc9]
vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x5e,0xc9]
vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x5e,0xff]
vdivss %xmm15, %xmm15, %xmm15

// CHECK: vdivss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x5e,0xff]
vdivss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vdivss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x5e,0xff]
vdivss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vdivss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5e,0xc9]
vdivss %xmm1, %xmm1, %xmm1

// CHECK: vdivss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0xc9]
vdivss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0xc9]
vdivss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm15, %xmm15

// CHECK: vfixupimmsd $0, 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x55,0x7c,0x82,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfixupimmsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x55,0x7c,0x82,0xc0,0x00]
vfixupimmsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfixupimmsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x55,0x7c,0x82,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x55,0x7c,0x82,0xc0,0x00]
vfixupimmsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x55,0x7c,0x82,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x55,0x7c,0x82,0xc0,0x00]
vfixupimmsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x4c,0x82,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x4c,0x82,0xc0,0x00]
vfixupimmsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x4c,0x82,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x4c,0x82,0xc0,0x00]
vfixupimmsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x4c,0x82,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x4c,0x82,0xc0,0x00]
vfixupimmsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x55,0x7c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfixupimmsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x55,0x7c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x55,0x7c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x4c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x55,0x7a,0x40,0x00]
vfixupimmsd $0, 512(%rdx), %xmm15, %xmm15

// CHECK: vfixupimmsd $0, 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x55,0x7a,0x40,0x00]
vfixupimmsd $0, 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x55,0x7a,0x40,0x00]
vfixupimmsd $0, 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x4a,0x40,0x00]
vfixupimmsd $0, 512(%rdx), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x4a,0x40,0x00]
vfixupimmsd $0, 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x4a,0x40,0x00]
vfixupimmsd $0, 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x55,0x3a,0x00]
vfixupimmsd $0, (%rdx), %xmm15, %xmm15

// CHECK: vfixupimmsd $0, (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x55,0x3a,0x00]
vfixupimmsd $0, (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x55,0x3a,0x00]
vfixupimmsd $0, (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x0a,0x00]
vfixupimmsd $0, (%rdx), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x0a,0x00]
vfixupimmsd $0, (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x0a,0x00]
vfixupimmsd $0, (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x85,0x18,0x55,0xff,0x00]
vfixupimmsd $0, {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfixupimmsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x85,0x1a,0x55,0xff,0x00]
vfixupimmsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x85,0x9a,0x55,0xff,0x00]
vfixupimmsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x18,0x55,0xc9,0x00]
vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x1a,0x55,0xc9,0x00]
vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x9a,0x55,0xc9,0x00]
vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x85,0x08,0x55,0xff,0x00]
vfixupimmsd $0, %xmm15, %xmm15, %xmm15

// CHECK: vfixupimmsd $0, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x85,0x0a,0x55,0xff,0x00]
vfixupimmsd $0, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmsd $0, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x85,0x8a,0x55,0xff,0x00]
vfixupimmsd $0, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmsd $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0xc9,0x00]
vfixupimmsd $0, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0xc9,0x00]
vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0xc9,0x00]
vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x55,0x7c,0x82,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfixupimmss $0, -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x55,0x7c,0x82,0xc0,0x00]
vfixupimmss $0, -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfixupimmss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x55,0x7c,0x82,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x55,0x7c,0x82,0xc0,0x00]
vfixupimmss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x55,0x7c,0x82,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x55,0x7c,0x82,0xc0,0x00]
vfixupimmss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x4c,0x82,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfixupimmss $0, -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x4c,0x82,0xc0,0x00]
vfixupimmss $0, -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfixupimmss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x4c,0x82,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x4c,0x82,0xc0,0x00]
vfixupimmss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x4c,0x82,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x4c,0x82,0xc0,0x00]
vfixupimmss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x55,0x7c,0x02,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfixupimmss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x55,0x7c,0x02,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x55,0x7c,0x02,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x4c,0x02,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfixupimmss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x55,0x7a,0x40,0x00]
vfixupimmss $0, 256(%rdx), %xmm15, %xmm15

// CHECK: vfixupimmss $0, 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x55,0x7a,0x40,0x00]
vfixupimmss $0, 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x55,0x7a,0x40,0x00]
vfixupimmss $0, 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x4a,0x40,0x00]
vfixupimmss $0, 256(%rdx), %xmm1, %xmm1

// CHECK: vfixupimmss $0, 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x4a,0x40,0x00]
vfixupimmss $0, 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x4a,0x40,0x00]
vfixupimmss $0, 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm15, %xmm15

// CHECK: vfixupimmss $0, 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x55,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm1, %xmm1

// CHECK: vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x55,0x3a,0x00]
vfixupimmss $0, (%rdx), %xmm15, %xmm15

// CHECK: vfixupimmss $0, (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x55,0x3a,0x00]
vfixupimmss $0, (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x55,0x3a,0x00]
vfixupimmss $0, (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x0a,0x00]
vfixupimmss $0, (%rdx), %xmm1, %xmm1

// CHECK: vfixupimmss $0, (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x0a,0x00]
vfixupimmss $0, (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x0a,0x00]
vfixupimmss $0, (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x05,0x18,0x55,0xff,0x00]
vfixupimmss $0, {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfixupimmss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x05,0x1a,0x55,0xff,0x00]
vfixupimmss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x05,0x9a,0x55,0xff,0x00]
vfixupimmss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x18,0x55,0xc9,0x00]
vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x1a,0x55,0xc9,0x00]
vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x9a,0x55,0xc9,0x00]
vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x05,0x08,0x55,0xff,0x00]
vfixupimmss $0, %xmm15, %xmm15, %xmm15

// CHECK: vfixupimmss $0, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x05,0x0a,0x55,0xff,0x00]
vfixupimmss $0, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfixupimmss $0, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x05,0x8a,0x55,0xff,0x00]
vfixupimmss $0, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfixupimmss $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0xc9,0x00]
vfixupimmss $0, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0xc9,0x00]
vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0xc9,0x00]
vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm15, %xmm15

// CHECK: vfmadd132sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm1, %xmm1

// CHECK: vfmadd132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0xbc,0x82,0x00,0x02,0x00,0x00]
vfmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x99,0x7c,0x82,0x40]
vfmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x99,0x7c,0x82,0xc0]
vfmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x99,0x7c,0x82,0x40]
vfmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x99,0x7c,0x82,0xc0]
vfmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x8c,0x82,0x00,0x02,0x00,0x00]
vfmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x4c,0x82,0x40]
vfmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x4c,0x82,0xc0]
vfmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x4c,0x82,0x40]
vfmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x4c,0x82,0xc0]
vfmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0xbc,0x02,0x00,0x02,0x00,0x00]
vfmadd132sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x99,0x7c,0x02,0x40]
vfmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x99,0x7c,0x02,0x40]
vfmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x8c,0x02,0x00,0x02,0x00,0x00]
vfmadd132sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x4c,0x02,0x40]
vfmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x4c,0x02,0x40]
vfmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0xba,0x00,0x02,0x00,0x00]
vfmadd132sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x99,0x7a,0x40]
vfmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x99,0x7a,0x40]
vfmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x8a,0x00,0x02,0x00,0x00]
vfmadd132sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x4a,0x40]
vfmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x4a,0x40]
vfmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0x99,0xff]
vfmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0x99,0xff]
vfmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0x99,0xff]
vfmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x99,0xc9]
vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x99,0xc9]
vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x99,0xc9]
vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x99,0x3a]
vfmadd132sd (%rdx), %xmm15, %xmm15

// CHECK: vfmadd132sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x99,0x3a]
vfmadd132sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x99,0x3a]
vfmadd132sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0x0a]
vfmadd132sd (%rdx), %xmm1, %xmm1

// CHECK: vfmadd132sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x0a]
vfmadd132sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x0a]
vfmadd132sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0x99,0xff]
vfmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0x99,0xff]
vfmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0x99,0xff]
vfmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x99,0xc9]
vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x99,0xc9]
vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x99,0xc9]
vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0x99,0xff]
vfmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0x99,0xff]
vfmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0x99,0xff]
vfmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x99,0xc9]
vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x99,0xc9]
vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x99,0xc9]
vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0x99,0xff]
vfmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0x99,0xff]
vfmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0x99,0xff]
vfmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x99,0xc9]
vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x99,0xc9]
vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x99,0xc9]
vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0x99,0xff]
vfmadd132sd %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x99,0xff]
vfmadd132sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x99,0xff]
vfmadd132sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x99,0xc9]
vfmadd132sd %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0xc9]
vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0xc9]
vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0xbc,0x82,0x00,0x01,0x00,0x00]
vfmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0xbc,0x82,0x00,0xff,0xff,0xff]
vfmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x99,0x7c,0x82,0x40]
vfmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x99,0x7c,0x82,0xc0]
vfmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x99,0x7c,0x82,0x40]
vfmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x99,0x7c,0x82,0xc0]
vfmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x8c,0x82,0x00,0x01,0x00,0x00]
vfmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x8c,0x82,0x00,0xff,0xff,0xff]
vfmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x4c,0x82,0x40]
vfmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x4c,0x82,0xc0]
vfmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x4c,0x82,0x40]
vfmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x4c,0x82,0xc0]
vfmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0xbc,0x02,0x00,0x01,0x00,0x00]
vfmadd132ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x99,0x7c,0x02,0x40]
vfmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x99,0x7c,0x02,0x40]
vfmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x8c,0x02,0x00,0x01,0x00,0x00]
vfmadd132ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x4c,0x02,0x40]
vfmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x4c,0x02,0x40]
vfmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0xba,0x00,0x01,0x00,0x00]
vfmadd132ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x99,0x7a,0x40]
vfmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x99,0x7a,0x40]
vfmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x8a,0x00,0x01,0x00,0x00]
vfmadd132ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x4a,0x40]
vfmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x4a,0x40]
vfmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm15, %xmm15

// CHECK: vfmadd132ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x99,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm1, %xmm1

// CHECK: vfmadd132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0x99,0xff]
vfmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0x99,0xff]
vfmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0x99,0xff]
vfmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x99,0xc9]
vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x99,0xc9]
vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x99,0xc9]
vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x99,0x3a]
vfmadd132ss (%rdx), %xmm15, %xmm15

// CHECK: vfmadd132ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x99,0x3a]
vfmadd132ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x99,0x3a]
vfmadd132ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0x0a]
vfmadd132ss (%rdx), %xmm1, %xmm1

// CHECK: vfmadd132ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x0a]
vfmadd132ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x0a]
vfmadd132ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0x99,0xff]
vfmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0x99,0xff]
vfmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0x99,0xff]
vfmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x99,0xc9]
vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x99,0xc9]
vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x99,0xc9]
vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0x99,0xff]
vfmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0x99,0xff]
vfmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0x99,0xff]
vfmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x99,0xc9]
vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x99,0xc9]
vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x99,0xc9]
vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0x99,0xff]
vfmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0x99,0xff]
vfmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0x99,0xff]
vfmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x99,0xc9]
vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x99,0xc9]
vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x99,0xc9]
vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0x99,0xff]
vfmadd132ss %xmm15, %xmm15, %xmm15

// CHECK: vfmadd132ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x99,0xff]
vfmadd132ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd132ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x99,0xff]
vfmadd132ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x99,0xc9]
vfmadd132ss %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0xc9]
vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0xc9]
vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm15, %xmm15

// CHECK: vfmadd213sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm1, %xmm1

// CHECK: vfmadd213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0xbc,0x82,0x00,0x02,0x00,0x00]
vfmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xa9,0x7c,0x82,0x40]
vfmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xa9,0x7c,0x82,0xc0]
vfmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xa9,0x7c,0x82,0x40]
vfmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xa9,0x7c,0x82,0xc0]
vfmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x8c,0x82,0x00,0x02,0x00,0x00]
vfmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x4c,0x82,0x40]
vfmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x4c,0x82,0xc0]
vfmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x4c,0x82,0x40]
vfmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x4c,0x82,0xc0]
vfmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0xbc,0x02,0x00,0x02,0x00,0x00]
vfmadd213sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xa9,0x7c,0x02,0x40]
vfmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xa9,0x7c,0x02,0x40]
vfmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x8c,0x02,0x00,0x02,0x00,0x00]
vfmadd213sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x4c,0x02,0x40]
vfmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x4c,0x02,0x40]
vfmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0xba,0x00,0x02,0x00,0x00]
vfmadd213sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xa9,0x7a,0x40]
vfmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xa9,0x7a,0x40]
vfmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x8a,0x00,0x02,0x00,0x00]
vfmadd213sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x4a,0x40]
vfmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x4a,0x40]
vfmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xa9,0xff]
vfmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xa9,0xff]
vfmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xa9,0xff]
vfmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xa9,0xc9]
vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xa9,0xc9]
vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xa9,0xc9]
vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xa9,0x3a]
vfmadd213sd (%rdx), %xmm15, %xmm15

// CHECK: vfmadd213sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xa9,0x3a]
vfmadd213sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xa9,0x3a]
vfmadd213sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0x0a]
vfmadd213sd (%rdx), %xmm1, %xmm1

// CHECK: vfmadd213sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x0a]
vfmadd213sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x0a]
vfmadd213sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xa9,0xff]
vfmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xa9,0xff]
vfmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xa9,0xff]
vfmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xa9,0xc9]
vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xa9,0xc9]
vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xa9,0xc9]
vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xa9,0xff]
vfmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xa9,0xff]
vfmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0xa9,0xff]
vfmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xa9,0xc9]
vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xa9,0xc9]
vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xa9,0xc9]
vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0xa9,0xff]
vfmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0xa9,0xff]
vfmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0xa9,0xff]
vfmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xa9,0xc9]
vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xa9,0xc9]
vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xa9,0xc9]
vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0xa9,0xff]
vfmadd213sd %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0xa9,0xff]
vfmadd213sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0xa9,0xff]
vfmadd213sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xa9,0xc9]
vfmadd213sd %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0xc9]
vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0xc9]
vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0xbc,0x82,0x00,0x01,0x00,0x00]
vfmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0xbc,0x82,0x00,0xff,0xff,0xff]
vfmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xa9,0x7c,0x82,0x40]
vfmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xa9,0x7c,0x82,0xc0]
vfmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xa9,0x7c,0x82,0x40]
vfmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xa9,0x7c,0x82,0xc0]
vfmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x8c,0x82,0x00,0x01,0x00,0x00]
vfmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x8c,0x82,0x00,0xff,0xff,0xff]
vfmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x4c,0x82,0x40]
vfmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x4c,0x82,0xc0]
vfmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x4c,0x82,0x40]
vfmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x4c,0x82,0xc0]
vfmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0xbc,0x02,0x00,0x01,0x00,0x00]
vfmadd213ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xa9,0x7c,0x02,0x40]
vfmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xa9,0x7c,0x02,0x40]
vfmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x8c,0x02,0x00,0x01,0x00,0x00]
vfmadd213ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x4c,0x02,0x40]
vfmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x4c,0x02,0x40]
vfmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0xba,0x00,0x01,0x00,0x00]
vfmadd213ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xa9,0x7a,0x40]
vfmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xa9,0x7a,0x40]
vfmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x8a,0x00,0x01,0x00,0x00]
vfmadd213ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x4a,0x40]
vfmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x4a,0x40]
vfmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm15, %xmm15

// CHECK: vfmadd213ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xa9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm1, %xmm1

// CHECK: vfmadd213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xa9,0xff]
vfmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xa9,0xff]
vfmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xa9,0xff]
vfmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xa9,0xc9]
vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xa9,0xc9]
vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xa9,0xc9]
vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xa9,0x3a]
vfmadd213ss (%rdx), %xmm15, %xmm15

// CHECK: vfmadd213ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xa9,0x3a]
vfmadd213ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xa9,0x3a]
vfmadd213ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0x0a]
vfmadd213ss (%rdx), %xmm1, %xmm1

// CHECK: vfmadd213ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x0a]
vfmadd213ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x0a]
vfmadd213ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xa9,0xff]
vfmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xa9,0xff]
vfmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xa9,0xff]
vfmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xa9,0xc9]
vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xa9,0xc9]
vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xa9,0xc9]
vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xa9,0xff]
vfmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xa9,0xff]
vfmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xa9,0xff]
vfmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xa9,0xc9]
vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xa9,0xc9]
vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xa9,0xc9]
vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xa9,0xff]
vfmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xa9,0xff]
vfmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xa9,0xff]
vfmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xa9,0xc9]
vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xa9,0xc9]
vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xa9,0xc9]
vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xa9,0xff]
vfmadd213ss %xmm15, %xmm15, %xmm15

// CHECK: vfmadd213ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xa9,0xff]
vfmadd213ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd213ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xa9,0xff]
vfmadd213ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xa9,0xc9]
vfmadd213ss %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0xc9]
vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0xc9]
vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm15, %xmm15

// CHECK: vfmadd231sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm1, %xmm1

// CHECK: vfmadd231sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0xbc,0x82,0x00,0x02,0x00,0x00]
vfmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xb9,0x7c,0x82,0x40]
vfmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xb9,0x7c,0x82,0xc0]
vfmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xb9,0x7c,0x82,0x40]
vfmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xb9,0x7c,0x82,0xc0]
vfmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x8c,0x82,0x00,0x02,0x00,0x00]
vfmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x4c,0x82,0x40]
vfmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x4c,0x82,0xc0]
vfmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x4c,0x82,0x40]
vfmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x4c,0x82,0xc0]
vfmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0xbc,0x02,0x00,0x02,0x00,0x00]
vfmadd231sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xb9,0x7c,0x02,0x40]
vfmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xb9,0x7c,0x02,0x40]
vfmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x8c,0x02,0x00,0x02,0x00,0x00]
vfmadd231sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x4c,0x02,0x40]
vfmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x4c,0x02,0x40]
vfmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0xba,0x00,0x02,0x00,0x00]
vfmadd231sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xb9,0x7a,0x40]
vfmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xb9,0x7a,0x40]
vfmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x8a,0x00,0x02,0x00,0x00]
vfmadd231sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x4a,0x40]
vfmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x4a,0x40]
vfmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xb9,0xff]
vfmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xb9,0xff]
vfmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xb9,0xff]
vfmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xb9,0xc9]
vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xb9,0xc9]
vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xb9,0xc9]
vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xb9,0x3a]
vfmadd231sd (%rdx), %xmm15, %xmm15

// CHECK: vfmadd231sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xb9,0x3a]
vfmadd231sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xb9,0x3a]
vfmadd231sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0x0a]
vfmadd231sd (%rdx), %xmm1, %xmm1

// CHECK: vfmadd231sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x0a]
vfmadd231sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x0a]
vfmadd231sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xb9,0xff]
vfmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xb9,0xff]
vfmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xb9,0xff]
vfmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xb9,0xc9]
vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xb9,0xc9]
vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xb9,0xc9]
vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xb9,0xff]
vfmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xb9,0xff]
vfmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0xb9,0xff]
vfmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xb9,0xc9]
vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xb9,0xc9]
vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xb9,0xc9]
vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0xb9,0xff]
vfmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0xb9,0xff]
vfmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0xb9,0xff]
vfmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xb9,0xc9]
vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xb9,0xc9]
vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xb9,0xc9]
vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0xb9,0xff]
vfmadd231sd %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0xb9,0xff]
vfmadd231sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0xb9,0xff]
vfmadd231sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xb9,0xc9]
vfmadd231sd %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0xc9]
vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0xc9]
vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0xbc,0x82,0x00,0x01,0x00,0x00]
vfmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0xbc,0x82,0x00,0xff,0xff,0xff]
vfmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xb9,0x7c,0x82,0x40]
vfmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xb9,0x7c,0x82,0xc0]
vfmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xb9,0x7c,0x82,0x40]
vfmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xb9,0x7c,0x82,0xc0]
vfmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x8c,0x82,0x00,0x01,0x00,0x00]
vfmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x8c,0x82,0x00,0xff,0xff,0xff]
vfmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x4c,0x82,0x40]
vfmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x4c,0x82,0xc0]
vfmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x4c,0x82,0x40]
vfmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x4c,0x82,0xc0]
vfmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0xbc,0x02,0x00,0x01,0x00,0x00]
vfmadd231ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xb9,0x7c,0x02,0x40]
vfmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xb9,0x7c,0x02,0x40]
vfmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x8c,0x02,0x00,0x01,0x00,0x00]
vfmadd231ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x4c,0x02,0x40]
vfmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x4c,0x02,0x40]
vfmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0xba,0x00,0x01,0x00,0x00]
vfmadd231ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xb9,0x7a,0x40]
vfmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xb9,0x7a,0x40]
vfmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x8a,0x00,0x01,0x00,0x00]
vfmadd231ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x4a,0x40]
vfmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x4a,0x40]
vfmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm15, %xmm15

// CHECK: vfmadd231ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xb9,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm1, %xmm1

// CHECK: vfmadd231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xb9,0xff]
vfmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xb9,0xff]
vfmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xb9,0xff]
vfmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xb9,0xc9]
vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xb9,0xc9]
vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xb9,0xc9]
vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xb9,0x3a]
vfmadd231ss (%rdx), %xmm15, %xmm15

// CHECK: vfmadd231ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xb9,0x3a]
vfmadd231ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xb9,0x3a]
vfmadd231ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0x0a]
vfmadd231ss (%rdx), %xmm1, %xmm1

// CHECK: vfmadd231ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x0a]
vfmadd231ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x0a]
vfmadd231ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xb9,0xff]
vfmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xb9,0xff]
vfmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xb9,0xff]
vfmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xb9,0xc9]
vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xb9,0xc9]
vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xb9,0xc9]
vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xb9,0xff]
vfmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xb9,0xff]
vfmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xb9,0xff]
vfmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xb9,0xc9]
vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xb9,0xc9]
vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xb9,0xc9]
vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xb9,0xff]
vfmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xb9,0xff]
vfmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xb9,0xff]
vfmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xb9,0xc9]
vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xb9,0xc9]
vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xb9,0xc9]
vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xb9,0xff]
vfmadd231ss %xmm15, %xmm15, %xmm15

// CHECK: vfmadd231ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xb9,0xff]
vfmadd231ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmadd231ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xb9,0xff]
vfmadd231ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmadd231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xb9,0xc9]
vfmadd231ss %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0xc9]
vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0xc9]
vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm15, %xmm15

// CHECK: vfmsub132sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm1, %xmm1

// CHECK: vfmsub132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0xbc,0x82,0x00,0x02,0x00,0x00]
vfmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9b,0x7c,0x82,0x40]
vfmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9b,0x7c,0x82,0xc0]
vfmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9b,0x7c,0x82,0x40]
vfmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9b,0x7c,0x82,0xc0]
vfmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x8c,0x82,0x00,0x02,0x00,0x00]
vfmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x4c,0x82,0x40]
vfmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x4c,0x82,0xc0]
vfmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x4c,0x82,0x40]
vfmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x4c,0x82,0xc0]
vfmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0xbc,0x02,0x00,0x02,0x00,0x00]
vfmsub132sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9b,0x7c,0x02,0x40]
vfmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9b,0x7c,0x02,0x40]
vfmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x8c,0x02,0x00,0x02,0x00,0x00]
vfmsub132sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x4c,0x02,0x40]
vfmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x4c,0x02,0x40]
vfmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0xba,0x00,0x02,0x00,0x00]
vfmsub132sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9b,0x7a,0x40]
vfmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9b,0x7a,0x40]
vfmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x8a,0x00,0x02,0x00,0x00]
vfmsub132sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x4a,0x40]
vfmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x4a,0x40]
vfmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0x9b,0xff]
vfmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0x9b,0xff]
vfmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0x9b,0xff]
vfmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x9b,0xc9]
vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x9b,0xc9]
vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x9b,0xc9]
vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9b,0x3a]
vfmsub132sd (%rdx), %xmm15, %xmm15

// CHECK: vfmsub132sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9b,0x3a]
vfmsub132sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9b,0x3a]
vfmsub132sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0x0a]
vfmsub132sd (%rdx), %xmm1, %xmm1

// CHECK: vfmsub132sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x0a]
vfmsub132sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x0a]
vfmsub132sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0x9b,0xff]
vfmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0x9b,0xff]
vfmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0x9b,0xff]
vfmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x9b,0xc9]
vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x9b,0xc9]
vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x9b,0xc9]
vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0x9b,0xff]
vfmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0x9b,0xff]
vfmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0x9b,0xff]
vfmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x9b,0xc9]
vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x9b,0xc9]
vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x9b,0xc9]
vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0x9b,0xff]
vfmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0x9b,0xff]
vfmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0x9b,0xff]
vfmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x9b,0xc9]
vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x9b,0xc9]
vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x9b,0xc9]
vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0x9b,0xff]
vfmsub132sd %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x9b,0xff]
vfmsub132sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x9b,0xff]
vfmsub132sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9b,0xc9]
vfmsub132sd %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0xc9]
vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0xc9]
vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0xbc,0x82,0x00,0x01,0x00,0x00]
vfmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0xbc,0x82,0x00,0xff,0xff,0xff]
vfmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9b,0x7c,0x82,0x40]
vfmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9b,0x7c,0x82,0xc0]
vfmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9b,0x7c,0x82,0x40]
vfmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9b,0x7c,0x82,0xc0]
vfmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x8c,0x82,0x00,0x01,0x00,0x00]
vfmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x8c,0x82,0x00,0xff,0xff,0xff]
vfmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x4c,0x82,0x40]
vfmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x4c,0x82,0xc0]
vfmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x4c,0x82,0x40]
vfmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x4c,0x82,0xc0]
vfmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0xbc,0x02,0x00,0x01,0x00,0x00]
vfmsub132ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9b,0x7c,0x02,0x40]
vfmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9b,0x7c,0x02,0x40]
vfmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x8c,0x02,0x00,0x01,0x00,0x00]
vfmsub132ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x4c,0x02,0x40]
vfmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x4c,0x02,0x40]
vfmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0xba,0x00,0x01,0x00,0x00]
vfmsub132ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9b,0x7a,0x40]
vfmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9b,0x7a,0x40]
vfmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x8a,0x00,0x01,0x00,0x00]
vfmsub132ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x4a,0x40]
vfmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x4a,0x40]
vfmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm15, %xmm15

// CHECK: vfmsub132ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm1, %xmm1

// CHECK: vfmsub132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0x9b,0xff]
vfmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0x9b,0xff]
vfmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0x9b,0xff]
vfmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x9b,0xc9]
vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x9b,0xc9]
vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x9b,0xc9]
vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9b,0x3a]
vfmsub132ss (%rdx), %xmm15, %xmm15

// CHECK: vfmsub132ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9b,0x3a]
vfmsub132ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9b,0x3a]
vfmsub132ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0x0a]
vfmsub132ss (%rdx), %xmm1, %xmm1

// CHECK: vfmsub132ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x0a]
vfmsub132ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x0a]
vfmsub132ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0x9b,0xff]
vfmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0x9b,0xff]
vfmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0x9b,0xff]
vfmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x9b,0xc9]
vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x9b,0xc9]
vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x9b,0xc9]
vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0x9b,0xff]
vfmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0x9b,0xff]
vfmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0x9b,0xff]
vfmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x9b,0xc9]
vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x9b,0xc9]
vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x9b,0xc9]
vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0x9b,0xff]
vfmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0x9b,0xff]
vfmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0x9b,0xff]
vfmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x9b,0xc9]
vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x9b,0xc9]
vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x9b,0xc9]
vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0x9b,0xff]
vfmsub132ss %xmm15, %xmm15, %xmm15

// CHECK: vfmsub132ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x9b,0xff]
vfmsub132ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub132ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x9b,0xff]
vfmsub132ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9b,0xc9]
vfmsub132ss %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0xc9]
vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0xc9]
vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm15, %xmm15

// CHECK: vfmsub213sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm1, %xmm1

// CHECK: vfmsub213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0xbc,0x82,0x00,0x02,0x00,0x00]
vfmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xab,0x7c,0x82,0x40]
vfmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xab,0x7c,0x82,0xc0]
vfmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xab,0x7c,0x82,0x40]
vfmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xab,0x7c,0x82,0xc0]
vfmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x8c,0x82,0x00,0x02,0x00,0x00]
vfmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x4c,0x82,0x40]
vfmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x4c,0x82,0xc0]
vfmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x4c,0x82,0x40]
vfmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x4c,0x82,0xc0]
vfmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0xbc,0x02,0x00,0x02,0x00,0x00]
vfmsub213sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xab,0x7c,0x02,0x40]
vfmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xab,0x7c,0x02,0x40]
vfmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x8c,0x02,0x00,0x02,0x00,0x00]
vfmsub213sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x4c,0x02,0x40]
vfmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x4c,0x02,0x40]
vfmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0xba,0x00,0x02,0x00,0x00]
vfmsub213sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xab,0x7a,0x40]
vfmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xab,0x7a,0x40]
vfmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x8a,0x00,0x02,0x00,0x00]
vfmsub213sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x4a,0x40]
vfmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x4a,0x40]
vfmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xab,0xff]
vfmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xab,0xff]
vfmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xab,0xff]
vfmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xab,0xc9]
vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xab,0xc9]
vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xab,0xc9]
vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xab,0x3a]
vfmsub213sd (%rdx), %xmm15, %xmm15

// CHECK: vfmsub213sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xab,0x3a]
vfmsub213sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xab,0x3a]
vfmsub213sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0x0a]
vfmsub213sd (%rdx), %xmm1, %xmm1

// CHECK: vfmsub213sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x0a]
vfmsub213sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x0a]
vfmsub213sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xab,0xff]
vfmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xab,0xff]
vfmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xab,0xff]
vfmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xab,0xc9]
vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xab,0xc9]
vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xab,0xc9]
vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xab,0xff]
vfmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xab,0xff]
vfmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0xab,0xff]
vfmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xab,0xc9]
vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xab,0xc9]
vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xab,0xc9]
vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0xab,0xff]
vfmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0xab,0xff]
vfmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0xab,0xff]
vfmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xab,0xc9]
vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xab,0xc9]
vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xab,0xc9]
vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0xab,0xff]
vfmsub213sd %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0xab,0xff]
vfmsub213sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0xab,0xff]
vfmsub213sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xab,0xc9]
vfmsub213sd %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0xc9]
vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0xc9]
vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0xbc,0x82,0x00,0x01,0x00,0x00]
vfmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0xbc,0x82,0x00,0xff,0xff,0xff]
vfmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xab,0x7c,0x82,0x40]
vfmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xab,0x7c,0x82,0xc0]
vfmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xab,0x7c,0x82,0x40]
vfmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xab,0x7c,0x82,0xc0]
vfmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x8c,0x82,0x00,0x01,0x00,0x00]
vfmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x8c,0x82,0x00,0xff,0xff,0xff]
vfmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x4c,0x82,0x40]
vfmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x4c,0x82,0xc0]
vfmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x4c,0x82,0x40]
vfmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x4c,0x82,0xc0]
vfmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0xbc,0x02,0x00,0x01,0x00,0x00]
vfmsub213ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xab,0x7c,0x02,0x40]
vfmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xab,0x7c,0x02,0x40]
vfmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x8c,0x02,0x00,0x01,0x00,0x00]
vfmsub213ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x4c,0x02,0x40]
vfmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x4c,0x02,0x40]
vfmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0xba,0x00,0x01,0x00,0x00]
vfmsub213ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xab,0x7a,0x40]
vfmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xab,0x7a,0x40]
vfmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x8a,0x00,0x01,0x00,0x00]
vfmsub213ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x4a,0x40]
vfmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x4a,0x40]
vfmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm15, %xmm15

// CHECK: vfmsub213ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xab,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm1, %xmm1

// CHECK: vfmsub213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xab,0xff]
vfmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xab,0xff]
vfmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xab,0xff]
vfmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xab,0xc9]
vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xab,0xc9]
vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xab,0xc9]
vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xab,0x3a]
vfmsub213ss (%rdx), %xmm15, %xmm15

// CHECK: vfmsub213ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xab,0x3a]
vfmsub213ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xab,0x3a]
vfmsub213ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0x0a]
vfmsub213ss (%rdx), %xmm1, %xmm1

// CHECK: vfmsub213ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x0a]
vfmsub213ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x0a]
vfmsub213ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xab,0xff]
vfmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xab,0xff]
vfmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xab,0xff]
vfmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xab,0xc9]
vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xab,0xc9]
vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xab,0xc9]
vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xab,0xff]
vfmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xab,0xff]
vfmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xab,0xff]
vfmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xab,0xc9]
vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xab,0xc9]
vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xab,0xc9]
vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xab,0xff]
vfmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xab,0xff]
vfmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xab,0xff]
vfmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xab,0xc9]
vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xab,0xc9]
vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xab,0xc9]
vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xab,0xff]
vfmsub213ss %xmm15, %xmm15, %xmm15

// CHECK: vfmsub213ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xab,0xff]
vfmsub213ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub213ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xab,0xff]
vfmsub213ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xab,0xc9]
vfmsub213ss %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0xc9]
vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0xc9]
vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm15, %xmm15

// CHECK: vfmsub231sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm1, %xmm1

// CHECK: vfmsub231sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0xbc,0x82,0x00,0x02,0x00,0x00]
vfmsub231sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub231sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfmsub231sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbb,0x7c,0x82,0x40]
vfmsub231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbb,0x7c,0x82,0xc0]
vfmsub231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbb,0x7c,0x82,0x40]
vfmsub231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbb,0x7c,0x82,0xc0]
vfmsub231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x8c,0x82,0x00,0x02,0x00,0x00]
vfmsub231sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub231sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfmsub231sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x4c,0x82,0x40]
vfmsub231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x4c,0x82,0xc0]
vfmsub231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x4c,0x82,0x40]
vfmsub231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x4c,0x82,0xc0]
vfmsub231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0xbc,0x02,0x00,0x02,0x00,0x00]
vfmsub231sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmsub231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbb,0x7c,0x02,0x40]
vfmsub231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbb,0x7c,0x02,0x40]
vfmsub231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x8c,0x02,0x00,0x02,0x00,0x00]
vfmsub231sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmsub231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x4c,0x02,0x40]
vfmsub231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x4c,0x02,0x40]
vfmsub231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0xba,0x00,0x02,0x00,0x00]
vfmsub231sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfmsub231sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbb,0x7a,0x40]
vfmsub231sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbb,0x7a,0x40]
vfmsub231sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x8a,0x00,0x02,0x00,0x00]
vfmsub231sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfmsub231sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x4a,0x40]
vfmsub231sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x4a,0x40]
vfmsub231sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xbb,0xff]
vfmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xbb,0xff]
vfmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xbb,0xff]
vfmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xbb,0xc9]
vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xbb,0xc9]
vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xbb,0xc9]
vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbb,0x3a]
vfmsub231sd (%rdx), %xmm15, %xmm15

// CHECK: vfmsub231sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbb,0x3a]
vfmsub231sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbb,0x3a]
vfmsub231sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0x0a]
vfmsub231sd (%rdx), %xmm1, %xmm1

// CHECK: vfmsub231sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x0a]
vfmsub231sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x0a]
vfmsub231sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xbb,0xff]
vfmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xbb,0xff]
vfmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xbb,0xff]
vfmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xbb,0xc9]
vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xbb,0xc9]
vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xbb,0xc9]
vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xbb,0xff]
vfmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xbb,0xff]
vfmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0xbb,0xff]
vfmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xbb,0xc9]
vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xbb,0xc9]
vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xbb,0xc9]
vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0xbb,0xff]
vfmsub231sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0xbb,0xff]
vfmsub231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0xbb,0xff]
vfmsub231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xbb,0xc9]
vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xbb,0xc9]
vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xbb,0xc9]
vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0xbb,0xff]
vfmsub231sd %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0xbb,0xff]
vfmsub231sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0xbb,0xff]
vfmsub231sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbb,0xc9]
vfmsub231sd %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0xc9]
vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0xc9]
vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0xbc,0x82,0x00,0x01,0x00,0x00]
vfmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0xbc,0x82,0x00,0xff,0xff,0xff]
vfmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbb,0x7c,0x82,0x40]
vfmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbb,0x7c,0x82,0xc0]
vfmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbb,0x7c,0x82,0x40]
vfmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbb,0x7c,0x82,0xc0]
vfmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x8c,0x82,0x00,0x01,0x00,0x00]
vfmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x8c,0x82,0x00,0xff,0xff,0xff]
vfmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x4c,0x82,0x40]
vfmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x4c,0x82,0xc0]
vfmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x4c,0x82,0x40]
vfmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x4c,0x82,0xc0]
vfmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0xbc,0x02,0x00,0x01,0x00,0x00]
vfmsub231ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbb,0x7c,0x02,0x40]
vfmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbb,0x7c,0x02,0x40]
vfmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x8c,0x02,0x00,0x01,0x00,0x00]
vfmsub231ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x4c,0x02,0x40]
vfmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x4c,0x02,0x40]
vfmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0xba,0x00,0x01,0x00,0x00]
vfmsub231ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbb,0x7a,0x40]
vfmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbb,0x7a,0x40]
vfmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x8a,0x00,0x01,0x00,0x00]
vfmsub231ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x4a,0x40]
vfmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x4a,0x40]
vfmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm15, %xmm15

// CHECK: vfmsub231ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbb,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm1, %xmm1

// CHECK: vfmsub231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xbb,0xff]
vfmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xbb,0xff]
vfmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xbb,0xff]
vfmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xbb,0xc9]
vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xbb,0xc9]
vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xbb,0xc9]
vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbb,0x3a]
vfmsub231ss (%rdx), %xmm15, %xmm15

// CHECK: vfmsub231ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbb,0x3a]
vfmsub231ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbb,0x3a]
vfmsub231ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0x0a]
vfmsub231ss (%rdx), %xmm1, %xmm1

// CHECK: vfmsub231ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x0a]
vfmsub231ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x0a]
vfmsub231ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xbb,0xff]
vfmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xbb,0xff]
vfmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xbb,0xff]
vfmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xbb,0xc9]
vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xbb,0xc9]
vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xbb,0xc9]
vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xbb,0xff]
vfmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xbb,0xff]
vfmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xbb,0xff]
vfmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xbb,0xc9]
vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xbb,0xc9]
vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xbb,0xc9]
vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xbb,0xff]
vfmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xbb,0xff]
vfmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xbb,0xff]
vfmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xbb,0xc9]
vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xbb,0xc9]
vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xbb,0xc9]
vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xbb,0xff]
vfmsub231ss %xmm15, %xmm15, %xmm15

// CHECK: vfmsub231ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xbb,0xff]
vfmsub231ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfmsub231ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xbb,0xff]
vfmsub231ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfmsub231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbb,0xc9]
vfmsub231ss %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0xc9]
vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0xc9]
vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm15, %xmm15

// CHECK: vfnmadd132sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm1, %xmm1

// CHECK: vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0xbc,0x82,0x00,0x02,0x00,0x00]
vfnmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfnmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9d,0x7c,0x82,0x40]
vfnmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9d,0x7c,0x82,0xc0]
vfnmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9d,0x7c,0x82,0x40]
vfnmadd132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9d,0x7c,0x82,0xc0]
vfnmadd132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x8c,0x82,0x00,0x02,0x00,0x00]
vfnmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfnmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x4c,0x82,0x40]
vfnmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x4c,0x82,0xc0]
vfnmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x4c,0x82,0x40]
vfnmadd132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x4c,0x82,0xc0]
vfnmadd132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0xbc,0x02,0x00,0x02,0x00,0x00]
vfnmadd132sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9d,0x7c,0x02,0x40]
vfnmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9d,0x7c,0x02,0x40]
vfnmadd132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x8c,0x02,0x00,0x02,0x00,0x00]
vfnmadd132sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x4c,0x02,0x40]
vfnmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x4c,0x02,0x40]
vfnmadd132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0xba,0x00,0x02,0x00,0x00]
vfnmadd132sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfnmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9d,0x7a,0x40]
vfnmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9d,0x7a,0x40]
vfnmadd132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x8a,0x00,0x02,0x00,0x00]
vfnmadd132sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfnmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x4a,0x40]
vfnmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x4a,0x40]
vfnmadd132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0x9d,0xff]
vfnmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0x9d,0xff]
vfnmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0x9d,0xff]
vfnmadd132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x9d,0xc9]
vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x9d,0xc9]
vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x9d,0xc9]
vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9d,0x3a]
vfnmadd132sd (%rdx), %xmm15, %xmm15

// CHECK: vfnmadd132sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9d,0x3a]
vfnmadd132sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9d,0x3a]
vfnmadd132sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0x0a]
vfnmadd132sd (%rdx), %xmm1, %xmm1

// CHECK: vfnmadd132sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x0a]
vfnmadd132sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x0a]
vfnmadd132sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0x9d,0xff]
vfnmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0x9d,0xff]
vfnmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0x9d,0xff]
vfnmadd132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x9d,0xc9]
vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x9d,0xc9]
vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x9d,0xc9]
vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0x9d,0xff]
vfnmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0x9d,0xff]
vfnmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0x9d,0xff]
vfnmadd132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x9d,0xc9]
vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x9d,0xc9]
vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x9d,0xc9]
vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0x9d,0xff]
vfnmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0x9d,0xff]
vfnmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0x9d,0xff]
vfnmadd132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x9d,0xc9]
vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x9d,0xc9]
vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x9d,0xc9]
vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0x9d,0xff]
vfnmadd132sd %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x9d,0xff]
vfnmadd132sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x9d,0xff]
vfnmadd132sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9d,0xc9]
vfnmadd132sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0xc9]
vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0xc9]
vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0xbc,0x82,0x00,0x01,0x00,0x00]
vfnmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0xbc,0x82,0x00,0xff,0xff,0xff]
vfnmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9d,0x7c,0x82,0x40]
vfnmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9d,0x7c,0x82,0xc0]
vfnmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9d,0x7c,0x82,0x40]
vfnmadd132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9d,0x7c,0x82,0xc0]
vfnmadd132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x8c,0x82,0x00,0x01,0x00,0x00]
vfnmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x8c,0x82,0x00,0xff,0xff,0xff]
vfnmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x4c,0x82,0x40]
vfnmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x4c,0x82,0xc0]
vfnmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x4c,0x82,0x40]
vfnmadd132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x4c,0x82,0xc0]
vfnmadd132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0xbc,0x02,0x00,0x01,0x00,0x00]
vfnmadd132ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9d,0x7c,0x02,0x40]
vfnmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9d,0x7c,0x02,0x40]
vfnmadd132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x8c,0x02,0x00,0x01,0x00,0x00]
vfnmadd132ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x4c,0x02,0x40]
vfnmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x4c,0x02,0x40]
vfnmadd132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0xba,0x00,0x01,0x00,0x00]
vfnmadd132ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfnmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9d,0x7a,0x40]
vfnmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9d,0x7a,0x40]
vfnmadd132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x8a,0x00,0x01,0x00,0x00]
vfnmadd132ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfnmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x4a,0x40]
vfnmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x4a,0x40]
vfnmadd132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm15, %xmm15

// CHECK: vfnmadd132ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm1, %xmm1

// CHECK: vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0x9d,0xff]
vfnmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0x9d,0xff]
vfnmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0x9d,0xff]
vfnmadd132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x9d,0xc9]
vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x9d,0xc9]
vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x9d,0xc9]
vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9d,0x3a]
vfnmadd132ss (%rdx), %xmm15, %xmm15

// CHECK: vfnmadd132ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9d,0x3a]
vfnmadd132ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9d,0x3a]
vfnmadd132ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0x0a]
vfnmadd132ss (%rdx), %xmm1, %xmm1

// CHECK: vfnmadd132ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x0a]
vfnmadd132ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x0a]
vfnmadd132ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0x9d,0xff]
vfnmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0x9d,0xff]
vfnmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0x9d,0xff]
vfnmadd132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x9d,0xc9]
vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x9d,0xc9]
vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x9d,0xc9]
vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0x9d,0xff]
vfnmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0x9d,0xff]
vfnmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0x9d,0xff]
vfnmadd132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x9d,0xc9]
vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x9d,0xc9]
vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x9d,0xc9]
vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0x9d,0xff]
vfnmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0x9d,0xff]
vfnmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0x9d,0xff]
vfnmadd132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x9d,0xc9]
vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x9d,0xc9]
vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x9d,0xc9]
vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0x9d,0xff]
vfnmadd132ss %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd132ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x9d,0xff]
vfnmadd132ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd132ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x9d,0xff]
vfnmadd132ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9d,0xc9]
vfnmadd132ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0xc9]
vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0xc9]
vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm15, %xmm15

// CHECK: vfnmadd213sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm1, %xmm1

// CHECK: vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0xbc,0x82,0x00,0x02,0x00,0x00]
vfnmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfnmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xad,0x7c,0x82,0x40]
vfnmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xad,0x7c,0x82,0xc0]
vfnmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xad,0x7c,0x82,0x40]
vfnmadd213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xad,0x7c,0x82,0xc0]
vfnmadd213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x8c,0x82,0x00,0x02,0x00,0x00]
vfnmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfnmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x4c,0x82,0x40]
vfnmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x4c,0x82,0xc0]
vfnmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x4c,0x82,0x40]
vfnmadd213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x4c,0x82,0xc0]
vfnmadd213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0xbc,0x02,0x00,0x02,0x00,0x00]
vfnmadd213sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xad,0x7c,0x02,0x40]
vfnmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xad,0x7c,0x02,0x40]
vfnmadd213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x8c,0x02,0x00,0x02,0x00,0x00]
vfnmadd213sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x4c,0x02,0x40]
vfnmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x4c,0x02,0x40]
vfnmadd213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0xba,0x00,0x02,0x00,0x00]
vfnmadd213sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfnmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xad,0x7a,0x40]
vfnmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xad,0x7a,0x40]
vfnmadd213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x8a,0x00,0x02,0x00,0x00]
vfnmadd213sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfnmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x4a,0x40]
vfnmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x4a,0x40]
vfnmadd213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xad,0xff]
vfnmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xad,0xff]
vfnmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xad,0xff]
vfnmadd213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xad,0xc9]
vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xad,0xc9]
vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xad,0xc9]
vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xad,0x3a]
vfnmadd213sd (%rdx), %xmm15, %xmm15

// CHECK: vfnmadd213sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xad,0x3a]
vfnmadd213sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xad,0x3a]
vfnmadd213sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0x0a]
vfnmadd213sd (%rdx), %xmm1, %xmm1

// CHECK: vfnmadd213sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x0a]
vfnmadd213sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x0a]
vfnmadd213sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xad,0xff]
vfnmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xad,0xff]
vfnmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xad,0xff]
vfnmadd213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xad,0xc9]
vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xad,0xc9]
vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xad,0xc9]
vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xad,0xff]
vfnmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xad,0xff]
vfnmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0xad,0xff]
vfnmadd213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xad,0xc9]
vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xad,0xc9]
vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xad,0xc9]
vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0xad,0xff]
vfnmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0xad,0xff]
vfnmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0xad,0xff]
vfnmadd213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xad,0xc9]
vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xad,0xc9]
vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xad,0xc9]
vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0xad,0xff]
vfnmadd213sd %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0xad,0xff]
vfnmadd213sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0xad,0xff]
vfnmadd213sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xad,0xc9]
vfnmadd213sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0xc9]
vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0xc9]
vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0xbc,0x82,0x00,0x01,0x00,0x00]
vfnmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0xbc,0x82,0x00,0xff,0xff,0xff]
vfnmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xad,0x7c,0x82,0x40]
vfnmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xad,0x7c,0x82,0xc0]
vfnmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xad,0x7c,0x82,0x40]
vfnmadd213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xad,0x7c,0x82,0xc0]
vfnmadd213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x8c,0x82,0x00,0x01,0x00,0x00]
vfnmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x8c,0x82,0x00,0xff,0xff,0xff]
vfnmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x4c,0x82,0x40]
vfnmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x4c,0x82,0xc0]
vfnmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x4c,0x82,0x40]
vfnmadd213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x4c,0x82,0xc0]
vfnmadd213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0xbc,0x02,0x00,0x01,0x00,0x00]
vfnmadd213ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xad,0x7c,0x02,0x40]
vfnmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xad,0x7c,0x02,0x40]
vfnmadd213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x8c,0x02,0x00,0x01,0x00,0x00]
vfnmadd213ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x4c,0x02,0x40]
vfnmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x4c,0x02,0x40]
vfnmadd213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0xba,0x00,0x01,0x00,0x00]
vfnmadd213ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfnmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xad,0x7a,0x40]
vfnmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xad,0x7a,0x40]
vfnmadd213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x8a,0x00,0x01,0x00,0x00]
vfnmadd213ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfnmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x4a,0x40]
vfnmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x4a,0x40]
vfnmadd213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm15, %xmm15

// CHECK: vfnmadd213ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xad,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm1, %xmm1

// CHECK: vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xad,0xff]
vfnmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xad,0xff]
vfnmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xad,0xff]
vfnmadd213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xad,0xc9]
vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xad,0xc9]
vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xad,0xc9]
vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xad,0x3a]
vfnmadd213ss (%rdx), %xmm15, %xmm15

// CHECK: vfnmadd213ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xad,0x3a]
vfnmadd213ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xad,0x3a]
vfnmadd213ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0x0a]
vfnmadd213ss (%rdx), %xmm1, %xmm1

// CHECK: vfnmadd213ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x0a]
vfnmadd213ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x0a]
vfnmadd213ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xad,0xff]
vfnmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xad,0xff]
vfnmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xad,0xff]
vfnmadd213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xad,0xc9]
vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xad,0xc9]
vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xad,0xc9]
vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xad,0xff]
vfnmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xad,0xff]
vfnmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xad,0xff]
vfnmadd213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xad,0xc9]
vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xad,0xc9]
vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xad,0xc9]
vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xad,0xff]
vfnmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xad,0xff]
vfnmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xad,0xff]
vfnmadd213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xad,0xc9]
vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xad,0xc9]
vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xad,0xc9]
vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xad,0xff]
vfnmadd213ss %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd213ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xad,0xff]
vfnmadd213ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd213ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xad,0xff]
vfnmadd213ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xad,0xc9]
vfnmadd213ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0xc9]
vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0xc9]
vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm15, %xmm15

// CHECK: vfnmadd231sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm1, %xmm1

// CHECK: vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0xbc,0x82,0x00,0x02,0x00,0x00]
vfnmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfnmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbd,0x7c,0x82,0x40]
vfnmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbd,0x7c,0x82,0xc0]
vfnmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbd,0x7c,0x82,0x40]
vfnmadd231sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbd,0x7c,0x82,0xc0]
vfnmadd231sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x8c,0x82,0x00,0x02,0x00,0x00]
vfnmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfnmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x4c,0x82,0x40]
vfnmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x4c,0x82,0xc0]
vfnmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x4c,0x82,0x40]
vfnmadd231sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x4c,0x82,0xc0]
vfnmadd231sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0xbc,0x02,0x00,0x02,0x00,0x00]
vfnmadd231sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbd,0x7c,0x02,0x40]
vfnmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbd,0x7c,0x02,0x40]
vfnmadd231sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x8c,0x02,0x00,0x02,0x00,0x00]
vfnmadd231sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x4c,0x02,0x40]
vfnmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x4c,0x02,0x40]
vfnmadd231sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0xba,0x00,0x02,0x00,0x00]
vfnmadd231sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfnmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbd,0x7a,0x40]
vfnmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbd,0x7a,0x40]
vfnmadd231sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x8a,0x00,0x02,0x00,0x00]
vfnmadd231sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfnmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x4a,0x40]
vfnmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x4a,0x40]
vfnmadd231sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xbd,0xff]
vfnmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xbd,0xff]
vfnmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xbd,0xff]
vfnmadd231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xbd,0xc9]
vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xbd,0xc9]
vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xbd,0xc9]
vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xbd,0x3a]
vfnmadd231sd (%rdx), %xmm15, %xmm15

// CHECK: vfnmadd231sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xbd,0x3a]
vfnmadd231sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xbd,0x3a]
vfnmadd231sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0x0a]
vfnmadd231sd (%rdx), %xmm1, %xmm1

// CHECK: vfnmadd231sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x0a]
vfnmadd231sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x0a]
vfnmadd231sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xbd,0xff]
vfnmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xbd,0xff]
vfnmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xbd,0xff]
vfnmadd231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xbd,0xc9]
vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xbd,0xc9]
vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xbd,0xc9]
vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xbd,0xff]
vfnmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xbd,0xff]
vfnmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0xbd,0xff]
vfnmadd231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xbd,0xc9]
vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xbd,0xc9]
vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xbd,0xc9]
vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0xbd,0xff]
vfnmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0xbd,0xff]
vfnmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0xbd,0xff]
vfnmadd231sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xbd,0xc9]
vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xbd,0xc9]
vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xbd,0xc9]
vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0xbd,0xff]
vfnmadd231sd %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0xbd,0xff]
vfnmadd231sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0xbd,0xff]
vfnmadd231sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xbd,0xc9]
vfnmadd231sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0xc9]
vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0xc9]
vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0xbc,0x82,0x00,0x01,0x00,0x00]
vfnmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0xbc,0x82,0x00,0xff,0xff,0xff]
vfnmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbd,0x7c,0x82,0x40]
vfnmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbd,0x7c,0x82,0xc0]
vfnmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbd,0x7c,0x82,0x40]
vfnmadd231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbd,0x7c,0x82,0xc0]
vfnmadd231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x8c,0x82,0x00,0x01,0x00,0x00]
vfnmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x8c,0x82,0x00,0xff,0xff,0xff]
vfnmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x4c,0x82,0x40]
vfnmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x4c,0x82,0xc0]
vfnmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x4c,0x82,0x40]
vfnmadd231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x4c,0x82,0xc0]
vfnmadd231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0xbc,0x02,0x00,0x01,0x00,0x00]
vfnmadd231ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbd,0x7c,0x02,0x40]
vfnmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbd,0x7c,0x02,0x40]
vfnmadd231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x8c,0x02,0x00,0x01,0x00,0x00]
vfnmadd231ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x4c,0x02,0x40]
vfnmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x4c,0x02,0x40]
vfnmadd231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0xba,0x00,0x01,0x00,0x00]
vfnmadd231ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfnmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbd,0x7a,0x40]
vfnmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbd,0x7a,0x40]
vfnmadd231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x8a,0x00,0x01,0x00,0x00]
vfnmadd231ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfnmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x4a,0x40]
vfnmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x4a,0x40]
vfnmadd231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm15, %xmm15

// CHECK: vfnmadd231ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbd,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm1, %xmm1

// CHECK: vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xbd,0xff]
vfnmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xbd,0xff]
vfnmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xbd,0xff]
vfnmadd231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xbd,0xc9]
vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xbd,0xc9]
vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xbd,0xc9]
vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbd,0x3a]
vfnmadd231ss (%rdx), %xmm15, %xmm15

// CHECK: vfnmadd231ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbd,0x3a]
vfnmadd231ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbd,0x3a]
vfnmadd231ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0x0a]
vfnmadd231ss (%rdx), %xmm1, %xmm1

// CHECK: vfnmadd231ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x0a]
vfnmadd231ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x0a]
vfnmadd231ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xbd,0xff]
vfnmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xbd,0xff]
vfnmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xbd,0xff]
vfnmadd231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xbd,0xc9]
vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xbd,0xc9]
vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xbd,0xc9]
vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xbd,0xff]
vfnmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xbd,0xff]
vfnmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xbd,0xff]
vfnmadd231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xbd,0xc9]
vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xbd,0xc9]
vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xbd,0xc9]
vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xbd,0xff]
vfnmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xbd,0xff]
vfnmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xbd,0xff]
vfnmadd231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xbd,0xc9]
vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xbd,0xc9]
vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xbd,0xc9]
vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xbd,0xff]
vfnmadd231ss %xmm15, %xmm15, %xmm15

// CHECK: vfnmadd231ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xbd,0xff]
vfnmadd231ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmadd231ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xbd,0xff]
vfnmadd231ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmadd231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbd,0xc9]
vfnmadd231ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0xc9]
vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0xc9]
vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm15, %xmm15

// CHECK: vfnmsub132sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm1, %xmm1

// CHECK: vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0xbc,0x82,0x00,0x02,0x00,0x00]
vfnmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfnmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9f,0x7c,0x82,0x40]
vfnmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9f,0x7c,0x82,0xc0]
vfnmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9f,0x7c,0x82,0x40]
vfnmsub132sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9f,0x7c,0x82,0xc0]
vfnmsub132sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x8c,0x82,0x00,0x02,0x00,0x00]
vfnmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfnmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x4c,0x82,0x40]
vfnmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x4c,0x82,0xc0]
vfnmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x4c,0x82,0x40]
vfnmsub132sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x4c,0x82,0xc0]
vfnmsub132sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0xbc,0x02,0x00,0x02,0x00,0x00]
vfnmsub132sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9f,0x7c,0x02,0x40]
vfnmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9f,0x7c,0x02,0x40]
vfnmsub132sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x8c,0x02,0x00,0x02,0x00,0x00]
vfnmsub132sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x4c,0x02,0x40]
vfnmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x4c,0x02,0x40]
vfnmsub132sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0xba,0x00,0x02,0x00,0x00]
vfnmsub132sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfnmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9f,0x7a,0x40]
vfnmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9f,0x7a,0x40]
vfnmsub132sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x8a,0x00,0x02,0x00,0x00]
vfnmsub132sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfnmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x4a,0x40]
vfnmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x4a,0x40]
vfnmsub132sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0x9f,0xff]
vfnmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0x9f,0xff]
vfnmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0x9f,0xff]
vfnmsub132sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x9f,0xc9]
vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x9f,0xc9]
vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x9f,0xc9]
vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0x9f,0x3a]
vfnmsub132sd (%rdx), %xmm15, %xmm15

// CHECK: vfnmsub132sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x9f,0x3a]
vfnmsub132sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x9f,0x3a]
vfnmsub132sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0x0a]
vfnmsub132sd (%rdx), %xmm1, %xmm1

// CHECK: vfnmsub132sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x0a]
vfnmsub132sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x0a]
vfnmsub132sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0x9f,0xff]
vfnmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0x9f,0xff]
vfnmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0x9f,0xff]
vfnmsub132sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x9f,0xc9]
vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x9f,0xc9]
vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x9f,0xc9]
vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0x9f,0xff]
vfnmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0x9f,0xff]
vfnmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0x9f,0xff]
vfnmsub132sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x9f,0xc9]
vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x9f,0xc9]
vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x9f,0xc9]
vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0x9f,0xff]
vfnmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0x9f,0xff]
vfnmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0x9f,0xff]
vfnmsub132sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x9f,0xc9]
vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x9f,0xc9]
vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x9f,0xc9]
vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0x9f,0xff]
vfnmsub132sd %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x9f,0xff]
vfnmsub132sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x9f,0xff]
vfnmsub132sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0x9f,0xc9]
vfnmsub132sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0xc9]
vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0xc9]
vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0xbc,0x82,0x00,0x01,0x00,0x00]
vfnmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0xbc,0x82,0x00,0xff,0xff,0xff]
vfnmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9f,0x7c,0x82,0x40]
vfnmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9f,0x7c,0x82,0xc0]
vfnmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9f,0x7c,0x82,0x40]
vfnmsub132ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9f,0x7c,0x82,0xc0]
vfnmsub132ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x8c,0x82,0x00,0x01,0x00,0x00]
vfnmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x8c,0x82,0x00,0xff,0xff,0xff]
vfnmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x4c,0x82,0x40]
vfnmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x4c,0x82,0xc0]
vfnmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x4c,0x82,0x40]
vfnmsub132ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x4c,0x82,0xc0]
vfnmsub132ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0xbc,0x02,0x00,0x01,0x00,0x00]
vfnmsub132ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9f,0x7c,0x02,0x40]
vfnmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9f,0x7c,0x02,0x40]
vfnmsub132ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x8c,0x02,0x00,0x01,0x00,0x00]
vfnmsub132ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x4c,0x02,0x40]
vfnmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x4c,0x02,0x40]
vfnmsub132ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0xba,0x00,0x01,0x00,0x00]
vfnmsub132ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfnmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9f,0x7a,0x40]
vfnmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9f,0x7a,0x40]
vfnmsub132ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x8a,0x00,0x01,0x00,0x00]
vfnmsub132ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfnmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x4a,0x40]
vfnmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x4a,0x40]
vfnmsub132ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm15, %xmm15

// CHECK: vfnmsub132ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm1, %xmm1

// CHECK: vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0x9f,0xff]
vfnmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0x9f,0xff]
vfnmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0x9f,0xff]
vfnmsub132ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x9f,0xc9]
vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x9f,0xc9]
vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x9f,0xc9]
vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0x9f,0x3a]
vfnmsub132ss (%rdx), %xmm15, %xmm15

// CHECK: vfnmsub132ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x9f,0x3a]
vfnmsub132ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x9f,0x3a]
vfnmsub132ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0x0a]
vfnmsub132ss (%rdx), %xmm1, %xmm1

// CHECK: vfnmsub132ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x0a]
vfnmsub132ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x0a]
vfnmsub132ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0x9f,0xff]
vfnmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0x9f,0xff]
vfnmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0x9f,0xff]
vfnmsub132ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x9f,0xc9]
vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x9f,0xc9]
vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x9f,0xc9]
vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0x9f,0xff]
vfnmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0x9f,0xff]
vfnmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0x9f,0xff]
vfnmsub132ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x9f,0xc9]
vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x9f,0xc9]
vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x9f,0xc9]
vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0x9f,0xff]
vfnmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0x9f,0xff]
vfnmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0x9f,0xff]
vfnmsub132ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x9f,0xc9]
vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x9f,0xc9]
vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x9f,0xc9]
vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0x9f,0xff]
vfnmsub132ss %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub132ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x9f,0xff]
vfnmsub132ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub132ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x9f,0xff]
vfnmsub132ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x9f,0xc9]
vfnmsub132ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0xc9]
vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0xc9]
vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm15, %xmm15

// CHECK: vfnmsub213sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm1, %xmm1

// CHECK: vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0xbc,0x82,0x00,0x02,0x00,0x00]
vfnmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0xbc,0x82,0x00,0xfe,0xff,0xff]
vfnmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xaf,0x7c,0x82,0x40]
vfnmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xaf,0x7c,0x82,0xc0]
vfnmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xaf,0x7c,0x82,0x40]
vfnmsub213sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xaf,0x7c,0x82,0xc0]
vfnmsub213sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x8c,0x82,0x00,0x02,0x00,0x00]
vfnmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x8c,0x82,0x00,0xfe,0xff,0xff]
vfnmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x4c,0x82,0x40]
vfnmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x4c,0x82,0xc0]
vfnmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x4c,0x82,0x40]
vfnmsub213sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x4c,0x82,0xc0]
vfnmsub213sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0xbc,0x02,0x00,0x02,0x00,0x00]
vfnmsub213sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xaf,0x7c,0x02,0x40]
vfnmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xaf,0x7c,0x02,0x40]
vfnmsub213sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x8c,0x02,0x00,0x02,0x00,0x00]
vfnmsub213sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x4c,0x02,0x40]
vfnmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x4c,0x02,0x40]
vfnmsub213sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0xba,0x00,0x02,0x00,0x00]
vfnmsub213sd 512(%rdx), %xmm15, %xmm15

// CHECK: vfnmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xaf,0x7a,0x40]
vfnmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xaf,0x7a,0x40]
vfnmsub213sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x8a,0x00,0x02,0x00,0x00]
vfnmsub213sd 512(%rdx), %xmm1, %xmm1

// CHECK: vfnmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x4a,0x40]
vfnmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x4a,0x40]
vfnmsub213sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xaf,0xff]
vfnmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xaf,0xff]
vfnmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xaf,0xff]
vfnmsub213sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xaf,0xc9]
vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xaf,0xc9]
vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xaf,0xc9]
vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x81,0xaf,0x3a]
vfnmsub213sd (%rdx), %xmm15, %xmm15

// CHECK: vfnmsub213sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0xaf,0x3a]
vfnmsub213sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0xaf,0x3a]
vfnmsub213sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0x0a]
vfnmsub213sd (%rdx), %xmm1, %xmm1

// CHECK: vfnmsub213sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x0a]
vfnmsub213sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x0a]
vfnmsub213sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xaf,0xff]
vfnmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xaf,0xff]
vfnmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xaf,0xff]
vfnmsub213sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xaf,0xc9]
vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xaf,0xc9]
vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xaf,0xc9]
vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xaf,0xff]
vfnmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xaf,0xff]
vfnmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0xaf,0xff]
vfnmsub213sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xaf,0xc9]
vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xaf,0xc9]
vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xaf,0xc9]
vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0xaf,0xff]
vfnmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0xaf,0xff]
vfnmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0xaf,0xff]
vfnmsub213sd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xaf,0xc9]
vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xaf,0xc9]
vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xaf,0xc9]
vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x81,0xaf,0xff]
vfnmsub213sd %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0xaf,0xff]
vfnmsub213sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0xaf,0xff]
vfnmsub213sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0xf1,0xaf,0xc9]
vfnmsub213sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0xc9]
vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0xc9]
vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0xbc,0x82,0x00,0x01,0x00,0x00]
vfnmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0xbc,0x82,0x00,0xff,0xff,0xff]
vfnmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xaf,0x7c,0x82,0x40]
vfnmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xaf,0x7c,0x82,0xc0]
vfnmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xaf,0x7c,0x82,0x40]
vfnmsub213ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xaf,0x7c,0x82,0xc0]
vfnmsub213ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x8c,0x82,0x00,0x01,0x00,0x00]
vfnmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x8c,0x82,0x00,0xff,0xff,0xff]
vfnmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x4c,0x82,0x40]
vfnmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x4c,0x82,0xc0]
vfnmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x4c,0x82,0x40]
vfnmsub213ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x4c,0x82,0xc0]
vfnmsub213ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0xbc,0x02,0x00,0x01,0x00,0x00]
vfnmsub213ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xaf,0x7c,0x02,0x40]
vfnmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xaf,0x7c,0x02,0x40]
vfnmsub213ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x8c,0x02,0x00,0x01,0x00,0x00]
vfnmsub213ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x4c,0x02,0x40]
vfnmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x4c,0x02,0x40]
vfnmsub213ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0xba,0x00,0x01,0x00,0x00]
vfnmsub213ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfnmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xaf,0x7a,0x40]
vfnmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xaf,0x7a,0x40]
vfnmsub213ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x8a,0x00,0x01,0x00,0x00]
vfnmsub213ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfnmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x4a,0x40]
vfnmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x4a,0x40]
vfnmsub213ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm15, %xmm15

// CHECK: vfnmsub213ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xaf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm1, %xmm1

// CHECK: vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xaf,0xff]
vfnmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xaf,0xff]
vfnmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xaf,0xff]
vfnmsub213ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xaf,0xc9]
vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xaf,0xc9]
vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xaf,0xc9]
vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xaf,0x3a]
vfnmsub213ss (%rdx), %xmm15, %xmm15

// CHECK: vfnmsub213ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xaf,0x3a]
vfnmsub213ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xaf,0x3a]
vfnmsub213ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0x0a]
vfnmsub213ss (%rdx), %xmm1, %xmm1

// CHECK: vfnmsub213ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x0a]
vfnmsub213ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x0a]
vfnmsub213ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xaf,0xff]
vfnmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xaf,0xff]
vfnmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xaf,0xff]
vfnmsub213ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xaf,0xc9]
vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xaf,0xc9]
vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xaf,0xc9]
vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xaf,0xff]
vfnmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xaf,0xff]
vfnmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xaf,0xff]
vfnmsub213ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xaf,0xc9]
vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xaf,0xc9]
vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xaf,0xc9]
vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xaf,0xff]
vfnmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xaf,0xff]
vfnmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xaf,0xff]
vfnmsub213ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xaf,0xc9]
vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xaf,0xc9]
vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xaf,0xc9]
vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xaf,0xff]
vfnmsub213ss %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub213ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xaf,0xff]
vfnmsub213ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub213ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xaf,0xff]
vfnmsub213ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xaf,0xc9]
vfnmsub213ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0xc9]
vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0xc9]
vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0xbf,0xff]
vfnmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0xbf,0xff]
vfnmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0xbf,0xff]
vfnmsub231sd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xbf,0xc9]
vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xbf,0xc9]
vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xbf,0xc9]
vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0xbf,0xff]
vfnmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0xbf,0xff]
vfnmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0xbf,0xff]
vfnmsub231sd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xbf,0xc9]
vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xbf,0xc9]
vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xbf,0xc9]
vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0xbf,0xff]
vfnmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0xbf,0xff]
vfnmsub231sd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xbf,0xc9]
vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xbf,0xc9]
vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0xbc,0x82,0x00,0x01,0x00,0x00]
vfnmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0xbc,0x82,0x00,0xff,0xff,0xff]
vfnmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vfnmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbf,0x7c,0x82,0x40]
vfnmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbf,0x7c,0x82,0xc0]
vfnmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbf,0x7c,0x82,0x40]
vfnmsub231ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbf,0x7c,0x82,0xc0]
vfnmsub231ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x8c,0x82,0x00,0x01,0x00,0x00]
vfnmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x8c,0x82,0x00,0xff,0xff,0xff]
vfnmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vfnmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x4c,0x82,0x40]
vfnmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x4c,0x82,0xc0]
vfnmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x4c,0x82,0x40]
vfnmsub231ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x4c,0x82,0xc0]
vfnmsub231ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0xbc,0x02,0x00,0x01,0x00,0x00]
vfnmsub231ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vfnmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbf,0x7c,0x02,0x40]
vfnmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbf,0x7c,0x02,0x40]
vfnmsub231ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x8c,0x02,0x00,0x01,0x00,0x00]
vfnmsub231ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vfnmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x4c,0x02,0x40]
vfnmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x4c,0x02,0x40]
vfnmsub231ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0xba,0x00,0x01,0x00,0x00]
vfnmsub231ss 256(%rdx), %xmm15, %xmm15

// CHECK: vfnmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbf,0x7a,0x40]
vfnmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbf,0x7a,0x40]
vfnmsub231ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x8a,0x00,0x01,0x00,0x00]
vfnmsub231ss 256(%rdx), %xmm1, %xmm1

// CHECK: vfnmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x4a,0x40]
vfnmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x4a,0x40]
vfnmsub231ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm15, %xmm15

// CHECK: vfnmsub231ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbf,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm1, %xmm1

// CHECK: vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0xbf,0xff]
vfnmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0xbf,0xff]
vfnmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0xbf,0xff]
vfnmsub231ss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xbf,0xc9]
vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xbf,0xc9]
vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xbf,0xc9]
vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x62,0x01,0xbf,0x3a]
vfnmsub231ss (%rdx), %xmm15, %xmm15

// CHECK: vfnmsub231ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0xbf,0x3a]
vfnmsub231ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0xbf,0x3a]
vfnmsub231ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0x0a]
vfnmsub231ss (%rdx), %xmm1, %xmm1

// CHECK: vfnmsub231ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x0a]
vfnmsub231ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x0a]
vfnmsub231ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0xbf,0xff]
vfnmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0xbf,0xff]
vfnmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0xbf,0xff]
vfnmsub231ss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xbf,0xc9]
vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xbf,0xc9]
vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xbf,0xc9]
vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0xbf,0xff]
vfnmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0xbf,0xff]
vfnmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0xbf,0xff]
vfnmsub231ss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xbf,0xc9]
vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xbf,0xc9]
vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xbf,0xc9]
vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0xbf,0xff]
vfnmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0xbf,0xff]
vfnmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0xbf,0xff]
vfnmsub231ss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xbf,0xc9]
vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xbf,0xc9]
vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xbf,0xc9]
vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x42,0x01,0xbf,0xff]
vfnmsub231ss %xmm15, %xmm15, %xmm15

// CHECK: vfnmsub231ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0xbf,0xff]
vfnmsub231ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vfnmsub231ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0xbf,0xff]
vfnmsub231ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vfnmsub231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0xbf,0xc9]
vfnmsub231ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0xc9]
vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0xc9]
vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x43,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm15, %xmm15

// CHECK: vgetexpsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x43,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x43,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm1, %xmm1

// CHECK: vgetexpsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x43,0x7c,0x82,0x40]
vgetexpsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetexpsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x43,0x7c,0x82,0xc0]
vgetexpsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetexpsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x43,0x7c,0x82,0x40]
vgetexpsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x43,0x7c,0x82,0xc0]
vgetexpsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x43,0x7c,0x82,0x40]
vgetexpsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x43,0x7c,0x82,0xc0]
vgetexpsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x4c,0x82,0x40]
vgetexpsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetexpsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x4c,0x82,0xc0]
vgetexpsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetexpsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x4c,0x82,0x40]
vgetexpsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x4c,0x82,0xc0]
vgetexpsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x4c,0x82,0x40]
vgetexpsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x4c,0x82,0xc0]
vgetexpsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x43,0x7c,0x02,0x40]
vgetexpsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vgetexpsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x43,0x7c,0x02,0x40]
vgetexpsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x43,0x7c,0x02,0x40]
vgetexpsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x4c,0x02,0x40]
vgetexpsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vgetexpsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x4c,0x02,0x40]
vgetexpsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x4c,0x02,0x40]
vgetexpsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x43,0x7a,0x40]
vgetexpsd 512(%rdx), %xmm15, %xmm15

// CHECK: vgetexpsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x43,0x7a,0x40]
vgetexpsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x43,0x7a,0x40]
vgetexpsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x4a,0x40]
vgetexpsd 512(%rdx), %xmm1, %xmm1

// CHECK: vgetexpsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x4a,0x40]
vgetexpsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x4a,0x40]
vgetexpsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x43,0x3a]
vgetexpsd (%rdx), %xmm15, %xmm15

// CHECK: vgetexpsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x43,0x3a]
vgetexpsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x43,0x3a]
vgetexpsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x0a]
vgetexpsd (%rdx), %xmm1, %xmm1

// CHECK: vgetexpsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x0a]
vgetexpsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x0a]
vgetexpsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0x43,0xff]
vgetexpsd {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vgetexpsd {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0x43,0xff]
vgetexpsd {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0x43,0xff]
vgetexpsd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x43,0xc9]
vgetexpsd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x43,0xc9]
vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x43,0xc9]
vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x08,0x43,0xff]
vgetexpsd %xmm15, %xmm15, %xmm15

// CHECK: vgetexpsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x43,0xff]
vgetexpsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetexpsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x43,0xff]
vgetexpsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0xc9]
vgetexpsd %xmm1, %xmm1, %xmm1

// CHECK: vgetexpsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0xc9]
vgetexpsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0xc9]
vgetexpsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x43,0x7c,0x82,0x40]
vgetexpss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetexpss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x43,0x7c,0x82,0xc0]
vgetexpss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetexpss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x43,0x7c,0x82,0x40]
vgetexpss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x43,0x7c,0x82,0xc0]
vgetexpss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x43,0x7c,0x82,0x40]
vgetexpss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x43,0x7c,0x82,0xc0]
vgetexpss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x4c,0x82,0x40]
vgetexpss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetexpss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x4c,0x82,0xc0]
vgetexpss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetexpss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x4c,0x82,0x40]
vgetexpss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x4c,0x82,0xc0]
vgetexpss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x4c,0x82,0x40]
vgetexpss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x4c,0x82,0xc0]
vgetexpss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x43,0x7c,0x02,0x40]
vgetexpss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vgetexpss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x43,0x7c,0x02,0x40]
vgetexpss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x43,0x7c,0x02,0x40]
vgetexpss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x4c,0x02,0x40]
vgetexpss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vgetexpss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x4c,0x02,0x40]
vgetexpss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x4c,0x02,0x40]
vgetexpss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x43,0x7a,0x40]
vgetexpss 256(%rdx), %xmm15, %xmm15

// CHECK: vgetexpss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x43,0x7a,0x40]
vgetexpss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x43,0x7a,0x40]
vgetexpss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x4a,0x40]
vgetexpss 256(%rdx), %xmm1, %xmm1

// CHECK: vgetexpss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x4a,0x40]
vgetexpss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x4a,0x40]
vgetexpss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x43,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm15, %xmm15

// CHECK: vgetexpss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x43,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x43,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm1, %xmm1

// CHECK: vgetexpss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x43,0x3a]
vgetexpss (%rdx), %xmm15, %xmm15

// CHECK: vgetexpss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x43,0x3a]
vgetexpss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x43,0x3a]
vgetexpss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x0a]
vgetexpss (%rdx), %xmm1, %xmm1

// CHECK: vgetexpss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x0a]
vgetexpss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x0a]
vgetexpss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0x43,0xff]
vgetexpss {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vgetexpss {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0x43,0xff]
vgetexpss {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0x43,0xff]
vgetexpss {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x43,0xc9]
vgetexpss {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x43,0xc9]
vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x43,0xc9]
vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x08,0x43,0xff]
vgetexpss %xmm15, %xmm15, %xmm15

// CHECK: vgetexpss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x43,0xff]
vgetexpss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetexpss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x43,0xff]
vgetexpss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetexpss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0xc9]
vgetexpss %xmm1, %xmm1, %xmm1

// CHECK: vgetexpss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0xc9]
vgetexpss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0xc9]
vgetexpss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x27,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm15, %xmm15

// CHECK: vgetmantsd $0, 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x27,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x27,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm1, %xmm1

// CHECK: vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x27,0x7c,0x82,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetmantsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x27,0x7c,0x82,0xc0,0x00]
vgetmantsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetmantsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x27,0x7c,0x82,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x27,0x7c,0x82,0xc0,0x00]
vgetmantsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x27,0x7c,0x82,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x27,0x7c,0x82,0xc0,0x00]
vgetmantsd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x4c,0x82,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetmantsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x4c,0x82,0xc0,0x00]
vgetmantsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetmantsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x4c,0x82,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x4c,0x82,0xc0,0x00]
vgetmantsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x4c,0x82,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x4c,0x82,0xc0,0x00]
vgetmantsd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x27,0x7c,0x02,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vgetmantsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x27,0x7c,0x02,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x27,0x7c,0x02,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x4c,0x02,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vgetmantsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x4c,0x02,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x4c,0x02,0x40,0x00]
vgetmantsd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x27,0x7a,0x40,0x00]
vgetmantsd $0, 512(%rdx), %xmm15, %xmm15

// CHECK: vgetmantsd $0, 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x27,0x7a,0x40,0x00]
vgetmantsd $0, 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x27,0x7a,0x40,0x00]
vgetmantsd $0, 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x4a,0x40,0x00]
vgetmantsd $0, 512(%rdx), %xmm1, %xmm1

// CHECK: vgetmantsd $0, 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x4a,0x40,0x00]
vgetmantsd $0, 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x4a,0x40,0x00]
vgetmantsd $0, 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x27,0x3a,0x00]
vgetmantsd $0, (%rdx), %xmm15, %xmm15

// CHECK: vgetmantsd $0, (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x27,0x3a,0x00]
vgetmantsd $0, (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x27,0x3a,0x00]
vgetmantsd $0, (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x0a,0x00]
vgetmantsd $0, (%rdx), %xmm1, %xmm1

// CHECK: vgetmantsd $0, (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x0a,0x00]
vgetmantsd $0, (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x0a,0x00]
vgetmantsd $0, (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x85,0x18,0x27,0xff,0x00]
vgetmantsd $0, {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vgetmantsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x85,0x1a,0x27,0xff,0x00]
vgetmantsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x85,0x9a,0x27,0xff,0x00]
vgetmantsd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x18,0x27,0xc9,0x00]
vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x1a,0x27,0xc9,0x00]
vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x9a,0x27,0xc9,0x00]
vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x85,0x08,0x27,0xff,0x00]
vgetmantsd $0, %xmm15, %xmm15, %xmm15

// CHECK: vgetmantsd $0, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x85,0x0a,0x27,0xff,0x00]
vgetmantsd $0, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetmantsd $0, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x85,0x8a,0x27,0xff,0x00]
vgetmantsd $0, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantsd $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0xc9,0x00]
vgetmantsd $0, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0xc9,0x00]
vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0xc9,0x00]
vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x27,0x7c,0x82,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetmantss $0, -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x27,0x7c,0x82,0xc0,0x00]
vgetmantss $0, -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vgetmantss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x27,0x7c,0x82,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x27,0x7c,0x82,0xc0,0x00]
vgetmantss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x27,0x7c,0x82,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x27,0x7c,0x82,0xc0,0x00]
vgetmantss $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x4c,0x82,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetmantss $0, -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x4c,0x82,0xc0,0x00]
vgetmantss $0, -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vgetmantss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x4c,0x82,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x4c,0x82,0xc0,0x00]
vgetmantss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x4c,0x82,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x4c,0x82,0xc0,0x00]
vgetmantss $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x27,0x7c,0x02,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vgetmantss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x27,0x7c,0x02,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x27,0x7c,0x02,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x4c,0x02,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vgetmantss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x4c,0x02,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x4c,0x02,0x40,0x00]
vgetmantss $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x27,0x7a,0x40,0x00]
vgetmantss $0, 256(%rdx), %xmm15, %xmm15

// CHECK: vgetmantss $0, 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x27,0x7a,0x40,0x00]
vgetmantss $0, 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x27,0x7a,0x40,0x00]
vgetmantss $0, 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x4a,0x40,0x00]
vgetmantss $0, 256(%rdx), %xmm1, %xmm1

// CHECK: vgetmantss $0, 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x4a,0x40,0x00]
vgetmantss $0, 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x4a,0x40,0x00]
vgetmantss $0, 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x27,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm15, %xmm15

// CHECK: vgetmantss $0, 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x27,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x27,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm1, %xmm1

// CHECK: vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x27,0x3a,0x00]
vgetmantss $0, (%rdx), %xmm15, %xmm15

// CHECK: vgetmantss $0, (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x27,0x3a,0x00]
vgetmantss $0, (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x27,0x3a,0x00]
vgetmantss $0, (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x0a,0x00]
vgetmantss $0, (%rdx), %xmm1, %xmm1

// CHECK: vgetmantss $0, (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x0a,0x00]
vgetmantss $0, (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x0a,0x00]
vgetmantss $0, (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x05,0x18,0x27,0xff,0x00]
vgetmantss $0, {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vgetmantss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x05,0x1a,0x27,0xff,0x00]
vgetmantss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x05,0x9a,0x27,0xff,0x00]
vgetmantss $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x18,0x27,0xc9,0x00]
vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x1a,0x27,0xc9,0x00]
vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x9a,0x27,0xc9,0x00]
vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x05,0x08,0x27,0xff,0x00]
vgetmantss $0, %xmm15, %xmm15, %xmm15

// CHECK: vgetmantss $0, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x05,0x0a,0x27,0xff,0x00]
vgetmantss $0, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vgetmantss $0, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x05,0x8a,0x27,0xff,0x00]
vgetmantss $0, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vgetmantss $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0xc9,0x00]
vgetmantss $0, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0xc9,0x00]
vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0xc9,0x00]
vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm15, %xmm15

// CHECK: vmaxsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm1, %xmm1

// CHECK: vmaxsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5f,0xbc,0x82,0x00,0x02,0x00,0x00]
vmaxsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmaxsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5f,0xbc,0x82,0x00,0xfe,0xff,0xff]
vmaxsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmaxsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5f,0x7c,0x82,0x40]
vmaxsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5f,0x7c,0x82,0xc0]
vmaxsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5f,0x7c,0x82,0x40]
vmaxsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5f,0x7c,0x82,0xc0]
vmaxsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5f,0x8c,0x82,0x00,0x02,0x00,0x00]
vmaxsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmaxsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5f,0x8c,0x82,0x00,0xfe,0xff,0xff]
vmaxsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmaxsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x4c,0x82,0x40]
vmaxsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x4c,0x82,0xc0]
vmaxsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x4c,0x82,0x40]
vmaxsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x4c,0x82,0xc0]
vmaxsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5f,0xbc,0x02,0x00,0x02,0x00,0x00]
vmaxsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vmaxsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5f,0x7c,0x02,0x40]
vmaxsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5f,0x7c,0x02,0x40]
vmaxsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5f,0x8c,0x02,0x00,0x02,0x00,0x00]
vmaxsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vmaxsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x4c,0x02,0x40]
vmaxsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x4c,0x02,0x40]
vmaxsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5f,0xba,0x00,0x02,0x00,0x00]
vmaxsd 512(%rdx), %xmm15, %xmm15

// CHECK: vmaxsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5f,0x7a,0x40]
vmaxsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5f,0x7a,0x40]
vmaxsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5f,0x8a,0x00,0x02,0x00,0x00]
vmaxsd 512(%rdx), %xmm1, %xmm1

// CHECK: vmaxsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x4a,0x40]
vmaxsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x4a,0x40]
vmaxsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5f,0x3a]
vmaxsd (%rdx), %xmm15, %xmm15

// CHECK: vmaxsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5f,0x3a]
vmaxsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5f,0x3a]
vmaxsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5f,0x0a]
vmaxsd (%rdx), %xmm1, %xmm1

// CHECK: vmaxsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x0a]
vmaxsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x0a]
vmaxsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x5f,0xff]
vmaxsd {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmaxsd {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x5f,0xff]
vmaxsd {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x5f,0xff]
vmaxsd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5f,0xc9]
vmaxsd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5f,0xc9]
vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5f,0xc9]
vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x5f,0xff]
vmaxsd %xmm15, %xmm15, %xmm15

// CHECK: vmaxsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x5f,0xff]
vmaxsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmaxsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x5f,0xff]
vmaxsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5f,0xc9]
vmaxsd %xmm1, %xmm1, %xmm1

// CHECK: vmaxsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0xc9]
vmaxsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0xc9]
vmaxsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5f,0xbc,0x82,0x00,0x01,0x00,0x00]
vmaxss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmaxss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5f,0xbc,0x82,0x00,0xff,0xff,0xff]
vmaxss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmaxss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5f,0x7c,0x82,0x40]
vmaxss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmaxss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5f,0x7c,0x82,0xc0]
vmaxss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmaxss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5f,0x7c,0x82,0x40]
vmaxss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5f,0x7c,0x82,0xc0]
vmaxss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5f,0x8c,0x82,0x00,0x01,0x00,0x00]
vmaxss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmaxss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5f,0x8c,0x82,0x00,0xff,0xff,0xff]
vmaxss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmaxss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x4c,0x82,0x40]
vmaxss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x4c,0x82,0xc0]
vmaxss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x4c,0x82,0x40]
vmaxss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x4c,0x82,0xc0]
vmaxss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5f,0xbc,0x02,0x00,0x01,0x00,0x00]
vmaxss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vmaxss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5f,0x7c,0x02,0x40]
vmaxss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vmaxss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5f,0x7c,0x02,0x40]
vmaxss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5f,0x8c,0x02,0x00,0x01,0x00,0x00]
vmaxss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vmaxss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x4c,0x02,0x40]
vmaxss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x4c,0x02,0x40]
vmaxss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5f,0xba,0x00,0x01,0x00,0x00]
vmaxss 256(%rdx), %xmm15, %xmm15

// CHECK: vmaxss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5f,0x7a,0x40]
vmaxss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmaxss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5f,0x7a,0x40]
vmaxss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5f,0x8a,0x00,0x01,0x00,0x00]
vmaxss 256(%rdx), %xmm1, %xmm1

// CHECK: vmaxss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x4a,0x40]
vmaxss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x4a,0x40]
vmaxss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm15, %xmm15

// CHECK: vmaxss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vmaxss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm1, %xmm1

// CHECK: vmaxss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5f,0x3a]
vmaxss (%rdx), %xmm15, %xmm15

// CHECK: vmaxss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5f,0x3a]
vmaxss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmaxss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5f,0x3a]
vmaxss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5f,0x0a]
vmaxss (%rdx), %xmm1, %xmm1

// CHECK: vmaxss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x0a]
vmaxss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x0a]
vmaxss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x5f,0xff]
vmaxss {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmaxss {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x5f,0xff]
vmaxss {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmaxss {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x5f,0xff]
vmaxss {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5f,0xc9]
vmaxss {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5f,0xc9]
vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5f,0xc9]
vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x5f,0xff]
vmaxss %xmm15, %xmm15, %xmm15

// CHECK: vmaxss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x5f,0xff]
vmaxss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmaxss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x5f,0xff]
vmaxss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmaxss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5f,0xc9]
vmaxss %xmm1, %xmm1, %xmm1

// CHECK: vmaxss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0xc9]
vmaxss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0xc9]
vmaxss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm15, %xmm15

// CHECK: vminsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vminsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm1, %xmm1

// CHECK: vminsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vminsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5d,0xbc,0x82,0x00,0x02,0x00,0x00]
vminsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vminsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5d,0xbc,0x82,0x00,0xfe,0xff,0xff]
vminsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vminsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5d,0x7c,0x82,0x40]
vminsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vminsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5d,0x7c,0x82,0xc0]
vminsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vminsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5d,0x7c,0x82,0x40]
vminsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5d,0x7c,0x82,0xc0]
vminsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5d,0x8c,0x82,0x00,0x02,0x00,0x00]
vminsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vminsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5d,0x8c,0x82,0x00,0xfe,0xff,0xff]
vminsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vminsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x4c,0x82,0x40]
vminsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x4c,0x82,0xc0]
vminsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x4c,0x82,0x40]
vminsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x4c,0x82,0xc0]
vminsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5d,0xbc,0x02,0x00,0x02,0x00,0x00]
vminsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vminsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5d,0x7c,0x02,0x40]
vminsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vminsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5d,0x7c,0x02,0x40]
vminsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5d,0x8c,0x02,0x00,0x02,0x00,0x00]
vminsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vminsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x4c,0x02,0x40]
vminsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vminsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x4c,0x02,0x40]
vminsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5d,0xba,0x00,0x02,0x00,0x00]
vminsd 512(%rdx), %xmm15, %xmm15

// CHECK: vminsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5d,0x7a,0x40]
vminsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vminsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5d,0x7a,0x40]
vminsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5d,0x8a,0x00,0x02,0x00,0x00]
vminsd 512(%rdx), %xmm1, %xmm1

// CHECK: vminsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x4a,0x40]
vminsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vminsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x4a,0x40]
vminsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5d,0x3a]
vminsd (%rdx), %xmm15, %xmm15

// CHECK: vminsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5d,0x3a]
vminsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vminsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5d,0x3a]
vminsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5d,0x0a]
vminsd (%rdx), %xmm1, %xmm1

// CHECK: vminsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x0a]
vminsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vminsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x0a]
vminsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x5d,0xff]
vminsd {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vminsd {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x5d,0xff]
vminsd {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vminsd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x5d,0xff]
vminsd {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5d,0xc9]
vminsd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5d,0xc9]
vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5d,0xc9]
vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x5d,0xff]
vminsd %xmm15, %xmm15, %xmm15

// CHECK: vminsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x5d,0xff]
vminsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vminsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x5d,0xff]
vminsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vminsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5d,0xc9]
vminsd %xmm1, %xmm1, %xmm1

// CHECK: vminsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0xc9]
vminsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0xc9]
vminsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5d,0xbc,0x82,0x00,0x01,0x00,0x00]
vminss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vminss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5d,0xbc,0x82,0x00,0xff,0xff,0xff]
vminss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vminss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5d,0x7c,0x82,0x40]
vminss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vminss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5d,0x7c,0x82,0xc0]
vminss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vminss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5d,0x7c,0x82,0x40]
vminss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5d,0x7c,0x82,0xc0]
vminss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5d,0x8c,0x82,0x00,0x01,0x00,0x00]
vminss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vminss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5d,0x8c,0x82,0x00,0xff,0xff,0xff]
vminss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vminss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x4c,0x82,0x40]
vminss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x4c,0x82,0xc0]
vminss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x4c,0x82,0x40]
vminss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x4c,0x82,0xc0]
vminss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5d,0xbc,0x02,0x00,0x01,0x00,0x00]
vminss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vminss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5d,0x7c,0x02,0x40]
vminss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vminss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5d,0x7c,0x02,0x40]
vminss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5d,0x8c,0x02,0x00,0x01,0x00,0x00]
vminss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vminss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x4c,0x02,0x40]
vminss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vminss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x4c,0x02,0x40]
vminss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5d,0xba,0x00,0x01,0x00,0x00]
vminss 256(%rdx), %xmm15, %xmm15

// CHECK: vminss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5d,0x7a,0x40]
vminss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vminss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5d,0x7a,0x40]
vminss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5d,0x8a,0x00,0x01,0x00,0x00]
vminss 256(%rdx), %xmm1, %xmm1

// CHECK: vminss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x4a,0x40]
vminss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vminss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x4a,0x40]
vminss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm15, %xmm15

// CHECK: vminss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vminss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm1, %xmm1

// CHECK: vminss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vminss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5d,0x3a]
vminss (%rdx), %xmm15, %xmm15

// CHECK: vminss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5d,0x3a]
vminss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vminss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5d,0x3a]
vminss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5d,0x0a]
vminss (%rdx), %xmm1, %xmm1

// CHECK: vminss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x0a]
vminss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vminss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x0a]
vminss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x5d,0xff]
vminss {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vminss {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x5d,0xff]
vminss {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vminss {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x5d,0xff]
vminss {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5d,0xc9]
vminss {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5d,0xc9]
vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5d,0xc9]
vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x5d,0xff]
vminss %xmm15, %xmm15, %xmm15

// CHECK: vminss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x5d,0xff]
vminss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vminss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x5d,0xff]
vminss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vminss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5d,0xc9]
vminss %xmm1, %xmm1, %xmm1

// CHECK: vminss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0xc9]
vminss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0xc9]
vminss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovsd 485498096, %xmm15
// CHECK: encoding: [0xc5,0x7b,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm15

// CHECK: vmovsd 485498096, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm15 {%k2}

// CHECK: vmovsd 485498096, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0xff,0x8a,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm15 {%k2} {z}

// CHECK: vmovsd 485498096, %xmm1
// CHECK: encoding: [0xc5,0xfb,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm1

// CHECK: vmovsd 485498096, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm1 {%k2}

// CHECK: vmovsd 485498096, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm1 {%k2} {z}

// CHECK: vmovsd 512(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x7b,0x10,0xbc,0x82,0x00,0x02,0x00,0x00]
vmovsd 512(%rdx,%rax,4), %xmm15

// CHECK: vmovsd -512(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x7b,0x10,0xbc,0x82,0x00,0xfe,0xff,0xff]
vmovsd -512(%rdx,%rax,4), %xmm15

// CHECK: vmovsd 512(%rdx,%rax,4), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x10,0x7c,0x82,0x40]
vmovsd 512(%rdx,%rax,4), %xmm15 {%k2}

// CHECK: vmovsd -512(%rdx,%rax,4), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x10,0x7c,0x82,0xc0]
vmovsd -512(%rdx,%rax,4), %xmm15 {%k2}

// CHECK: vmovsd 512(%rdx,%rax,4), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0xff,0x8a,0x10,0x7c,0x82,0x40]
vmovsd 512(%rdx,%rax,4), %xmm15 {%k2} {z}

// CHECK: vmovsd -512(%rdx,%rax,4), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0xff,0x8a,0x10,0x7c,0x82,0xc0]
vmovsd -512(%rdx,%rax,4), %xmm15 {%k2} {z}

// CHECK: vmovsd 512(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xfb,0x10,0x8c,0x82,0x00,0x02,0x00,0x00]
vmovsd 512(%rdx,%rax,4), %xmm1

// CHECK: vmovsd -512(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xfb,0x10,0x8c,0x82,0x00,0xfe,0xff,0xff]
vmovsd -512(%rdx,%rax,4), %xmm1

// CHECK: vmovsd 512(%rdx,%rax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x4c,0x82,0x40]
vmovsd 512(%rdx,%rax,4), %xmm1 {%k2}

// CHECK: vmovsd -512(%rdx,%rax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x4c,0x82,0xc0]
vmovsd -512(%rdx,%rax,4), %xmm1 {%k2}

// CHECK: vmovsd 512(%rdx,%rax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x4c,0x82,0x40]
vmovsd 512(%rdx,%rax,4), %xmm1 {%k2} {z}

// CHECK: vmovsd -512(%rdx,%rax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x4c,0x82,0xc0]
vmovsd -512(%rdx,%rax,4), %xmm1 {%k2} {z}

// CHECK: vmovsd 512(%rdx,%rax), %xmm15
// CHECK: encoding: [0xc5,0x7b,0x10,0xbc,0x02,0x00,0x02,0x00,0x00]
vmovsd 512(%rdx,%rax), %xmm15

// CHECK: vmovsd 512(%rdx,%rax), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x10,0x7c,0x02,0x40]
vmovsd 512(%rdx,%rax), %xmm15 {%k2}

// CHECK: vmovsd 512(%rdx,%rax), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0xff,0x8a,0x10,0x7c,0x02,0x40]
vmovsd 512(%rdx,%rax), %xmm15 {%k2} {z}

// CHECK: vmovsd 512(%rdx,%rax), %xmm1
// CHECK: encoding: [0xc5,0xfb,0x10,0x8c,0x02,0x00,0x02,0x00,0x00]
vmovsd 512(%rdx,%rax), %xmm1

// CHECK: vmovsd 512(%rdx,%rax), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x4c,0x02,0x40]
vmovsd 512(%rdx,%rax), %xmm1 {%k2}

// CHECK: vmovsd 512(%rdx,%rax), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x4c,0x02,0x40]
vmovsd 512(%rdx,%rax), %xmm1 {%k2} {z}

// CHECK: vmovsd 512(%rdx), %xmm15
// CHECK: encoding: [0xc5,0x7b,0x10,0xba,0x00,0x02,0x00,0x00]
vmovsd 512(%rdx), %xmm15

// CHECK: vmovsd 512(%rdx), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x10,0x7a,0x40]
vmovsd 512(%rdx), %xmm15 {%k2}

// CHECK: vmovsd 512(%rdx), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0xff,0x8a,0x10,0x7a,0x40]
vmovsd 512(%rdx), %xmm15 {%k2} {z}

// CHECK: vmovsd 512(%rdx), %xmm1
// CHECK: encoding: [0xc5,0xfb,0x10,0x8a,0x00,0x02,0x00,0x00]
vmovsd 512(%rdx), %xmm1

// CHECK: vmovsd 512(%rdx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x4a,0x40]
vmovsd 512(%rdx), %xmm1 {%k2}

// CHECK: vmovsd 512(%rdx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x4a,0x40]
vmovsd 512(%rdx), %xmm1 {%k2} {z}

// CHECK: vmovsd (%rdx), %xmm15
// CHECK: encoding: [0xc5,0x7b,0x10,0x3a]
vmovsd (%rdx), %xmm15

// CHECK: vmovsd (%rdx), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x10,0x3a]
vmovsd (%rdx), %xmm15 {%k2}

// CHECK: vmovsd (%rdx), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0xff,0x8a,0x10,0x3a]
vmovsd (%rdx), %xmm15 {%k2} {z}

// CHECK: vmovsd (%rdx), %xmm1
// CHECK: encoding: [0xc5,0xfb,0x10,0x0a]
vmovsd (%rdx), %xmm1

// CHECK: vmovsd (%rdx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x0a]
vmovsd (%rdx), %xmm1 {%k2}

// CHECK: vmovsd (%rdx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x0a]
vmovsd (%rdx), %xmm1 {%k2} {z}

// CHECK: vmovsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x11,0xff]
vmovsd.s %xmm15, %xmm15, %xmm15

// CHECK: vmovsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x11,0xff]
vmovsd.s %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmovsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x11,0xff]
vmovsd.s %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x11,0xc9]
vmovsd.s %xmm1, %xmm1, %xmm1

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x11,0xc9]
vmovsd.s %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x11,0xc9]
vmovsd.s %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovsd %xmm1, 485498096
// CHECK: encoding: [0xc5,0xfb,0x11,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd %xmm1, 485498096

// CHECK: vmovsd %xmm1, 485498096 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd %xmm1, 485498096 {%k2}

// CHECK: vmovsd %xmm1, 512(%rdx)
// CHECK: encoding: [0xc5,0xfb,0x11,0x8a,0x00,0x02,0x00,0x00]
vmovsd %xmm1, 512(%rdx)

// CHECK: vmovsd %xmm1, 512(%rdx) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x4a,0x40]
vmovsd %xmm1, 512(%rdx) {%k2}

// CHECK: vmovsd %xmm1, 512(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0xfb,0x11,0x8c,0x82,0x00,0x02,0x00,0x00]
vmovsd %xmm1, 512(%rdx,%rax,4)

// CHECK: vmovsd %xmm1, -512(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0xfb,0x11,0x8c,0x82,0x00,0xfe,0xff,0xff]
vmovsd %xmm1, -512(%rdx,%rax,4)

// CHECK: vmovsd %xmm1, 512(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x4c,0x82,0x40]
vmovsd %xmm1, 512(%rdx,%rax,4) {%k2}

// CHECK: vmovsd %xmm1, -512(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x4c,0x82,0xc0]
vmovsd %xmm1, -512(%rdx,%rax,4) {%k2}

// CHECK: vmovsd %xmm1, 512(%rdx,%rax)
// CHECK: encoding: [0xc5,0xfb,0x11,0x8c,0x02,0x00,0x02,0x00,0x00]
vmovsd %xmm1, 512(%rdx,%rax)

// CHECK: vmovsd %xmm1, 512(%rdx,%rax) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x4c,0x02,0x40]
vmovsd %xmm1, 512(%rdx,%rax) {%k2}

// CHECK: vmovsd %xmm15, 485498096
// CHECK: encoding: [0xc5,0x7b,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd %xmm15, 485498096

// CHECK: vmovsd %xmm15, 485498096 {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovsd %xmm15, 485498096 {%k2}

// CHECK: vmovsd %xmm15, 512(%rdx)
// CHECK: encoding: [0xc5,0x7b,0x11,0xba,0x00,0x02,0x00,0x00]
vmovsd %xmm15, 512(%rdx)

// CHECK: vmovsd %xmm15, 512(%rdx) {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x11,0x7a,0x40]
vmovsd %xmm15, 512(%rdx) {%k2}

// CHECK: vmovsd %xmm15, 512(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0x7b,0x11,0xbc,0x82,0x00,0x02,0x00,0x00]
vmovsd %xmm15, 512(%rdx,%rax,4)

// CHECK: vmovsd %xmm15, -512(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0x7b,0x11,0xbc,0x82,0x00,0xfe,0xff,0xff]
vmovsd %xmm15, -512(%rdx,%rax,4)

// CHECK: vmovsd %xmm15, 512(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x11,0x7c,0x82,0x40]
vmovsd %xmm15, 512(%rdx,%rax,4) {%k2}

// CHECK: vmovsd %xmm15, -512(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x11,0x7c,0x82,0xc0]
vmovsd %xmm15, -512(%rdx,%rax,4) {%k2}

// CHECK: vmovsd %xmm15, 512(%rdx,%rax)
// CHECK: encoding: [0xc5,0x7b,0x11,0xbc,0x02,0x00,0x02,0x00,0x00]
vmovsd %xmm15, 512(%rdx,%rax)

// CHECK: vmovsd %xmm15, 512(%rdx,%rax) {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x11,0x7c,0x02,0x40]
vmovsd %xmm15, 512(%rdx,%rax) {%k2}

// CHECK: vmovsd %xmm15, (%rdx)
// CHECK: encoding: [0xc5,0x7b,0x11,0x3a]
vmovsd %xmm15, (%rdx)

// CHECK: vmovsd %xmm15, (%rdx) {%k2}
// CHECK: encoding: [0x62,0x71,0xff,0x0a,0x11,0x3a]
vmovsd %xmm15, (%rdx) {%k2}

// CHECK: vmovsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x10,0xff]
vmovsd %xmm15, %xmm15, %xmm15

// CHECK: vmovsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x10,0xff]
vmovsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmovsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x10,0xff]
vmovsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmovsd %xmm1, (%rdx)
// CHECK: encoding: [0xc5,0xfb,0x11,0x0a]
vmovsd %xmm1, (%rdx)

// CHECK: vmovsd %xmm1, (%rdx) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x0a]
vmovsd %xmm1, (%rdx) {%k2}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x10,0xc9]
vmovsd %xmm1, %xmm1, %xmm1

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x10,0xc9]
vmovsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x10,0xc9]
vmovsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovss 256(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x7a,0x10,0xbc,0x82,0x00,0x01,0x00,0x00]
vmovss 256(%rdx,%rax,4), %xmm15

// CHECK: vmovss -256(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x7a,0x10,0xbc,0x82,0x00,0xff,0xff,0xff]
vmovss -256(%rdx,%rax,4), %xmm15

// CHECK: vmovss 256(%rdx,%rax,4), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x10,0x7c,0x82,0x40]
vmovss 256(%rdx,%rax,4), %xmm15 {%k2}

// CHECK: vmovss -256(%rdx,%rax,4), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x10,0x7c,0x82,0xc0]
vmovss -256(%rdx,%rax,4), %xmm15 {%k2}

// CHECK: vmovss 256(%rdx,%rax,4), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x7e,0x8a,0x10,0x7c,0x82,0x40]
vmovss 256(%rdx,%rax,4), %xmm15 {%k2} {z}

// CHECK: vmovss -256(%rdx,%rax,4), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x7e,0x8a,0x10,0x7c,0x82,0xc0]
vmovss -256(%rdx,%rax,4), %xmm15 {%k2} {z}

// CHECK: vmovss 256(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xfa,0x10,0x8c,0x82,0x00,0x01,0x00,0x00]
vmovss 256(%rdx,%rax,4), %xmm1

// CHECK: vmovss -256(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xfa,0x10,0x8c,0x82,0x00,0xff,0xff,0xff]
vmovss -256(%rdx,%rax,4), %xmm1

// CHECK: vmovss 256(%rdx,%rax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x4c,0x82,0x40]
vmovss 256(%rdx,%rax,4), %xmm1 {%k2}

// CHECK: vmovss -256(%rdx,%rax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x4c,0x82,0xc0]
vmovss -256(%rdx,%rax,4), %xmm1 {%k2}

// CHECK: vmovss 256(%rdx,%rax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x4c,0x82,0x40]
vmovss 256(%rdx,%rax,4), %xmm1 {%k2} {z}

// CHECK: vmovss -256(%rdx,%rax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x4c,0x82,0xc0]
vmovss -256(%rdx,%rax,4), %xmm1 {%k2} {z}

// CHECK: vmovss 256(%rdx,%rax), %xmm15
// CHECK: encoding: [0xc5,0x7a,0x10,0xbc,0x02,0x00,0x01,0x00,0x00]
vmovss 256(%rdx,%rax), %xmm15

// CHECK: vmovss 256(%rdx,%rax), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x10,0x7c,0x02,0x40]
vmovss 256(%rdx,%rax), %xmm15 {%k2}

// CHECK: vmovss 256(%rdx,%rax), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x7e,0x8a,0x10,0x7c,0x02,0x40]
vmovss 256(%rdx,%rax), %xmm15 {%k2} {z}

// CHECK: vmovss 256(%rdx,%rax), %xmm1
// CHECK: encoding: [0xc5,0xfa,0x10,0x8c,0x02,0x00,0x01,0x00,0x00]
vmovss 256(%rdx,%rax), %xmm1

// CHECK: vmovss 256(%rdx,%rax), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x4c,0x02,0x40]
vmovss 256(%rdx,%rax), %xmm1 {%k2}

// CHECK: vmovss 256(%rdx,%rax), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x4c,0x02,0x40]
vmovss 256(%rdx,%rax), %xmm1 {%k2} {z}

// CHECK: vmovss 256(%rdx), %xmm15
// CHECK: encoding: [0xc5,0x7a,0x10,0xba,0x00,0x01,0x00,0x00]
vmovss 256(%rdx), %xmm15

// CHECK: vmovss 256(%rdx), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x10,0x7a,0x40]
vmovss 256(%rdx), %xmm15 {%k2}

// CHECK: vmovss 256(%rdx), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x7e,0x8a,0x10,0x7a,0x40]
vmovss 256(%rdx), %xmm15 {%k2} {z}

// CHECK: vmovss 256(%rdx), %xmm1
// CHECK: encoding: [0xc5,0xfa,0x10,0x8a,0x00,0x01,0x00,0x00]
vmovss 256(%rdx), %xmm1

// CHECK: vmovss 256(%rdx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x4a,0x40]
vmovss 256(%rdx), %xmm1 {%k2}

// CHECK: vmovss 256(%rdx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x4a,0x40]
vmovss 256(%rdx), %xmm1 {%k2} {z}

// CHECK: vmovss 485498096, %xmm15
// CHECK: encoding: [0xc5,0x7a,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm15

// CHECK: vmovss 485498096, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm15 {%k2}

// CHECK: vmovss 485498096, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x7e,0x8a,0x10,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm15 {%k2} {z}

// CHECK: vmovss 485498096, %xmm1
// CHECK: encoding: [0xc5,0xfa,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm1

// CHECK: vmovss 485498096, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm1 {%k2}

// CHECK: vmovss 485498096, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm1 {%k2} {z}

// CHECK: vmovss (%rdx), %xmm15
// CHECK: encoding: [0xc5,0x7a,0x10,0x3a]
vmovss (%rdx), %xmm15

// CHECK: vmovss (%rdx), %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x10,0x3a]
vmovss (%rdx), %xmm15 {%k2}

// CHECK: vmovss (%rdx), %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x7e,0x8a,0x10,0x3a]
vmovss (%rdx), %xmm15 {%k2} {z}

// CHECK: vmovss (%rdx), %xmm1
// CHECK: encoding: [0xc5,0xfa,0x10,0x0a]
vmovss (%rdx), %xmm1

// CHECK: vmovss (%rdx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x0a]
vmovss (%rdx), %xmm1 {%k2}

// CHECK: vmovss (%rdx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x0a]
vmovss (%rdx), %xmm1 {%k2} {z}

// CHECK: vmovss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x11,0xff]
vmovss.s %xmm15, %xmm15, %xmm15

// CHECK: vmovss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x11,0xff]
vmovss.s %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmovss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x11,0xff]
vmovss.s %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmovss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x11,0xc9]
vmovss.s %xmm1, %xmm1, %xmm1

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x11,0xc9]
vmovss.s %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x11,0xc9]
vmovss.s %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovss %xmm1, 256(%rdx)
// CHECK: encoding: [0xc5,0xfa,0x11,0x8a,0x00,0x01,0x00,0x00]
vmovss %xmm1, 256(%rdx)

// CHECK: vmovss %xmm1, 256(%rdx) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x4a,0x40]
vmovss %xmm1, 256(%rdx) {%k2}

// CHECK: vmovss %xmm1, 256(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0xfa,0x11,0x8c,0x82,0x00,0x01,0x00,0x00]
vmovss %xmm1, 256(%rdx,%rax,4)

// CHECK: vmovss %xmm1, -256(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0xfa,0x11,0x8c,0x82,0x00,0xff,0xff,0xff]
vmovss %xmm1, -256(%rdx,%rax,4)

// CHECK: vmovss %xmm1, 256(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x4c,0x82,0x40]
vmovss %xmm1, 256(%rdx,%rax,4) {%k2}

// CHECK: vmovss %xmm1, -256(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x4c,0x82,0xc0]
vmovss %xmm1, -256(%rdx,%rax,4) {%k2}

// CHECK: vmovss %xmm1, 256(%rdx,%rax)
// CHECK: encoding: [0xc5,0xfa,0x11,0x8c,0x02,0x00,0x01,0x00,0x00]
vmovss %xmm1, 256(%rdx,%rax)

// CHECK: vmovss %xmm1, 256(%rdx,%rax) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x4c,0x02,0x40]
vmovss %xmm1, 256(%rdx,%rax) {%k2}

// CHECK: vmovss %xmm1, 485498096
// CHECK: encoding: [0xc5,0xfa,0x11,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss %xmm1, 485498096

// CHECK: vmovss %xmm1, 485498096 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss %xmm1, 485498096 {%k2}

// CHECK: vmovss %xmm15, 256(%rdx)
// CHECK: encoding: [0xc5,0x7a,0x11,0xba,0x00,0x01,0x00,0x00]
vmovss %xmm15, 256(%rdx)

// CHECK: vmovss %xmm15, 256(%rdx) {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x11,0x7a,0x40]
vmovss %xmm15, 256(%rdx) {%k2}

// CHECK: vmovss %xmm15, 256(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0x7a,0x11,0xbc,0x82,0x00,0x01,0x00,0x00]
vmovss %xmm15, 256(%rdx,%rax,4)

// CHECK: vmovss %xmm15, -256(%rdx,%rax,4)
// CHECK: encoding: [0xc5,0x7a,0x11,0xbc,0x82,0x00,0xff,0xff,0xff]
vmovss %xmm15, -256(%rdx,%rax,4)

// CHECK: vmovss %xmm15, 256(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x11,0x7c,0x82,0x40]
vmovss %xmm15, 256(%rdx,%rax,4) {%k2}

// CHECK: vmovss %xmm15, -256(%rdx,%rax,4) {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x11,0x7c,0x82,0xc0]
vmovss %xmm15, -256(%rdx,%rax,4) {%k2}

// CHECK: vmovss %xmm15, 256(%rdx,%rax)
// CHECK: encoding: [0xc5,0x7a,0x11,0xbc,0x02,0x00,0x01,0x00,0x00]
vmovss %xmm15, 256(%rdx,%rax)

// CHECK: vmovss %xmm15, 256(%rdx,%rax) {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x11,0x7c,0x02,0x40]
vmovss %xmm15, 256(%rdx,%rax) {%k2}

// CHECK: vmovss %xmm15, 485498096
// CHECK: encoding: [0xc5,0x7a,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss %xmm15, 485498096

// CHECK: vmovss %xmm15, 485498096 {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x11,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmovss %xmm15, 485498096 {%k2}

// CHECK: vmovss %xmm15, (%rdx)
// CHECK: encoding: [0xc5,0x7a,0x11,0x3a]
vmovss %xmm15, (%rdx)

// CHECK: vmovss %xmm15, (%rdx) {%k2}
// CHECK: encoding: [0x62,0x71,0x7e,0x0a,0x11,0x3a]
vmovss %xmm15, (%rdx) {%k2}

// CHECK: vmovss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x10,0xff]
vmovss %xmm15, %xmm15, %xmm15

// CHECK: vmovss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x10,0xff]
vmovss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmovss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x10,0xff]
vmovss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmovss %xmm1, (%rdx)
// CHECK: encoding: [0xc5,0xfa,0x11,0x0a]
vmovss %xmm1, (%rdx)

// CHECK: vmovss %xmm1, (%rdx) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x0a]
vmovss %xmm1, (%rdx) {%k2}

// CHECK: vmovss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x10,0xc9]
vmovss %xmm1, %xmm1, %xmm1

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x10,0xc9]
vmovss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x10,0xc9]
vmovss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm15, %xmm15

// CHECK: vmulsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vmulsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm1, %xmm1

// CHECK: vmulsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x59,0xbc,0x82,0x00,0x02,0x00,0x00]
vmulsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmulsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x59,0xbc,0x82,0x00,0xfe,0xff,0xff]
vmulsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmulsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x59,0x7c,0x82,0x40]
vmulsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmulsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x59,0x7c,0x82,0xc0]
vmulsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmulsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x59,0x7c,0x82,0x40]
vmulsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x59,0x7c,0x82,0xc0]
vmulsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x59,0x8c,0x82,0x00,0x02,0x00,0x00]
vmulsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmulsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x59,0x8c,0x82,0x00,0xfe,0xff,0xff]
vmulsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmulsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x4c,0x82,0x40]
vmulsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x4c,0x82,0xc0]
vmulsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x4c,0x82,0x40]
vmulsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x4c,0x82,0xc0]
vmulsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x59,0xbc,0x02,0x00,0x02,0x00,0x00]
vmulsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vmulsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x59,0x7c,0x02,0x40]
vmulsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vmulsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x59,0x7c,0x02,0x40]
vmulsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x59,0x8c,0x02,0x00,0x02,0x00,0x00]
vmulsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vmulsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x4c,0x02,0x40]
vmulsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x4c,0x02,0x40]
vmulsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x59,0xba,0x00,0x02,0x00,0x00]
vmulsd 512(%rdx), %xmm15, %xmm15

// CHECK: vmulsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x59,0x7a,0x40]
vmulsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmulsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x59,0x7a,0x40]
vmulsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x59,0x8a,0x00,0x02,0x00,0x00]
vmulsd 512(%rdx), %xmm1, %xmm1

// CHECK: vmulsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x4a,0x40]
vmulsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x4a,0x40]
vmulsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x59,0xff]
vmulsd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x3a,0x59,0xff]
vmulsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xba,0x59,0xff]
vmulsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x59,0xc9]
vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x59,0xc9]
vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x59,0xc9]
vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x59,0x3a]
vmulsd (%rdx), %xmm15, %xmm15

// CHECK: vmulsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x59,0x3a]
vmulsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmulsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x59,0x3a]
vmulsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x59,0x0a]
vmulsd (%rdx), %xmm1, %xmm1

// CHECK: vmulsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x0a]
vmulsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x0a]
vmulsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x59,0xff]
vmulsd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x59,0xff]
vmulsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x59,0xff]
vmulsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x59,0xc9]
vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x59,0xc9]
vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x59,0xc9]
vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x59,0xff]
vmulsd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x5a,0x59,0xff]
vmulsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xda,0x59,0xff]
vmulsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x59,0xc9]
vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x59,0xc9]
vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x59,0xc9]
vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x59,0xff]
vmulsd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x7a,0x59,0xff]
vmulsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xfa,0x59,0xff]
vmulsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x59,0xc9]
vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x59,0xc9]
vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x59,0xc9]
vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x59,0xff]
vmulsd %xmm15, %xmm15, %xmm15

// CHECK: vmulsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x59,0xff]
vmulsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x59,0xff]
vmulsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x59,0xc9]
vmulsd %xmm1, %xmm1, %xmm1

// CHECK: vmulsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0xc9]
vmulsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0xc9]
vmulsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x59,0xbc,0x82,0x00,0x01,0x00,0x00]
vmulss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmulss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x59,0xbc,0x82,0x00,0xff,0xff,0xff]
vmulss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vmulss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x59,0x7c,0x82,0x40]
vmulss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmulss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x59,0x7c,0x82,0xc0]
vmulss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vmulss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x59,0x7c,0x82,0x40]
vmulss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x59,0x7c,0x82,0xc0]
vmulss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x59,0x8c,0x82,0x00,0x01,0x00,0x00]
vmulss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmulss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x59,0x8c,0x82,0x00,0xff,0xff,0xff]
vmulss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vmulss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x4c,0x82,0x40]
vmulss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x4c,0x82,0xc0]
vmulss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x4c,0x82,0x40]
vmulss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x4c,0x82,0xc0]
vmulss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x59,0xbc,0x02,0x00,0x01,0x00,0x00]
vmulss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vmulss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x59,0x7c,0x02,0x40]
vmulss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vmulss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x59,0x7c,0x02,0x40]
vmulss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x59,0x8c,0x02,0x00,0x01,0x00,0x00]
vmulss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vmulss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x4c,0x02,0x40]
vmulss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vmulss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x4c,0x02,0x40]
vmulss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x59,0xba,0x00,0x01,0x00,0x00]
vmulss 256(%rdx), %xmm15, %xmm15

// CHECK: vmulss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x59,0x7a,0x40]
vmulss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmulss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x59,0x7a,0x40]
vmulss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x59,0x8a,0x00,0x01,0x00,0x00]
vmulss 256(%rdx), %xmm1, %xmm1

// CHECK: vmulss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x4a,0x40]
vmulss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmulss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x4a,0x40]
vmulss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm15, %xmm15

// CHECK: vmulss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vmulss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x59,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm1, %xmm1

// CHECK: vmulss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmulss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x38,0x59,0xff]
vmulss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x3a,0x59,0xff]
vmulss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xba,0x59,0xff]
vmulss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x59,0xc9]
vmulss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x59,0xc9]
vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x59,0xc9]
vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x59,0x3a]
vmulss (%rdx), %xmm15, %xmm15

// CHECK: vmulss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x59,0x3a]
vmulss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vmulss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x59,0x3a]
vmulss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x59,0x0a]
vmulss (%rdx), %xmm1, %xmm1

// CHECK: vmulss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x0a]
vmulss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vmulss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x0a]
vmulss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x59,0xff]
vmulss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x59,0xff]
vmulss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x59,0xff]
vmulss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x59,0xc9]
vmulss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x59,0xc9]
vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x59,0xc9]
vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x58,0x59,0xff]
vmulss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x5a,0x59,0xff]
vmulss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xda,0x59,0xff]
vmulss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x59,0xc9]
vmulss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x59,0xc9]
vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x59,0xc9]
vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x78,0x59,0xff]
vmulss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vmulss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x7a,0x59,0xff]
vmulss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xfa,0x59,0xff]
vmulss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x59,0xc9]
vmulss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x59,0xc9]
vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x59,0xc9]
vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x59,0xff]
vmulss %xmm15, %xmm15, %xmm15

// CHECK: vmulss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x59,0xff]
vmulss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vmulss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x59,0xff]
vmulss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vmulss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x59,0xc9]
vmulss %xmm1, %xmm1, %xmm1

// CHECK: vmulss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0xc9]
vmulss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0xc9]
vmulss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm15, %xmm15

// CHECK: vrcp14sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vrcp14sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm1, %xmm1

// CHECK: vrcp14sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4d,0x7c,0x82,0x40]
vrcp14sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrcp14sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4d,0x7c,0x82,0xc0]
vrcp14sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrcp14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4d,0x7c,0x82,0x40]
vrcp14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4d,0x7c,0x82,0xc0]
vrcp14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4d,0x7c,0x82,0x40]
vrcp14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4d,0x7c,0x82,0xc0]
vrcp14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x4c,0x82,0x40]
vrcp14sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrcp14sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x4c,0x82,0xc0]
vrcp14sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrcp14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x4c,0x82,0x40]
vrcp14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x4c,0x82,0xc0]
vrcp14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x4c,0x82,0x40]
vrcp14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x4c,0x82,0xc0]
vrcp14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4d,0x7c,0x02,0x40]
vrcp14sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vrcp14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4d,0x7c,0x02,0x40]
vrcp14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4d,0x7c,0x02,0x40]
vrcp14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x4c,0x02,0x40]
vrcp14sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vrcp14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x4c,0x02,0x40]
vrcp14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x4c,0x02,0x40]
vrcp14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4d,0x7a,0x40]
vrcp14sd 512(%rdx), %xmm15, %xmm15

// CHECK: vrcp14sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4d,0x7a,0x40]
vrcp14sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4d,0x7a,0x40]
vrcp14sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x4a,0x40]
vrcp14sd 512(%rdx), %xmm1, %xmm1

// CHECK: vrcp14sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x4a,0x40]
vrcp14sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x4a,0x40]
vrcp14sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4d,0x3a]
vrcp14sd (%rdx), %xmm15, %xmm15

// CHECK: vrcp14sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4d,0x3a]
vrcp14sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4d,0x3a]
vrcp14sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x0a]
vrcp14sd (%rdx), %xmm1, %xmm1

// CHECK: vrcp14sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x0a]
vrcp14sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x0a]
vrcp14sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x08,0x4d,0xff]
vrcp14sd %xmm15, %xmm15, %xmm15

// CHECK: vrcp14sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x4d,0xff]
vrcp14sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrcp14sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x4d,0xff]
vrcp14sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0xc9]
vrcp14sd %xmm1, %xmm1, %xmm1

// CHECK: vrcp14sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0xc9]
vrcp14sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0xc9]
vrcp14sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4d,0x7c,0x82,0x40]
vrcp14ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrcp14ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4d,0x7c,0x82,0xc0]
vrcp14ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrcp14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4d,0x7c,0x82,0x40]
vrcp14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4d,0x7c,0x82,0xc0]
vrcp14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4d,0x7c,0x82,0x40]
vrcp14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4d,0x7c,0x82,0xc0]
vrcp14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x4c,0x82,0x40]
vrcp14ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrcp14ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x4c,0x82,0xc0]
vrcp14ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrcp14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x4c,0x82,0x40]
vrcp14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x4c,0x82,0xc0]
vrcp14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x4c,0x82,0x40]
vrcp14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x4c,0x82,0xc0]
vrcp14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4d,0x7c,0x02,0x40]
vrcp14ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vrcp14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4d,0x7c,0x02,0x40]
vrcp14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4d,0x7c,0x02,0x40]
vrcp14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x4c,0x02,0x40]
vrcp14ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vrcp14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x4c,0x02,0x40]
vrcp14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x4c,0x02,0x40]
vrcp14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4d,0x7a,0x40]
vrcp14ss 256(%rdx), %xmm15, %xmm15

// CHECK: vrcp14ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4d,0x7a,0x40]
vrcp14ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4d,0x7a,0x40]
vrcp14ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x4a,0x40]
vrcp14ss 256(%rdx), %xmm1, %xmm1

// CHECK: vrcp14ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x4a,0x40]
vrcp14ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x4a,0x40]
vrcp14ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm15, %xmm15

// CHECK: vrcp14ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vrcp14ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm1, %xmm1

// CHECK: vrcp14ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4d,0x3a]
vrcp14ss (%rdx), %xmm15, %xmm15

// CHECK: vrcp14ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4d,0x3a]
vrcp14ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrcp14ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4d,0x3a]
vrcp14ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x0a]
vrcp14ss (%rdx), %xmm1, %xmm1

// CHECK: vrcp14ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x0a]
vrcp14ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x0a]
vrcp14ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x08,0x4d,0xff]
vrcp14ss %xmm15, %xmm15, %xmm15

// CHECK: vrcp14ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x4d,0xff]
vrcp14ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrcp14ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x4d,0xff]
vrcp14ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrcp14ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0xc9]
vrcp14ss %xmm1, %xmm1, %xmm1

// CHECK: vrcp14ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0xc9]
vrcp14ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0xc9]
vrcp14ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x0b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm15, %xmm15

// CHECK: vrndscalesd $0, 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x0b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x0b,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm1, %xmm1

// CHECK: vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x0b,0x7c,0x82,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrndscalesd $0, -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x0b,0x7c,0x82,0xc0,0x00]
vrndscalesd $0, -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrndscalesd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x0b,0x7c,0x82,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x0b,0x7c,0x82,0xc0,0x00]
vrndscalesd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x0b,0x7c,0x82,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x0b,0x7c,0x82,0xc0,0x00]
vrndscalesd $0, -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x4c,0x82,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrndscalesd $0, -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x4c,0x82,0xc0,0x00]
vrndscalesd $0, -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrndscalesd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x4c,0x82,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x4c,0x82,0xc0,0x00]
vrndscalesd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x4c,0x82,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x4c,0x82,0xc0,0x00]
vrndscalesd $0, -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x0b,0x7c,0x02,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vrndscalesd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x0b,0x7c,0x02,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x0b,0x7c,0x02,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x4c,0x02,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vrndscalesd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x4c,0x02,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x4c,0x02,0x40,0x00]
vrndscalesd $0, 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x0b,0x7a,0x40,0x00]
vrndscalesd $0, 512(%rdx), %xmm15, %xmm15

// CHECK: vrndscalesd $0, 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x0b,0x7a,0x40,0x00]
vrndscalesd $0, 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x0b,0x7a,0x40,0x00]
vrndscalesd $0, 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x4a,0x40,0x00]
vrndscalesd $0, 512(%rdx), %xmm1, %xmm1

// CHECK: vrndscalesd $0, 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x4a,0x40,0x00]
vrndscalesd $0, 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x4a,0x40,0x00]
vrndscalesd $0, 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x85,0x08,0x0b,0x3a,0x00]
vrndscalesd $0, (%rdx), %xmm15, %xmm15

// CHECK: vrndscalesd $0, (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x85,0x0a,0x0b,0x3a,0x00]
vrndscalesd $0, (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x85,0x8a,0x0b,0x3a,0x00]
vrndscalesd $0, (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x0a,0x00]
vrndscalesd $0, (%rdx), %xmm1, %xmm1

// CHECK: vrndscalesd $0, (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x0a,0x00]
vrndscalesd $0, (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x0a,0x00]
vrndscalesd $0, (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x85,0x18,0x0b,0xff,0x00]
vrndscalesd $0, {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vrndscalesd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x85,0x1a,0x0b,0xff,0x00]
vrndscalesd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x85,0x9a,0x0b,0xff,0x00]
vrndscalesd $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x18,0x0b,0xc9,0x00]
vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x1a,0x0b,0xc9,0x00]
vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x9a,0x0b,0xc9,0x00]
vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x85,0x08,0x0b,0xff,0x00]
vrndscalesd $0, %xmm15, %xmm15, %xmm15

// CHECK: vrndscalesd $0, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x85,0x0a,0x0b,0xff,0x00]
vrndscalesd $0, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrndscalesd $0, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x85,0x8a,0x0b,0xff,0x00]
vrndscalesd $0, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscalesd $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0xc9,0x00]
vrndscalesd $0, %xmm1, %xmm1, %xmm1

// CHECK: vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0xc9,0x00]
vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0xc9,0x00]
vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x0a,0x7c,0x82,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrndscaless $0, -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x0a,0x7c,0x82,0xc0,0x00]
vrndscaless $0, -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrndscaless $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x0a,0x7c,0x82,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x0a,0x7c,0x82,0xc0,0x00]
vrndscaless $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x0a,0x7c,0x82,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x0a,0x7c,0x82,0xc0,0x00]
vrndscaless $0, -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x4c,0x82,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrndscaless $0, -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x4c,0x82,0xc0,0x00]
vrndscaless $0, -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrndscaless $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x4c,0x82,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x4c,0x82,0xc0,0x00]
vrndscaless $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x4c,0x82,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x4c,0x82,0xc0,0x00]
vrndscaless $0, -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x0a,0x7c,0x02,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vrndscaless $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x0a,0x7c,0x02,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x0a,0x7c,0x02,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x4c,0x02,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vrndscaless $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x4c,0x02,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x4c,0x02,0x40,0x00]
vrndscaless $0, 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x0a,0x7a,0x40,0x00]
vrndscaless $0, 256(%rdx), %xmm15, %xmm15

// CHECK: vrndscaless $0, 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x0a,0x7a,0x40,0x00]
vrndscaless $0, 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x0a,0x7a,0x40,0x00]
vrndscaless $0, 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x4a,0x40,0x00]
vrndscaless $0, 256(%rdx), %xmm1, %xmm1

// CHECK: vrndscaless $0, 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x4a,0x40,0x00]
vrndscaless $0, 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x4a,0x40,0x00]
vrndscaless $0, 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x0a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm15, %xmm15

// CHECK: vrndscaless $0, 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x0a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x0a,0x3c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm1, %xmm1

// CHECK: vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x0c,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x73,0x05,0x08,0x0a,0x3a,0x00]
vrndscaless $0, (%rdx), %xmm15, %xmm15

// CHECK: vrndscaless $0, (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x73,0x05,0x0a,0x0a,0x3a,0x00]
vrndscaless $0, (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x73,0x05,0x8a,0x0a,0x3a,0x00]
vrndscaless $0, (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x0a,0x00]
vrndscaless $0, (%rdx), %xmm1, %xmm1

// CHECK: vrndscaless $0, (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x0a,0x00]
vrndscaless $0, (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x0a,0x00]
vrndscaless $0, (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, {sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x05,0x18,0x0a,0xff,0x00]
vrndscaless $0, {sae}, %xmm15, %xmm15, %xmm15

// CHECK: vrndscaless $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x05,0x1a,0x0a,0xff,0x00]
vrndscaless $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x05,0x9a,0x0a,0xff,0x00]
vrndscaless $0, {sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x18,0x0a,0xc9,0x00]
vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x1a,0x0a,0xc9,0x00]
vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x9a,0x0a,0xc9,0x00]
vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x53,0x05,0x08,0x0a,0xff,0x00]
vrndscaless $0, %xmm15, %xmm15, %xmm15

// CHECK: vrndscaless $0, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x53,0x05,0x0a,0x0a,0xff,0x00]
vrndscaless $0, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrndscaless $0, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x53,0x05,0x8a,0x0a,0xff,0x00]
vrndscaless $0, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrndscaless $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0xc9,0x00]
vrndscaless $0, %xmm1, %xmm1, %xmm1

// CHECK: vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0xc9,0x00]
vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0xc9,0x00]
vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm15, %xmm15

// CHECK: vrsqrt14sd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14sd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm1, %xmm1

// CHECK: vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4f,0x7c,0x82,0x40]
vrsqrt14sd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrsqrt14sd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4f,0x7c,0x82,0xc0]
vrsqrt14sd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrsqrt14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4f,0x7c,0x82,0x40]
vrsqrt14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4f,0x7c,0x82,0xc0]
vrsqrt14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4f,0x7c,0x82,0x40]
vrsqrt14sd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4f,0x7c,0x82,0xc0]
vrsqrt14sd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14sd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x4c,0x82,0x40]
vrsqrt14sd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrsqrt14sd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x4c,0x82,0xc0]
vrsqrt14sd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrsqrt14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x4c,0x82,0x40]
vrsqrt14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x4c,0x82,0xc0]
vrsqrt14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x4c,0x82,0x40]
vrsqrt14sd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x4c,0x82,0xc0]
vrsqrt14sd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4f,0x7c,0x02,0x40]
vrsqrt14sd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vrsqrt14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4f,0x7c,0x02,0x40]
vrsqrt14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4f,0x7c,0x02,0x40]
vrsqrt14sd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14sd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x4c,0x02,0x40]
vrsqrt14sd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vrsqrt14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x4c,0x02,0x40]
vrsqrt14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x4c,0x02,0x40]
vrsqrt14sd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4f,0x7a,0x40]
vrsqrt14sd 512(%rdx), %xmm15, %xmm15

// CHECK: vrsqrt14sd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4f,0x7a,0x40]
vrsqrt14sd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4f,0x7a,0x40]
vrsqrt14sd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14sd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x4a,0x40]
vrsqrt14sd 512(%rdx), %xmm1, %xmm1

// CHECK: vrsqrt14sd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x4a,0x40]
vrsqrt14sd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x4a,0x40]
vrsqrt14sd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x4f,0x3a]
vrsqrt14sd (%rdx), %xmm15, %xmm15

// CHECK: vrsqrt14sd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x4f,0x3a]
vrsqrt14sd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14sd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x4f,0x3a]
vrsqrt14sd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14sd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x0a]
vrsqrt14sd (%rdx), %xmm1, %xmm1

// CHECK: vrsqrt14sd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x0a]
vrsqrt14sd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x0a]
vrsqrt14sd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x08,0x4f,0xff]
vrsqrt14sd %xmm15, %xmm15, %xmm15

// CHECK: vrsqrt14sd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x4f,0xff]
vrsqrt14sd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14sd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x4f,0xff]
vrsqrt14sd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0xc9]
vrsqrt14sd %xmm1, %xmm1, %xmm1

// CHECK: vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0xc9]
vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0xc9]
vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4f,0x7c,0x82,0x40]
vrsqrt14ss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrsqrt14ss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4f,0x7c,0x82,0xc0]
vrsqrt14ss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vrsqrt14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4f,0x7c,0x82,0x40]
vrsqrt14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4f,0x7c,0x82,0xc0]
vrsqrt14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4f,0x7c,0x82,0x40]
vrsqrt14ss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4f,0x7c,0x82,0xc0]
vrsqrt14ss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14ss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x4c,0x82,0x40]
vrsqrt14ss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrsqrt14ss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x4c,0x82,0xc0]
vrsqrt14ss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vrsqrt14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x4c,0x82,0x40]
vrsqrt14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x4c,0x82,0xc0]
vrsqrt14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x4c,0x82,0x40]
vrsqrt14ss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x4c,0x82,0xc0]
vrsqrt14ss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4f,0x7c,0x02,0x40]
vrsqrt14ss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vrsqrt14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4f,0x7c,0x02,0x40]
vrsqrt14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4f,0x7c,0x02,0x40]
vrsqrt14ss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14ss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x4c,0x02,0x40]
vrsqrt14ss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vrsqrt14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x4c,0x02,0x40]
vrsqrt14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x4c,0x02,0x40]
vrsqrt14ss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4f,0x7a,0x40]
vrsqrt14ss 256(%rdx), %xmm15, %xmm15

// CHECK: vrsqrt14ss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4f,0x7a,0x40]
vrsqrt14ss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4f,0x7a,0x40]
vrsqrt14ss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14ss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x4a,0x40]
vrsqrt14ss 256(%rdx), %xmm1, %xmm1

// CHECK: vrsqrt14ss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x4a,0x40]
vrsqrt14ss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x4a,0x40]
vrsqrt14ss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm15, %xmm15

// CHECK: vrsqrt14ss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14ss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4f,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm1, %xmm1

// CHECK: vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x4f,0x3a]
vrsqrt14ss (%rdx), %xmm15, %xmm15

// CHECK: vrsqrt14ss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x4f,0x3a]
vrsqrt14ss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14ss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x4f,0x3a]
vrsqrt14ss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14ss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x0a]
vrsqrt14ss (%rdx), %xmm1, %xmm1

// CHECK: vrsqrt14ss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x0a]
vrsqrt14ss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x0a]
vrsqrt14ss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x08,0x4f,0xff]
vrsqrt14ss %xmm15, %xmm15, %xmm15

// CHECK: vrsqrt14ss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x4f,0xff]
vrsqrt14ss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vrsqrt14ss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x4f,0xff]
vrsqrt14ss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vrsqrt14ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0xc9]
vrsqrt14ss %xmm1, %xmm1, %xmm1

// CHECK: vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0xc9]
vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0xc9]
vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm15, %xmm15

// CHECK: vscalefsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm1, %xmm1

// CHECK: vscalefsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x2d,0x7c,0x82,0x40]
vscalefsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vscalefsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x2d,0x7c,0x82,0xc0]
vscalefsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vscalefsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x2d,0x7c,0x82,0x40]
vscalefsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x2d,0x7c,0x82,0xc0]
vscalefsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x2d,0x7c,0x82,0x40]
vscalefsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x2d,0x7c,0x82,0xc0]
vscalefsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x4c,0x82,0x40]
vscalefsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vscalefsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x4c,0x82,0xc0]
vscalefsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vscalefsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x4c,0x82,0x40]
vscalefsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x4c,0x82,0xc0]
vscalefsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x4c,0x82,0x40]
vscalefsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x4c,0x82,0xc0]
vscalefsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x2d,0x7c,0x02,0x40]
vscalefsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vscalefsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x2d,0x7c,0x02,0x40]
vscalefsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x2d,0x7c,0x02,0x40]
vscalefsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x4c,0x02,0x40]
vscalefsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vscalefsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x4c,0x02,0x40]
vscalefsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x4c,0x02,0x40]
vscalefsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x2d,0x7a,0x40]
vscalefsd 512(%rdx), %xmm15, %xmm15

// CHECK: vscalefsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x2d,0x7a,0x40]
vscalefsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x2d,0x7a,0x40]
vscalefsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x4a,0x40]
vscalefsd 512(%rdx), %xmm1, %xmm1

// CHECK: vscalefsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x4a,0x40]
vscalefsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x4a,0x40]
vscalefsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x38,0x2d,0xff]
vscalefsd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x3a,0x2d,0xff]
vscalefsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xba,0x2d,0xff]
vscalefsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x2d,0xc9]
vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x2d,0xc9]
vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x2d,0xc9]
vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x85,0x08,0x2d,0x3a]
vscalefsd (%rdx), %xmm15, %xmm15

// CHECK: vscalefsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x85,0x0a,0x2d,0x3a]
vscalefsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x85,0x8a,0x2d,0x3a]
vscalefsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x0a]
vscalefsd (%rdx), %xmm1, %xmm1

// CHECK: vscalefsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x0a]
vscalefsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x0a]
vscalefsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x18,0x2d,0xff]
vscalefsd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x1a,0x2d,0xff]
vscalefsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x9a,0x2d,0xff]
vscalefsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x2d,0xc9]
vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x2d,0xc9]
vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x2d,0xc9]
vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x58,0x2d,0xff]
vscalefsd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x5a,0x2d,0xff]
vscalefsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xda,0x2d,0xff]
vscalefsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x2d,0xc9]
vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x2d,0xc9]
vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x2d,0xc9]
vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x78,0x2d,0xff]
vscalefsd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x7a,0x2d,0xff]
vscalefsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0xfa,0x2d,0xff]
vscalefsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x2d,0xc9]
vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x2d,0xc9]
vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x2d,0xc9]
vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x85,0x08,0x2d,0xff]
vscalefsd %xmm15, %xmm15, %xmm15

// CHECK: vscalefsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x85,0x0a,0x2d,0xff]
vscalefsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x85,0x8a,0x2d,0xff]
vscalefsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0xc9]
vscalefsd %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0xc9]
vscalefsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0xc9]
vscalefsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x2d,0x7c,0x82,0x40]
vscalefss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vscalefss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x2d,0x7c,0x82,0xc0]
vscalefss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vscalefss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x2d,0x7c,0x82,0x40]
vscalefss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vscalefss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x2d,0x7c,0x82,0xc0]
vscalefss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vscalefss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x2d,0x7c,0x82,0x40]
vscalefss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x2d,0x7c,0x82,0xc0]
vscalefss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x4c,0x82,0x40]
vscalefss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vscalefss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x4c,0x82,0xc0]
vscalefss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vscalefss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x4c,0x82,0x40]
vscalefss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x4c,0x82,0xc0]
vscalefss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x4c,0x82,0x40]
vscalefss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x4c,0x82,0xc0]
vscalefss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x2d,0x7c,0x02,0x40]
vscalefss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vscalefss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x2d,0x7c,0x02,0x40]
vscalefss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vscalefss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x2d,0x7c,0x02,0x40]
vscalefss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x4c,0x02,0x40]
vscalefss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vscalefss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x4c,0x02,0x40]
vscalefss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x4c,0x02,0x40]
vscalefss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x2d,0x7a,0x40]
vscalefss 256(%rdx), %xmm15, %xmm15

// CHECK: vscalefss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x2d,0x7a,0x40]
vscalefss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vscalefss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x2d,0x7a,0x40]
vscalefss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x4a,0x40]
vscalefss 256(%rdx), %xmm1, %xmm1

// CHECK: vscalefss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x4a,0x40]
vscalefss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x4a,0x40]
vscalefss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm15, %xmm15

// CHECK: vscalefss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vscalefss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm1, %xmm1

// CHECK: vscalefss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x38,0x2d,0xff]
vscalefss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x3a,0x2d,0xff]
vscalefss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xba,0x2d,0xff]
vscalefss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x2d,0xc9]
vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x2d,0xc9]
vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x2d,0xc9]
vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0x62,0x72,0x05,0x08,0x2d,0x3a]
vscalefss (%rdx), %xmm15, %xmm15

// CHECK: vscalefss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x72,0x05,0x0a,0x2d,0x3a]
vscalefss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vscalefss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x72,0x05,0x8a,0x2d,0x3a]
vscalefss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x0a]
vscalefss (%rdx), %xmm1, %xmm1

// CHECK: vscalefss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x0a]
vscalefss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x0a]
vscalefss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x18,0x2d,0xff]
vscalefss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x1a,0x2d,0xff]
vscalefss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x9a,0x2d,0xff]
vscalefss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x2d,0xc9]
vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x2d,0xc9]
vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x2d,0xc9]
vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x58,0x2d,0xff]
vscalefss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x5a,0x2d,0xff]
vscalefss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xda,0x2d,0xff]
vscalefss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x2d,0xc9]
vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x2d,0xc9]
vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x2d,0xc9]
vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x78,0x2d,0xff]
vscalefss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vscalefss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x7a,0x2d,0xff]
vscalefss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0xfa,0x2d,0xff]
vscalefss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x2d,0xc9]
vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x2d,0xc9]
vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x2d,0xc9]
vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x52,0x05,0x08,0x2d,0xff]
vscalefss %xmm15, %xmm15, %xmm15

// CHECK: vscalefss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x52,0x05,0x0a,0x2d,0xff]
vscalefss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vscalefss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x52,0x05,0x8a,0x2d,0xff]
vscalefss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vscalefss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0xc9]
vscalefss %xmm1, %xmm1, %xmm1

// CHECK: vscalefss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0xc9]
vscalefss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0xc9]
vscalefss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm15, %xmm15

// CHECK: vsqrtsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm1, %xmm1

// CHECK: vsqrtsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x51,0xbc,0x82,0x00,0x02,0x00,0x00]
vsqrtsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsqrtsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x51,0xbc,0x82,0x00,0xfe,0xff,0xff]
vsqrtsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsqrtsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x51,0x7c,0x82,0x40]
vsqrtsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x51,0x7c,0x82,0xc0]
vsqrtsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x51,0x7c,0x82,0x40]
vsqrtsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x51,0x7c,0x82,0xc0]
vsqrtsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x51,0x8c,0x82,0x00,0x02,0x00,0x00]
vsqrtsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsqrtsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x51,0x8c,0x82,0x00,0xfe,0xff,0xff]
vsqrtsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsqrtsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x4c,0x82,0x40]
vsqrtsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x4c,0x82,0xc0]
vsqrtsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x4c,0x82,0x40]
vsqrtsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x4c,0x82,0xc0]
vsqrtsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x51,0xbc,0x02,0x00,0x02,0x00,0x00]
vsqrtsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vsqrtsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x51,0x7c,0x02,0x40]
vsqrtsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x51,0x7c,0x02,0x40]
vsqrtsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x51,0x8c,0x02,0x00,0x02,0x00,0x00]
vsqrtsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vsqrtsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x4c,0x02,0x40]
vsqrtsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x4c,0x02,0x40]
vsqrtsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x51,0xba,0x00,0x02,0x00,0x00]
vsqrtsd 512(%rdx), %xmm15, %xmm15

// CHECK: vsqrtsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x51,0x7a,0x40]
vsqrtsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x51,0x7a,0x40]
vsqrtsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x51,0x8a,0x00,0x02,0x00,0x00]
vsqrtsd 512(%rdx), %xmm1, %xmm1

// CHECK: vsqrtsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x4a,0x40]
vsqrtsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x4a,0x40]
vsqrtsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x51,0xff]
vsqrtsd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x3a,0x51,0xff]
vsqrtsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xba,0x51,0xff]
vsqrtsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x51,0xc9]
vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x51,0xc9]
vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x51,0xc9]
vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x51,0x3a]
vsqrtsd (%rdx), %xmm15, %xmm15

// CHECK: vsqrtsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x51,0x3a]
vsqrtsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x51,0x3a]
vsqrtsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x51,0x0a]
vsqrtsd (%rdx), %xmm1, %xmm1

// CHECK: vsqrtsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x0a]
vsqrtsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x0a]
vsqrtsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x51,0xff]
vsqrtsd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x51,0xff]
vsqrtsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x51,0xff]
vsqrtsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x51,0xc9]
vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x51,0xc9]
vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x51,0xc9]
vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x51,0xff]
vsqrtsd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x5a,0x51,0xff]
vsqrtsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xda,0x51,0xff]
vsqrtsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x51,0xc9]
vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x51,0xc9]
vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x51,0xc9]
vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x51,0xff]
vsqrtsd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x7a,0x51,0xff]
vsqrtsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xfa,0x51,0xff]
vsqrtsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x51,0xc9]
vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x51,0xc9]
vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x51,0xc9]
vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x51,0xff]
vsqrtsd %xmm15, %xmm15, %xmm15

// CHECK: vsqrtsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x51,0xff]
vsqrtsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x51,0xff]
vsqrtsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x51,0xc9]
vsqrtsd %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0xc9]
vsqrtsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0xc9]
vsqrtsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x51,0xbc,0x82,0x00,0x01,0x00,0x00]
vsqrtss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsqrtss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x51,0xbc,0x82,0x00,0xff,0xff,0xff]
vsqrtss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsqrtss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x51,0x7c,0x82,0x40]
vsqrtss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x51,0x7c,0x82,0xc0]
vsqrtss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x51,0x7c,0x82,0x40]
vsqrtss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x51,0x7c,0x82,0xc0]
vsqrtss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x51,0x8c,0x82,0x00,0x01,0x00,0x00]
vsqrtss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsqrtss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x51,0x8c,0x82,0x00,0xff,0xff,0xff]
vsqrtss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsqrtss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x4c,0x82,0x40]
vsqrtss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x4c,0x82,0xc0]
vsqrtss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x4c,0x82,0x40]
vsqrtss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x4c,0x82,0xc0]
vsqrtss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x51,0xbc,0x02,0x00,0x01,0x00,0x00]
vsqrtss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vsqrtss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x51,0x7c,0x02,0x40]
vsqrtss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x51,0x7c,0x02,0x40]
vsqrtss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x51,0x8c,0x02,0x00,0x01,0x00,0x00]
vsqrtss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vsqrtss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x4c,0x02,0x40]
vsqrtss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x4c,0x02,0x40]
vsqrtss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x51,0xba,0x00,0x01,0x00,0x00]
vsqrtss 256(%rdx), %xmm15, %xmm15

// CHECK: vsqrtss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x51,0x7a,0x40]
vsqrtss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x51,0x7a,0x40]
vsqrtss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x51,0x8a,0x00,0x01,0x00,0x00]
vsqrtss 256(%rdx), %xmm1, %xmm1

// CHECK: vsqrtss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x4a,0x40]
vsqrtss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x4a,0x40]
vsqrtss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm15, %xmm15

// CHECK: vsqrtss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x51,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm1, %xmm1

// CHECK: vsqrtss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x38,0x51,0xff]
vsqrtss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x3a,0x51,0xff]
vsqrtss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xba,0x51,0xff]
vsqrtss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x51,0xc9]
vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x51,0xc9]
vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x51,0xc9]
vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x51,0x3a]
vsqrtss (%rdx), %xmm15, %xmm15

// CHECK: vsqrtss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x51,0x3a]
vsqrtss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x51,0x3a]
vsqrtss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x51,0x0a]
vsqrtss (%rdx), %xmm1, %xmm1

// CHECK: vsqrtss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x0a]
vsqrtss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x0a]
vsqrtss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x51,0xff]
vsqrtss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x51,0xff]
vsqrtss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x51,0xff]
vsqrtss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x51,0xc9]
vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x51,0xc9]
vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x51,0xc9]
vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x58,0x51,0xff]
vsqrtss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x5a,0x51,0xff]
vsqrtss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xda,0x51,0xff]
vsqrtss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x51,0xc9]
vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x51,0xc9]
vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x51,0xc9]
vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x78,0x51,0xff]
vsqrtss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsqrtss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x7a,0x51,0xff]
vsqrtss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xfa,0x51,0xff]
vsqrtss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x51,0xc9]
vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x51,0xc9]
vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x51,0xc9]
vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x51,0xff]
vsqrtss %xmm15, %xmm15, %xmm15

// CHECK: vsqrtss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x51,0xff]
vsqrtss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsqrtss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x51,0xff]
vsqrtss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsqrtss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x51,0xc9]
vsqrtss %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0xc9]
vsqrtss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0xc9]
vsqrtss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm15, %xmm15

// CHECK: vsubsd 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vsubsd 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm1, %xmm1

// CHECK: vsubsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5c,0xbc,0x82,0x00,0x02,0x00,0x00]
vsubsd 512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsubsd -512(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5c,0xbc,0x82,0x00,0xfe,0xff,0xff]
vsubsd -512(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsubsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5c,0x7c,0x82,0x40]
vsubsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsubsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5c,0x7c,0x82,0xc0]
vsubsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsubsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5c,0x7c,0x82,0x40]
vsubsd 512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5c,0x7c,0x82,0xc0]
vsubsd -512(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd 512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5c,0x8c,0x82,0x00,0x02,0x00,0x00]
vsubsd 512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsubsd -512(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5c,0x8c,0x82,0x00,0xfe,0xff,0xff]
vsubsd -512(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsubsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x4c,0x82,0x40]
vsubsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x4c,0x82,0xc0]
vsubsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x4c,0x82,0x40]
vsubsd 512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x4c,0x82,0xc0]
vsubsd -512(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 512(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5c,0xbc,0x02,0x00,0x02,0x00,0x00]
vsubsd 512(%rdx,%rax), %xmm15, %xmm15

// CHECK: vsubsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5c,0x7c,0x02,0x40]
vsubsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vsubsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5c,0x7c,0x02,0x40]
vsubsd 512(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd 512(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5c,0x8c,0x02,0x00,0x02,0x00,0x00]
vsubsd 512(%rdx,%rax), %xmm1, %xmm1

// CHECK: vsubsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x4c,0x02,0x40]
vsubsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x4c,0x02,0x40]
vsubsd 512(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 512(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5c,0xba,0x00,0x02,0x00,0x00]
vsubsd 512(%rdx), %xmm15, %xmm15

// CHECK: vsubsd 512(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5c,0x7a,0x40]
vsubsd 512(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsubsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5c,0x7a,0x40]
vsubsd 512(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd 512(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5c,0x8a,0x00,0x02,0x00,0x00]
vsubsd 512(%rdx), %xmm1, %xmm1

// CHECK: vsubsd 512(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x4a,0x40]
vsubsd 512(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x4a,0x40]
vsubsd 512(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x38,0x5c,0xff]
vsubsd {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x3a,0x5c,0xff]
vsubsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xba,0x5c,0xff]
vsubsd {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x5c,0xc9]
vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x5c,0xc9]
vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x5c,0xc9]
vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x03,0x5c,0x3a]
vsubsd (%rdx), %xmm15, %xmm15

// CHECK: vsubsd (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x87,0x0a,0x5c,0x3a]
vsubsd (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsubsd (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x87,0x8a,0x5c,0x3a]
vsubsd (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5c,0x0a]
vsubsd (%rdx), %xmm1, %xmm1

// CHECK: vsubsd (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x0a]
vsubsd (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x0a]
vsubsd (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x18,0x5c,0xff]
vsubsd {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x1a,0x5c,0xff]
vsubsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x9a,0x5c,0xff]
vsubsd {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5c,0xc9]
vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5c,0xc9]
vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5c,0xc9]
vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x58,0x5c,0xff]
vsubsd {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x5a,0x5c,0xff]
vsubsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xda,0x5c,0xff]
vsubsd {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x5c,0xc9]
vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x5c,0xc9]
vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x5c,0xc9]
vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x87,0x78,0x5c,0xff]
vsubsd {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x7a,0x5c,0xff]
vsubsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0xfa,0x5c,0xff]
vsubsd {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x5c,0xc9]
vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x5c,0xc9]
vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x5c,0xc9]
vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x03,0x5c,0xff]
vsubsd %xmm15, %xmm15, %xmm15

// CHECK: vsubsd %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x87,0x0a,0x5c,0xff]
vsubsd %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubsd %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x87,0x8a,0x5c,0xff]
vsubsd %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf3,0x5c,0xc9]
vsubsd %xmm1, %xmm1, %xmm1

// CHECK: vsubsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0xc9]
vsubsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0xc9]
vsubsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5c,0xbc,0x82,0x00,0x01,0x00,0x00]
vsubss 256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsubss -256(%rdx,%rax,4), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5c,0xbc,0x82,0x00,0xff,0xff,0xff]
vsubss -256(%rdx,%rax,4), %xmm15, %xmm15

// CHECK: vsubss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5c,0x7c,0x82,0x40]
vsubss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsubss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5c,0x7c,0x82,0xc0]
vsubss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2}

// CHECK: vsubss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5c,0x7c,0x82,0x40]
vsubss 256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5c,0x7c,0x82,0xc0]
vsubss -256(%rdx,%rax,4), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss 256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5c,0x8c,0x82,0x00,0x01,0x00,0x00]
vsubss 256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsubss -256(%rdx,%rax,4), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5c,0x8c,0x82,0x00,0xff,0xff,0xff]
vsubss -256(%rdx,%rax,4), %xmm1, %xmm1

// CHECK: vsubss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x4c,0x82,0x40]
vsubss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x4c,0x82,0xc0]
vsubss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x4c,0x82,0x40]
vsubss 256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x4c,0x82,0xc0]
vsubss -256(%rdx,%rax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 256(%rdx,%rax), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5c,0xbc,0x02,0x00,0x01,0x00,0x00]
vsubss 256(%rdx,%rax), %xmm15, %xmm15

// CHECK: vsubss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5c,0x7c,0x02,0x40]
vsubss 256(%rdx,%rax), %xmm15, %xmm15 {%k2}

// CHECK: vsubss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5c,0x7c,0x02,0x40]
vsubss 256(%rdx,%rax), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss 256(%rdx,%rax), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5c,0x8c,0x02,0x00,0x01,0x00,0x00]
vsubss 256(%rdx,%rax), %xmm1, %xmm1

// CHECK: vsubss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x4c,0x02,0x40]
vsubss 256(%rdx,%rax), %xmm1, %xmm1 {%k2}

// CHECK: vsubss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x4c,0x02,0x40]
vsubss 256(%rdx,%rax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 256(%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5c,0xba,0x00,0x01,0x00,0x00]
vsubss 256(%rdx), %xmm15, %xmm15

// CHECK: vsubss 256(%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5c,0x7a,0x40]
vsubss 256(%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsubss 256(%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5c,0x7a,0x40]
vsubss 256(%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss 256(%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5c,0x8a,0x00,0x01,0x00,0x00]
vsubss 256(%rdx), %xmm1, %xmm1

// CHECK: vsubss 256(%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x4a,0x40]
vsubss 256(%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsubss 256(%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x4a,0x40]
vsubss 256(%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 485498096, %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm15, %xmm15

// CHECK: vsubss 485498096, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm15, %xmm15 {%k2}

// CHECK: vsubss 485498096, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm1, %xmm1

// CHECK: vsubss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsubss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {rd-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x38,0x5c,0xff]
vsubss {rd-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x3a,0x5c,0xff]
vsubss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xba,0x5c,0xff]
vsubss {rd-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x5c,0xc9]
vsubss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x5c,0xc9]
vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x5c,0xc9]
vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss (%rdx), %xmm15, %xmm15
// CHECK: encoding: [0xc5,0x02,0x5c,0x3a]
vsubss (%rdx), %xmm15, %xmm15

// CHECK: vsubss (%rdx), %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x71,0x06,0x0a,0x5c,0x3a]
vsubss (%rdx), %xmm15, %xmm15 {%k2}

// CHECK: vsubss (%rdx), %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x71,0x06,0x8a,0x5c,0x3a]
vsubss (%rdx), %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss (%rdx), %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5c,0x0a]
vsubss (%rdx), %xmm1, %xmm1

// CHECK: vsubss (%rdx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x0a]
vsubss (%rdx), %xmm1, %xmm1 {%k2}

// CHECK: vsubss (%rdx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x0a]
vsubss (%rdx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {rn-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x18,0x5c,0xff]
vsubss {rn-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x1a,0x5c,0xff]
vsubss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x9a,0x5c,0xff]
vsubss {rn-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5c,0xc9]
vsubss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5c,0xc9]
vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5c,0xc9]
vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {ru-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x58,0x5c,0xff]
vsubss {ru-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x5a,0x5c,0xff]
vsubss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xda,0x5c,0xff]
vsubss {ru-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x5c,0xc9]
vsubss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x5c,0xc9]
vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x5c,0xc9]
vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {rz-sae}, %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x06,0x78,0x5c,0xff]
vsubss {rz-sae}, %xmm15, %xmm15, %xmm15

// CHECK: vsubss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x7a,0x5c,0xff]
vsubss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0xfa,0x5c,0xff]
vsubss {rz-sae}, %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x5c,0xc9]
vsubss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x5c,0xc9]
vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x5c,0xc9]
vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss %xmm15, %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x02,0x5c,0xff]
vsubss %xmm15, %xmm15, %xmm15

// CHECK: vsubss %xmm15, %xmm15, %xmm15 {%k2}
// CHECK: encoding: [0x62,0x51,0x06,0x0a,0x5c,0xff]
vsubss %xmm15, %xmm15, %xmm15 {%k2}

// CHECK: vsubss %xmm15, %xmm15, %xmm15 {%k2} {z}
// CHECK: encoding: [0x62,0x51,0x06,0x8a,0x5c,0xff]
vsubss %xmm15, %xmm15, %xmm15 {%k2} {z}

// CHECK: vsubss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf2,0x5c,0xc9]
vsubss %xmm1, %xmm1, %xmm1

// CHECK: vsubss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0xc9]
vsubss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0xc9]
vsubss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vucomisd 485498096, %xmm15
// CHECK: encoding: [0xc5,0x79,0x2e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vucomisd 485498096, %xmm15

// CHECK: vucomisd 485498096, %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vucomisd 485498096, %xmm1

// CHECK: vucomisd 512(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2e,0xbc,0x82,0x00,0x02,0x00,0x00]
vucomisd 512(%rdx,%rax,4), %xmm15

// CHECK: vucomisd -512(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2e,0xbc,0x82,0x00,0xfe,0xff,0xff]
vucomisd -512(%rdx,%rax,4), %xmm15

// CHECK: vucomisd 512(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2e,0x8c,0x82,0x00,0x02,0x00,0x00]
vucomisd 512(%rdx,%rax,4), %xmm1

// CHECK: vucomisd -512(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2e,0x8c,0x82,0x00,0xfe,0xff,0xff]
vucomisd -512(%rdx,%rax,4), %xmm1

// CHECK: vucomisd 512(%rdx,%rax), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2e,0xbc,0x02,0x00,0x02,0x00,0x00]
vucomisd 512(%rdx,%rax), %xmm15

// CHECK: vucomisd 512(%rdx,%rax), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2e,0x8c,0x02,0x00,0x02,0x00,0x00]
vucomisd 512(%rdx,%rax), %xmm1

// CHECK: vucomisd 512(%rdx), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2e,0xba,0x00,0x02,0x00,0x00]
vucomisd 512(%rdx), %xmm15

// CHECK: vucomisd 512(%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2e,0x8a,0x00,0x02,0x00,0x00]
vucomisd 512(%rdx), %xmm1

// CHECK: vucomisd (%rdx), %xmm15
// CHECK: encoding: [0xc5,0x79,0x2e,0x3a]
vucomisd (%rdx), %xmm15

// CHECK: vucomisd (%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2e,0x0a]
vucomisd (%rdx), %xmm1

// CHECK: vucomisd {sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0xfd,0x18,0x2e,0xff]
vucomisd {sae}, %xmm15, %xmm15

// CHECK: vucomisd {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x18,0x2e,0xc9]
vucomisd {sae}, %xmm1, %xmm1

// CHECK: vucomisd %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x79,0x2e,0xff]
vucomisd %xmm15, %xmm15

// CHECK: vucomisd %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf9,0x2e,0xc9]
vucomisd %xmm1, %xmm1

// CHECK: vucomiss 256(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2e,0xbc,0x82,0x00,0x01,0x00,0x00]
vucomiss 256(%rdx,%rax,4), %xmm15

// CHECK: vucomiss -256(%rdx,%rax,4), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2e,0xbc,0x82,0x00,0xff,0xff,0xff]
vucomiss -256(%rdx,%rax,4), %xmm15

// CHECK: vucomiss 256(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2e,0x8c,0x82,0x00,0x01,0x00,0x00]
vucomiss 256(%rdx,%rax,4), %xmm1

// CHECK: vucomiss -256(%rdx,%rax,4), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2e,0x8c,0x82,0x00,0xff,0xff,0xff]
vucomiss -256(%rdx,%rax,4), %xmm1

// CHECK: vucomiss 256(%rdx,%rax), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2e,0xbc,0x02,0x00,0x01,0x00,0x00]
vucomiss 256(%rdx,%rax), %xmm15

// CHECK: vucomiss 256(%rdx,%rax), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2e,0x8c,0x02,0x00,0x01,0x00,0x00]
vucomiss 256(%rdx,%rax), %xmm1

// CHECK: vucomiss 256(%rdx), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2e,0xba,0x00,0x01,0x00,0x00]
vucomiss 256(%rdx), %xmm15

// CHECK: vucomiss 256(%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2e,0x8a,0x00,0x01,0x00,0x00]
vucomiss 256(%rdx), %xmm1

// CHECK: vucomiss 485498096, %xmm15
// CHECK: encoding: [0xc5,0x78,0x2e,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
vucomiss 485498096, %xmm15

// CHECK: vucomiss 485498096, %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2e,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
vucomiss 485498096, %xmm1

// CHECK: vucomiss (%rdx), %xmm15
// CHECK: encoding: [0xc5,0x78,0x2e,0x3a]
vucomiss (%rdx), %xmm15

// CHECK: vucomiss (%rdx), %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2e,0x0a]
vucomiss (%rdx), %xmm1

// CHECK: vucomiss {sae}, %xmm15, %xmm15
// CHECK: encoding: [0x62,0x51,0x7c,0x18,0x2e,0xff]
vucomiss {sae}, %xmm15, %xmm15

// CHECK: vucomiss {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x18,0x2e,0xc9]
vucomiss {sae}, %xmm1, %xmm1

// CHECK: vucomiss %xmm15, %xmm15
// CHECK: encoding: [0xc4,0x41,0x78,0x2e,0xff]
vucomiss %xmm15, %xmm15

// CHECK: vucomiss %xmm1, %xmm1
// CHECK: encoding: [0xc5,0xf8,0x2e,0xc9]
vucomiss %xmm1, %xmm1

