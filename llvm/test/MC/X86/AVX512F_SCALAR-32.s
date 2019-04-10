// RUN: llvm-mc -triple i386-unknown-unknown -mcpu=skx --show-encoding %s | FileCheck %s

// FIXME some of these tests use VEX encodings because we have no way to force
// an EVEX encoding. gas has an {evex} prefix that can force the mode, but
// we don't have that.

// CHECK: vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vaddsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vaddsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vaddsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vaddsd 485498096, %xmm1, %xmm1

// CHECK: vaddsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]
vaddsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x58,0x4c,0x02,0x40
{evex} vaddsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vaddsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x4c,0x02,0x40]
vaddsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x4c,0x02,0x40]
vaddsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x58,0x0a]
{evex} vaddsd (%edx), %xmm1, %xmm1

// CHECK: vaddsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0x0a]
vaddsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vaddsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0x0a]
vaddsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x58,0xc9]
vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x58,0xc9]
vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x58,0xc9]
vaddsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x58,0xc9]
vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x58,0xc9]
vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x58,0xc9]
vaddsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x58,0xc9]
vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x58,0xc9]
vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x58,0xc9]
vaddsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x58,0xc9]
vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x58,0xc9]
vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x58,0xc9]
vaddsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x58,0xc9]
{evex} vaddsd %xmm1, %xmm1, %xmm1

// CHECK: vaddsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x58,0xc9]
vaddsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x58,0xc9]
vaddsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x58,0x4c,0x02,0x40]
{evex} vaddss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vaddss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x4c,0x02,0x40]
vaddss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vaddss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x4c,0x02,0x40]
vaddss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vaddss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vaddss 485498096(%edx), %xmm1, %xmm1

// CHECK: vaddss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vaddss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x8a,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vaddss 485498096, %xmm1, %xmm1

// CHECK: vaddss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vaddss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x0d,0xf0,0x1c,0xf0,0x1c]
vaddss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x58,0x0a]
{evex} vaddss (%edx), %xmm1, %xmm1

// CHECK: vaddss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0x0a]
vaddss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vaddss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0x0a]
vaddss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x58,0xc9]
vaddss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x58,0xc9]
vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x58,0xc9]
vaddss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x58,0xc9]
vaddss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x58,0xc9]
vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x58,0xc9]
vaddss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x58,0xc9]
vaddss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x58,0xc9]
vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x58,0xc9]
vaddss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x58,0xc9]
vaddss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x58,0xc9]
vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x58,0xc9]
vaddss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vaddss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x58,0xc9]
{evex} vaddss %xmm1, %xmm1, %xmm1

// CHECK: vaddss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x58,0xc9]
vaddss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vaddss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x58,0xc9]
vaddss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcmpeqsd -485498096(%edx,%eax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vcmpeqsd -485498096(%edx,%eax,4), %xmm1, %k2

// CHECK: vcmpeqsd 485498096(%edx,%eax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096(%edx,%eax,4), %xmm1, %k2

// CHECK: vcmpeqsd -485498096(%edx,%eax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vcmpeqsd -485498096(%edx,%eax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd 485498096(%edx,%eax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096(%edx,%eax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd 485498096(%edx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x92,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096(%edx), %xmm1, %k2

// CHECK: vcmpeqsd 485498096(%edx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x92,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096(%edx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd 485498096, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x15,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096, %xmm1, %k2

// CHECK: vcmpeqsd 485498096, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x15,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqsd 485498096, %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd 512(%edx,%eax), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x54,0x02,0x40,0x00]
vcmpeqsd 512(%edx,%eax), %xmm1, %k2

// CHECK: vcmpeqsd 512(%edx,%eax), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x54,0x02,0x40,0x00]
vcmpeqsd 512(%edx,%eax), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd (%edx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0x12,0x00]
vcmpeqsd (%edx), %xmm1, %k2

// CHECK: vcmpeqsd (%edx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0x12,0x00]
vcmpeqsd (%edx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd {sae}, %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0xc2,0xd1,0x00]
vcmpeqsd {sae}, %xmm1, %xmm1, %k2

// CHECK: vcmpeqsd {sae}, %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0xc2,0xd1,0x00]
vcmpeqsd {sae}, %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcmpeqsd %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0xc2,0xd1,0x00]
vcmpeqsd %xmm1, %xmm1, %k2

// CHECK: vcmpeqsd %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0xc2,0xd1,0x00]
vcmpeqsd %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 256(%edx,%eax), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x54,0x02,0x40,0x00]
vcmpeqss 256(%edx,%eax), %xmm1, %k2

// CHECK: vcmpeqss 256(%edx,%eax), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x54,0x02,0x40,0x00]
vcmpeqss 256(%edx,%eax), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss -485498096(%edx,%eax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vcmpeqss -485498096(%edx,%eax,4), %xmm1, %k2

// CHECK: vcmpeqss 485498096(%edx,%eax,4), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096(%edx,%eax,4), %xmm1, %k2

// CHECK: vcmpeqss -485498096(%edx,%eax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x94,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vcmpeqss -485498096(%edx,%eax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 485498096(%edx,%eax,4), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x94,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096(%edx,%eax,4), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 485498096(%edx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x92,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096(%edx), %xmm1, %k2

// CHECK: vcmpeqss 485498096(%edx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x92,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096(%edx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss 485498096, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x15,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096, %xmm1, %k2

// CHECK: vcmpeqss 485498096, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x15,0xf0,0x1c,0xf0,0x1c,0x00]
vcmpeqss 485498096, %xmm1, %k2 {%k2}

// CHECK: vcmpeqss (%edx), %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0x12,0x00]
vcmpeqss (%edx), %xmm1, %k2

// CHECK: vcmpeqss (%edx), %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0x12,0x00]
vcmpeqss (%edx), %xmm1, %k2 {%k2}

// CHECK: vcmpeqss {sae}, %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0xc2,0xd1,0x00]
vcmpeqss {sae}, %xmm1, %xmm1, %k2

// CHECK: vcmpeqss {sae}, %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0xc2,0xd1,0x00]
vcmpeqss {sae}, %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcmpeqss %xmm1, %xmm1, %k2
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0xc2,0xd1,0x00]
vcmpeqss %xmm1, %xmm1, %k2

// CHECK: vcmpeqss %xmm1, %xmm1, %k2 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0xc2,0xd1,0x00]
vcmpeqss %xmm1, %xmm1, %k2 {%k2}

// CHECK: vcomisd -485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vcomisd -485498096(%edx,%eax,4), %xmm1

// CHECK: vcomisd 485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vcomisd 485498096(%edx,%eax,4), %xmm1

// CHECK: vcomisd 485498096(%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2f,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vcomisd 485498096(%edx), %xmm1

// CHECK: vcomisd 485498096, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2f,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vcomisd 485498096, %xmm1

// CHECK: vcomisd 512(%edx,%eax), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2f,0x4c,0x02,0x40]
{evex} vcomisd 512(%edx,%eax), %xmm1

// CHECK: vcomisd (%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2f,0x0a]
{evex} vcomisd (%edx), %xmm1

// CHECK: vcomisd {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x18,0x2f,0xc9]
vcomisd {sae}, %xmm1, %xmm1

// CHECK: vcomisd %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2f,0xc9]
{evex} vcomisd %xmm1, %xmm1

// CHECK: vcomiss 256(%edx,%eax), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2f,0x4c,0x02,0x40]
{evex} vcomiss 256(%edx,%eax), %xmm1

// CHECK: vcomiss -485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vcomiss -485498096(%edx,%eax,4), %xmm1

// CHECK: vcomiss 485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vcomiss 485498096(%edx,%eax,4), %xmm1

// CHECK: vcomiss 485498096(%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2f,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vcomiss 485498096(%edx), %xmm1

// CHECK: vcomiss 485498096, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2f,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vcomiss 485498096, %xmm1

// CHECK: vcomiss (%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2f,0x0a]
{evex} vcomiss (%edx), %xmm1

// CHECK: vcomiss {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x18,0x2f,0xc9]
vcomiss {sae}, %xmm1, %xmm1

// CHECK: vcomiss %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2f,0xc9]
{evex} vcomiss %xmm1, %xmm1

// CHECK: vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3
{evex} vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vcvtsd2ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsd2ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vcvtsd2ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsd2ss 485498096, %xmm1, %xmm1

// CHECK: vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]
vcvtsd2ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5a,0x4c,0x02,0x40]
{evex} vcvtsd2ss 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vcvtsd2ss 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x4c,0x02,0x40]
vcvtsd2ss 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x4c,0x02,0x40]
vcvtsd2ss 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5a,0x0a]
{evex} vcvtsd2ss (%edx), %xmm1, %xmm1

// CHECK: vcvtsd2ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0x0a]
vcvtsd2ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0x0a]
vcvtsd2ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x5a,0xc9]
vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x5a,0xc9]
vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x5a,0xc9]
vcvtsd2ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5a,0xc9]
vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5a,0xc9]
vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5a,0xc9]
vcvtsd2ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x5a,0xc9]
vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x5a,0xc9]
vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x5a,0xc9]
vcvtsd2ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x5a,0xc9]
vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x5a,0xc9]
vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x5a,0xc9]
vcvtsd2ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsd2ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5a,0xc9]
{evex} vcvtsd2ss %xmm1, %xmm1, %xmm1

// CHECK: vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5a,0xc9]
vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5a,0xc9]
vcvtsd2ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtsi2sdl 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x2a,0x4c,0x02,0x40]
{evex} vcvtsi2sdl 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vcvtsi2sdl -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x2a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vcvtsi2sdl -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtsi2sdl 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x2a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsi2sdl 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtsi2sdl 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x2a,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsi2sdl 485498096(%edx), %xmm1, %xmm1

// CHECK: vcvtsi2sdl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x2a,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsi2sdl 485498096, %xmm1, %xmm1

// CHECK: vcvtsi2sdl (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x2a,0x0a]
{evex} vcvtsi2sdl (%edx), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x2a,0x4c,0x02,0x40]
{evex} vcvtsi2ssl 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vcvtsi2ssl -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x2a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vcvtsi2ssl -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x2a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsi2ssl 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x2a,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsi2ssl 485498096(%edx), %xmm1, %xmm1

// CHECK: vcvtsi2ssl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x2a,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtsi2ssl 485498096, %xmm1, %xmm1

// CHECK: vcvtsi2ssl (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x2a,0x0a]
{evex} vcvtsi2ssl (%edx), %xmm1, %xmm1

// CHECK: vcvtss2sd 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5a,0x4c,0x02,0x40]
{evex} vcvtss2sd 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vcvtss2sd 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x4c,0x02,0x40]
vcvtss2sd 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x4c,0x02,0x40]
vcvtss2sd 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vcvtss2sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtss2sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vcvtss2sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x8a,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vcvtss2sd 485498096, %xmm1, %xmm1

// CHECK: vcvtss2sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x0d,0xf0,0x1c,0xf0,0x1c]
vcvtss2sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5a,0x0a]
{evex} vcvtss2sd (%edx), %xmm1, %xmm1

// CHECK: vcvtss2sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0x0a]
vcvtss2sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0x0a]
vcvtss2sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5a,0xc9]
vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5a,0xc9]
vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5a,0xc9]
vcvtss2sd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtss2sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5a,0xc9]
{evex} vcvtss2sd %xmm1, %xmm1, %xmm1

// CHECK: vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5a,0xc9]
vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5a,0xc9]
vcvtss2sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vcvtusi2sdl 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x4c,0x02,0x40]
vcvtusi2sdl 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vcvtusi2sdl -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vcvtusi2sdl -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtusi2sdl 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vcvtusi2sdl 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtusi2sdl 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x8a,0xf0,0x1c,0xf0,0x1c]
vcvtusi2sdl 485498096(%edx), %xmm1, %xmm1

// CHECK: vcvtusi2sdl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x0d,0xf0,0x1c,0xf0,0x1c]
vcvtusi2sdl 485498096, %xmm1, %xmm1

// CHECK: vcvtusi2sdl (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x77,0x08,0x7b,0x0a]
vcvtusi2sdl (%edx), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x4c,0x02,0x40]
vcvtusi2ssl 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vcvtusi2ssl -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vcvtusi2ssl -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vcvtusi2ssl 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x8a,0xf0,0x1c,0xf0,0x1c]
vcvtusi2ssl 485498096(%edx), %xmm1, %xmm1

// CHECK: vcvtusi2ssl 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x0d,0xf0,0x1c,0xf0,0x1c]
vcvtusi2ssl 485498096, %xmm1, %xmm1

// CHECK: vcvtusi2ssl (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x7b,0x0a]
vcvtusi2ssl (%edx), %xmm1, %xmm1

// CHECK: vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vdivsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vdivsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vdivsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vdivsd 485498096, %xmm1, %xmm1

// CHECK: vdivsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]
vdivsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5e,0x4c,0x02,0x40]
{evex} vdivsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vdivsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x4c,0x02,0x40]
vdivsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x4c,0x02,0x40]
vdivsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5e,0x0a]
{evex} vdivsd (%edx), %xmm1, %xmm1

// CHECK: vdivsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0x0a]
vdivsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vdivsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0x0a]
vdivsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x5e,0xc9]
vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x5e,0xc9]
vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x5e,0xc9]
vdivsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5e,0xc9]
vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5e,0xc9]
vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5e,0xc9]
vdivsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x5e,0xc9]
vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x5e,0xc9]
vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x5e,0xc9]
vdivsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x5e,0xc9]
vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x5e,0xc9]
vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x5e,0xc9]
vdivsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5e,0xc9]
{evex} vdivsd %xmm1, %xmm1, %xmm1

// CHECK: vdivsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5e,0xc9]
vdivsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5e,0xc9]
vdivsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5e,0x4c,0x02,0x40]
{evex} vdivss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vdivss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x4c,0x02,0x40]
vdivss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vdivss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x4c,0x02,0x40]
vdivss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vdivss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vdivss 485498096(%edx), %xmm1, %xmm1

// CHECK: vdivss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vdivss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x8a,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vdivss 485498096, %xmm1, %xmm1

// CHECK: vdivss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vdivss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x0d,0xf0,0x1c,0xf0,0x1c]
vdivss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5e,0x0a]
{evex} vdivss (%edx), %xmm1, %xmm1

// CHECK: vdivss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0x0a]
vdivss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vdivss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0x0a]
vdivss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x5e,0xc9]
vdivss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x5e,0xc9]
vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x5e,0xc9]
vdivss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5e,0xc9]
vdivss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5e,0xc9]
vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5e,0xc9]
vdivss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x5e,0xc9]
vdivss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x5e,0xc9]
vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x5e,0xc9]
vdivss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x5e,0xc9]
vdivss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x5e,0xc9]
vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x5e,0xc9]
vdivss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vdivss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5e,0xc9]
{evex} vdivss %xmm1, %xmm1, %xmm1

// CHECK: vdivss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5e,0xc9]
vdivss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vdivss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5e,0xc9]
vdivss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vfixupimmsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vfixupimmsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vfixupimmsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096(%edx), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x4c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0x0a,0x00]
vfixupimmsd $0, (%edx), %xmm1, %xmm1

// CHECK: vfixupimmsd $0, (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0x0a,0x00]
vfixupimmsd $0, (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0x0a,0x00]
vfixupimmsd $0, (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x18,0x55,0xc9,0x00]
vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x1a,0x55,0xc9,0x00]
vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x9a,0x55,0xc9,0x00]
vfixupimmsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmsd $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x55,0xc9,0x00]
vfixupimmsd $0, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x55,0xc9,0x00]
vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x55,0xc9,0x00]
vfixupimmsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x4c,0x02,0x40,0x00]
vfixupimmss $0, 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfixupimmss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x4c,0x02,0x40,0x00]
vfixupimmss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vfixupimmss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfixupimmss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfixupimmss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vfixupimmss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vfixupimmss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096(%edx), %xmm1, %xmm1

// CHECK: vfixupimmss $0, 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm1, %xmm1

// CHECK: vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vfixupimmss $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0x0a,0x00]
vfixupimmss $0, (%edx), %xmm1, %xmm1

// CHECK: vfixupimmss $0, (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0x0a,0x00]
vfixupimmss $0, (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0x0a,0x00]
vfixupimmss $0, (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x18,0x55,0xc9,0x00]
vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x1a,0x55,0xc9,0x00]
vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x9a,0x55,0xc9,0x00]
vfixupimmss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfixupimmss $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x55,0xc9,0x00]
vfixupimmss $0, %xmm1, %xmm1, %xmm1

// CHECK: vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x55,0xc9,0x00]
vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x55,0xc9,0x00]
vfixupimmss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd132sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd132sd 485498096, %xmm1, %xmm1

// CHECK: vfmadd132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x99,0x4c,0x02,0x40]
{evex} vfmadd132sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x4c,0x02,0x40]
vfmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x4c,0x02,0x40]
vfmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x99,0x0a]
{evex} vfmadd132sd (%edx), %xmm1, %xmm1

// CHECK: vfmadd132sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0x0a]
vfmadd132sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0x0a]
vfmadd132sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x99,0xc9]
vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x99,0xc9]
vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x99,0xc9]
vfmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x99,0xc9]
vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x99,0xc9]
vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x99,0xc9]
vfmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x99,0xc9]
vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x99,0xc9]
vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x99,0xc9]
vfmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x99,0xc9]
vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x99,0xc9]
vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x99,0xc9]
vfmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x99,0xc9]
{evex} vfmadd132sd %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x99,0xc9]
vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x99,0xc9]
vfmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x99,0x4c,0x02,0x40]
{evex} vfmadd132ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x4c,0x02,0x40]
vfmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x4c,0x02,0x40]
vfmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd132ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd132ss 485498096, %xmm1, %xmm1

// CHECK: vfmadd132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x99,0x0a]
{evex} vfmadd132ss (%edx), %xmm1, %xmm1

// CHECK: vfmadd132ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0x0a]
vfmadd132ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0x0a]
vfmadd132ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x99,0xc9]
vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x99,0xc9]
vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x99,0xc9]
vfmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x99,0xc9]
vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x99,0xc9]
vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x99,0xc9]
vfmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x99,0xc9]
vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x99,0xc9]
vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x99,0xc9]
vfmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x99,0xc9]
vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x99,0xc9]
vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x99,0xc9]
vfmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x99,0xc9]
{evex} vfmadd132ss %xmm1, %xmm1, %xmm1

// CHECK: vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x99,0xc9]
vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x99,0xc9]
vfmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd213sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd213sd 485498096, %xmm1, %xmm1

// CHECK: vfmadd213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xa9,0x4c,0x02,0x40]
{evex} vfmadd213sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x4c,0x02,0x40]
vfmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x4c,0x02,0x40]
vfmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xa9,0x0a]
{evex} vfmadd213sd (%edx), %xmm1, %xmm1

// CHECK: vfmadd213sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0x0a]
vfmadd213sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0x0a]
vfmadd213sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xa9,0xc9]
vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xa9,0xc9]
vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xa9,0xc9]
vfmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xa9,0xc9]
vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xa9,0xc9]
vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xa9,0xc9]
vfmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xa9,0xc9]
vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xa9,0xc9]
vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xa9,0xc9]
vfmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xa9,0xc9]
vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xa9,0xc9]
vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xa9,0xc9]
vfmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xa9,0xc9]
{evex} vfmadd213sd %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xa9,0xc9]
vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xa9,0xc9]
vfmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xa9,0x4c,0x02,0x40]
{evex} vfmadd213ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x4c,0x02,0x40]
vfmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x4c,0x02,0x40]
vfmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd213ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd213ss 485498096, %xmm1, %xmm1

// CHECK: vfmadd213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xa9,0x0a]
{evex} vfmadd213ss (%edx), %xmm1, %xmm1

// CHECK: vfmadd213ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0x0a]
vfmadd213ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0x0a]
vfmadd213ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xa9,0xc9]
vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xa9,0xc9]
vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xa9,0xc9]
vfmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xa9,0xc9]
vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xa9,0xc9]
vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xa9,0xc9]
vfmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xa9,0xc9]
vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xa9,0xc9]
vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xa9,0xc9]
vfmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xa9,0xc9]
vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xa9,0xc9]
vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xa9,0xc9]
vfmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xa9,0xc9]
{evex} vfmadd213ss %xmm1, %xmm1, %xmm1

// CHECK: vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xa9,0xc9]
vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xa9,0xc9]
vfmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd231sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd231sd 485498096, %xmm1, %xmm1

// CHECK: vfmadd231sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xb9,0x4c,0x02,0x40]
{evex} vfmadd231sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x4c,0x02,0x40]
vfmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x4c,0x02,0x40]
vfmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xb9,0x0a]
{evex} vfmadd231sd (%edx), %xmm1, %xmm1

// CHECK: vfmadd231sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0x0a]
vfmadd231sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0x0a]
vfmadd231sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xb9,0xc9]
vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xb9,0xc9]
vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xb9,0xc9]
vfmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xb9,0xc9]
vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xb9,0xc9]
vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xb9,0xc9]
vfmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xb9,0xc9]
vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xb9,0xc9]
vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xb9,0xc9]
vfmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xb9,0xc9]
vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xb9,0xc9]
vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xb9,0xc9]
vfmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xb9,0xc9]
{evex} vfmadd231sd %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xb9,0xc9]
vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb9,0xc9]
vfmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xb9,0x4c,0x02,0x40]
{evex} vfmadd231ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x4c,0x02,0x40]
vfmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x4c,0x02,0x40]
vfmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd231ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmadd231ss 485498096, %xmm1, %xmm1

// CHECK: vfmadd231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xb9,0x0a]
{evex} vfmadd231ss (%edx), %xmm1, %xmm1

// CHECK: vfmadd231ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0x0a]
vfmadd231ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0x0a]
vfmadd231ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xb9,0xc9]
vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xb9,0xc9]
vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xb9,0xc9]
vfmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xb9,0xc9]
vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xb9,0xc9]
vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xb9,0xc9]
vfmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xb9,0xc9]
vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xb9,0xc9]
vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xb9,0xc9]
vfmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xb9,0xc9]
vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xb9,0xc9]
vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xb9,0xc9]
vfmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmadd231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xb9,0xc9]
{evex} vfmadd231ss %xmm1, %xmm1, %xmm1

// CHECK: vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xb9,0xc9]
vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xb9,0xc9]
vfmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub132sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub132sd 485498096, %xmm1, %xmm1

// CHECK: vfmsub132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9b,0x4c,0x02,0x40]
{evex} vfmsub132sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x4c,0x02,0x40]
vfmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x4c,0x02,0x40]
vfmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9b,0x0a]
{evex} vfmsub132sd (%edx), %xmm1, %xmm1

// CHECK: vfmsub132sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0x0a]
vfmsub132sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0x0a]
vfmsub132sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x9b,0xc9]
vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x9b,0xc9]
vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x9b,0xc9]
vfmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x9b,0xc9]
vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x9b,0xc9]
vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x9b,0xc9]
vfmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x9b,0xc9]
vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x9b,0xc9]
vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x9b,0xc9]
vfmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x9b,0xc9]
vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x9b,0xc9]
vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x9b,0xc9]
vfmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9b,0xc9]
{evex} vfmsub132sd %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9b,0xc9]
vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9b,0xc9]
vfmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9b,0x4c,0x02,0x40]
{evex} vfmsub132ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x4c,0x02,0x40]
vfmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x4c,0x02,0x40]
vfmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub132ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub132ss 485498096, %xmm1, %xmm1

// CHECK: vfmsub132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9b,0x0a]
{evex} vfmsub132ss (%edx), %xmm1, %xmm1

// CHECK: vfmsub132ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0x0a]
vfmsub132ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0x0a]
vfmsub132ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x9b,0xc9]
vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x9b,0xc9]
vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x9b,0xc9]
vfmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x9b,0xc9]
vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x9b,0xc9]
vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x9b,0xc9]
vfmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x9b,0xc9]
vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x9b,0xc9]
vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x9b,0xc9]
vfmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x9b,0xc9]
vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x9b,0xc9]
vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x9b,0xc9]
vfmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9b,0xc9]
{evex} vfmsub132ss %xmm1, %xmm1, %xmm1

// CHECK: vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9b,0xc9]
vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9b,0xc9]
vfmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub213sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub213sd 485498096, %xmm1, %xmm1

// CHECK: vfmsub213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xab,0x4c,0x02,0x40]
{evex} vfmsub213sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x4c,0x02,0x40]
vfmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x4c,0x02,0x40]
vfmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xab,0x0a]
{evex} vfmsub213sd (%edx), %xmm1, %xmm1

// CHECK: vfmsub213sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0x0a]
vfmsub213sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0x0a]
vfmsub213sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xab,0xc9]
vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xab,0xc9]
vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xab,0xc9]
vfmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xab,0xc9]
vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xab,0xc9]
vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xab,0xc9]
vfmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xab,0xc9]
vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xab,0xc9]
vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xab,0xc9]
vfmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xab,0xc9]
vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xab,0xc9]
vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xab,0xc9]
vfmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xab,0xc9]
{evex} vfmsub213sd %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xab,0xc9]
vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xab,0xc9]
vfmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xab,0x4c,0x02,0x40]
{evex} vfmsub213ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x4c,0x02,0x40]
vfmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x4c,0x02,0x40]
vfmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub213ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub213ss 485498096, %xmm1, %xmm1

// CHECK: vfmsub213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xab,0x0a]
{evex} vfmsub213ss (%edx), %xmm1, %xmm1

// CHECK: vfmsub213ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0x0a]
vfmsub213ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0x0a]
vfmsub213ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xab,0xc9]
vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xab,0xc9]
vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xab,0xc9]
vfmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xab,0xc9]
vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xab,0xc9]
vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xab,0xc9]
vfmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xab,0xc9]
vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xab,0xc9]
vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xab,0xc9]
vfmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xab,0xc9]
vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xab,0xc9]
vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xab,0xc9]
vfmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xab,0xc9]
{evex} vfmsub213ss %xmm1, %xmm1, %xmm1

// CHECK: vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xab,0xc9]
vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xab,0xc9]
vfmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub231sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmsub231sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub231sd 485498096, %xmm1, %xmm1

// CHECK: vfmsub231sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub231sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbb,0x4c,0x02,0x40]
{evex} vfmsub231sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmsub231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x4c,0x02,0x40]
vfmsub231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x4c,0x02,0x40]
vfmsub231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbb,0x0a]
{evex} vfmsub231sd (%edx), %xmm1, %xmm1

// CHECK: vfmsub231sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0x0a]
vfmsub231sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0x0a]
vfmsub231sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xbb,0xc9]
vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xbb,0xc9]
vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xbb,0xc9]
vfmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xbb,0xc9]
vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xbb,0xc9]
vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xbb,0xc9]
vfmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xbb,0xc9]
vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xbb,0xc9]
vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xbb,0xc9]
vfmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xbb,0xc9]
vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xbb,0xc9]
vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xbb,0xc9]
vfmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbb,0xc9]
{evex} vfmsub231sd %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbb,0xc9]
vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbb,0xc9]
vfmsub231sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbb,0x4c,0x02,0x40]
{evex} vfmsub231ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x4c,0x02,0x40]
vfmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x4c,0x02,0x40]
vfmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub231ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x8a,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfmsub231ss 485498096, %xmm1, %xmm1

// CHECK: vfmsub231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x0d,0xf0,0x1c,0xf0,0x1c]
vfmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbb,0x0a]
{evex} vfmsub231ss (%edx), %xmm1, %xmm1

// CHECK: vfmsub231ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0x0a]
vfmsub231ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0x0a]
vfmsub231ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xbb,0xc9]
vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xbb,0xc9]
vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xbb,0xc9]
vfmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xbb,0xc9]
vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xbb,0xc9]
vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xbb,0xc9]
vfmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xbb,0xc9]
vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xbb,0xc9]
vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xbb,0xc9]
vfmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xbb,0xc9]
vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xbb,0xc9]
vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xbb,0xc9]
vfmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfmsub231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbb,0xc9]
{evex} vfmsub231ss %xmm1, %xmm1, %xmm1

// CHECK: vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbb,0xc9]
vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbb,0xc9]
vfmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd132sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd132sd 485498096, %xmm1, %xmm1

// CHECK: vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9d,0x4c,0x02,0x40]
{evex} vfnmadd132sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x4c,0x02,0x40]
vfnmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x4c,0x02,0x40]
vfnmadd132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9d,0x0a]
{evex} vfnmadd132sd (%edx), %xmm1, %xmm1

// CHECK: vfnmadd132sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0x0a]
vfnmadd132sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0x0a]
vfnmadd132sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x9d,0xc9]
vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x9d,0xc9]
vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x9d,0xc9]
vfnmadd132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x9d,0xc9]
vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x9d,0xc9]
vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x9d,0xc9]
vfnmadd132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x9d,0xc9]
vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x9d,0xc9]
vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x9d,0xc9]
vfnmadd132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x9d,0xc9]
vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x9d,0xc9]
vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x9d,0xc9]
vfnmadd132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9d,0xc9]
{evex} vfnmadd132sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9d,0xc9]
vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9d,0xc9]
vfnmadd132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9d,0x4c,0x02,0x40]
{evex} vfnmadd132ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x4c,0x02,0x40]
vfnmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x4c,0x02,0x40]
vfnmadd132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd132ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd132ss 485498096, %xmm1, %xmm1

// CHECK: vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9d,0x0a]
{evex} vfnmadd132ss (%edx), %xmm1, %xmm1

// CHECK: vfnmadd132ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0x0a]
vfnmadd132ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0x0a]
vfnmadd132ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x9d,0xc9]
vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x9d,0xc9]
vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x9d,0xc9]
vfnmadd132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x9d,0xc9]
vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x9d,0xc9]
vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x9d,0xc9]
vfnmadd132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x9d,0xc9]
vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x9d,0xc9]
vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x9d,0xc9]
vfnmadd132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x9d,0xc9]
vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x9d,0xc9]
vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x9d,0xc9]
vfnmadd132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9d,0xc9]
{evex} vfnmadd132ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9d,0xc9]
vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9d,0xc9]
vfnmadd132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd213sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd213sd 485498096, %xmm1, %xmm1

// CHECK: vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xad,0x4c,0x02,0x40]
{evex} vfnmadd213sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x4c,0x02,0x40]
vfnmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x4c,0x02,0x40]
vfnmadd213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xad,0x0a]
{evex} vfnmadd213sd (%edx), %xmm1, %xmm1

// CHECK: vfnmadd213sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0x0a]
vfnmadd213sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0x0a]
vfnmadd213sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xad,0xc9]
vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xad,0xc9]
vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xad,0xc9]
vfnmadd213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xad,0xc9]
vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xad,0xc9]
vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xad,0xc9]
vfnmadd213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xad,0xc9]
vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xad,0xc9]
vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xad,0xc9]
vfnmadd213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xad,0xc9]
vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xad,0xc9]
vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xad,0xc9]
vfnmadd213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xad,0xc9]
{evex} vfnmadd213sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xad,0xc9]
vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xad,0xc9]
vfnmadd213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xad,0x4c,0x02,0x40]
{evex} vfnmadd213ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x4c,0x02,0x40]
vfnmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x4c,0x02,0x40]
vfnmadd213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd213ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd213ss 485498096, %xmm1, %xmm1

// CHECK: vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xad,0x0a]
{evex} vfnmadd213ss (%edx), %xmm1, %xmm1

// CHECK: vfnmadd213ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0x0a]
vfnmadd213ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0x0a]
vfnmadd213ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xad,0xc9]
vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xad,0xc9]
vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xad,0xc9]
vfnmadd213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xad,0xc9]
vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xad,0xc9]
vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xad,0xc9]
vfnmadd213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xad,0xc9]
vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xad,0xc9]
vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xad,0xc9]
vfnmadd213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xad,0xc9]
vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xad,0xc9]
vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xad,0xc9]
vfnmadd213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xad,0xc9]
{evex} vfnmadd213ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xad,0xc9]
vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xad,0xc9]
vfnmadd213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd231sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd231sd 485498096, %xmm1, %xmm1

// CHECK: vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd231sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbd,0x4c,0x02,0x40]
{evex} vfnmadd231sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x4c,0x02,0x40]
vfnmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x4c,0x02,0x40]
vfnmadd231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbd,0x0a]
{evex} vfnmadd231sd (%edx), %xmm1, %xmm1

// CHECK: vfnmadd231sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0x0a]
vfnmadd231sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0x0a]
vfnmadd231sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xbd,0xc9]
vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xbd,0xc9]
vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xbd,0xc9]
vfnmadd231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xbd,0xc9]
vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xbd,0xc9]
vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xbd,0xc9]
vfnmadd231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xbd,0xc9]
vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xbd,0xc9]
vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xbd,0xc9]
vfnmadd231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xbd,0xc9]
vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xbd,0xc9]
vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xbd,0xc9]
vfnmadd231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbd,0xc9]
{evex} vfnmadd231sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbd,0xc9]
vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbd,0xc9]
vfnmadd231sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbd,0x4c,0x02,0x40]
{evex} vfnmadd231ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x4c,0x02,0x40]
vfnmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x4c,0x02,0x40]
vfnmadd231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmadd231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd231ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmadd231ss 485498096, %xmm1, %xmm1

// CHECK: vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmadd231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbd,0x0a]
{evex} vfnmadd231ss (%edx), %xmm1, %xmm1

// CHECK: vfnmadd231ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0x0a]
vfnmadd231ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0x0a]
vfnmadd231ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xbd,0xc9]
vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xbd,0xc9]
vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xbd,0xc9]
vfnmadd231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xbd,0xc9]
vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xbd,0xc9]
vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xbd,0xc9]
vfnmadd231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xbd,0xc9]
vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xbd,0xc9]
vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xbd,0xc9]
vfnmadd231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xbd,0xc9]
vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xbd,0xc9]
vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xbd,0xc9]
vfnmadd231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmadd231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbd,0xc9]
{evex} vfnmadd231ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbd,0xc9]
vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbd,0xc9]
vfnmadd231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub132sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub132sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub132sd 485498096, %xmm1, %xmm1

// CHECK: vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub132sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9f,0x4c,0x02,0x40]
{evex} vfnmsub132sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x4c,0x02,0x40]
vfnmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x4c,0x02,0x40]
vfnmsub132sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9f,0x0a]
{evex} vfnmsub132sd (%edx), %xmm1, %xmm1

// CHECK: vfnmsub132sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0x0a]
vfnmsub132sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0x0a]
vfnmsub132sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x9f,0xc9]
vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x9f,0xc9]
vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x9f,0xc9]
vfnmsub132sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x9f,0xc9]
vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x9f,0xc9]
vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x9f,0xc9]
vfnmsub132sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x9f,0xc9]
vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x9f,0xc9]
vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x9f,0xc9]
vfnmsub132sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x9f,0xc9]
vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x9f,0xc9]
vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x9f,0xc9]
vfnmsub132sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x9f,0xc9]
{evex} vfnmsub132sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x9f,0xc9]
vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x9f,0xc9]
vfnmsub132sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9f,0x4c,0x02,0x40]
{evex} vfnmsub132ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x4c,0x02,0x40]
vfnmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x4c,0x02,0x40]
vfnmsub132ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub132ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub132ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub132ss 485498096, %xmm1, %xmm1

// CHECK: vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub132ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9f,0x0a]
{evex} vfnmsub132ss (%edx), %xmm1, %xmm1

// CHECK: vfnmsub132ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0x0a]
vfnmsub132ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0x0a]
vfnmsub132ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x9f,0xc9]
vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x9f,0xc9]
vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x9f,0xc9]
vfnmsub132ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x9f,0xc9]
vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x9f,0xc9]
vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x9f,0xc9]
vfnmsub132ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x9f,0xc9]
vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x9f,0xc9]
vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x9f,0xc9]
vfnmsub132ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x9f,0xc9]
vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x9f,0xc9]
vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x9f,0xc9]
vfnmsub132ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub132ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x9f,0xc9]
{evex} vfnmsub132ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x9f,0xc9]
vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x9f,0xc9]
vfnmsub132ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub213sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub213sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub213sd 485498096, %xmm1, %xmm1

// CHECK: vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub213sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xaf,0x4c,0x02,0x40]
{evex} vfnmsub213sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x4c,0x02,0x40]
vfnmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x4c,0x02,0x40]
vfnmsub213sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xaf,0x0a]
{evex} vfnmsub213sd (%edx), %xmm1, %xmm1

// CHECK: vfnmsub213sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0x0a]
vfnmsub213sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0x0a]
vfnmsub213sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xaf,0xc9]
vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xaf,0xc9]
vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xaf,0xc9]
vfnmsub213sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xaf,0xc9]
vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xaf,0xc9]
vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xaf,0xc9]
vfnmsub213sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xaf,0xc9]
vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xaf,0xc9]
vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xaf,0xc9]
vfnmsub213sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xaf,0xc9]
vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xaf,0xc9]
vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xaf,0xc9]
vfnmsub213sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xaf,0xc9]
{evex} vfnmsub213sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xaf,0xc9]
vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xaf,0xc9]
vfnmsub213sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xaf,0x4c,0x02,0x40]
{evex} vfnmsub213ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x4c,0x02,0x40]
vfnmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x4c,0x02,0x40]
vfnmsub213ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub213ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub213ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub213ss 485498096, %xmm1, %xmm1

// CHECK: vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub213ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xaf,0x0a]
{evex} vfnmsub213ss (%edx), %xmm1, %xmm1

// CHECK: vfnmsub213ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0x0a]
vfnmsub213ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0x0a]
vfnmsub213ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xaf,0xc9]
vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xaf,0xc9]
vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xaf,0xc9]
vfnmsub213ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xaf,0xc9]
vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xaf,0xc9]
vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xaf,0xc9]
vfnmsub213ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xaf,0xc9]
vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xaf,0xc9]
vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xaf,0xc9]
vfnmsub213ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xaf,0xc9]
vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xaf,0xc9]
vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xaf,0xc9]
vfnmsub213ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub213ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xaf,0xc9]
{evex} vfnmsub213ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xaf,0xc9]
vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xaf,0xc9]
vfnmsub213ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub231sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub231sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbf,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub231sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmsub231sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbf,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub231sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbf,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub231sd 485498096, %xmm1, %xmm1

// CHECK: vfnmsub231sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbf,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub231sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xb
vfnmsub231sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbf,0x4c,0x02,0x40]
{evex} vfnmsub231sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmsub231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbf,0x4c,0x02,0x40]
vfnmsub231sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbf,0x0a]
{evex} vfnmsub231sd (%edx), %xmm1, %xmm1

// CHECK: vfnmsub231sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbf,0x0a]
vfnmsub231sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0xbf,0xc9]
vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0xbf,0xc9]
vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0xbf,0xc9]
vfnmsub231sd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0xbf,0xc9]
vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0xbf,0xc9]
vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0xbf,0xc9]
vfnmsub231sd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0xbf,0xc9]
vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0xbf,0xc9]
vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0xbf,0xc9]
vfnmsub231sd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0xbf,0xc9]
vfnmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0xbf,0xc9]
vfnmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0xbf,0xc9]
vfnmsub231sd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0xbf,0xc9]
{evex} vfnmsub231sd %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0xbf,0xc9]
vfnmsub231sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0xbf,0xc9]
vfnmsub231sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbf,0x4c,0x02,0x40]
{evex} vfnmsub231ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vfnmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x4c,0x02,0x40]
vfnmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x4c,0x02,0x40]
vfnmsub231ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vfnmsub231ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbf,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub231ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vfnmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x8a,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbf,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vfnmsub231ss 485498096, %xmm1, %xmm1

// CHECK: vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x0d,0xf0,0x1c,0xf0,0x1c]
vfnmsub231ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbf,0x0a]
{evex} vfnmsub231ss (%edx), %xmm1, %xmm1

// CHECK: vfnmsub231ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0x0a]
vfnmsub231ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0x0a]
vfnmsub231ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0xbf,0xc9]
vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0xbf,0xc9]
vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0xbf,0xc9]
vfnmsub231ss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0xbf,0xc9]
vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0xbf,0xc9]
vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0xbf,0xc9]
vfnmsub231ss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0xbf,0xc9]
vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0xbf,0xc9]
vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0xbf,0xc9]
vfnmsub231ss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0xbf,0xc9]
vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0xbf,0xc9]
vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0xbf,0xc9]
vfnmsub231ss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vfnmsub231ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0xbf,0xc9]
{evex} vfnmsub231ss %xmm1, %xmm1, %xmm1

// CHECK: vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0xbf,0xc9]
vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0xbf,0xc9]
vfnmsub231ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vgetexpsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetexpsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetexpsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vgetexpsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vgetexpsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x8a,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vgetexpsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x8a,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x8a,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x0d,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm1, %xmm1

// CHECK: vgetexpsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x0d,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x0d,0xf0,0x1c,0xf0,0x1c]
vgetexpsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x4c,0x02,0x40]
vgetexpsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vgetexpsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x4c,0x02,0x40]
vgetexpsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x4c,0x02,0x40]
vgetexpsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0x0a]
vgetexpsd (%edx), %xmm1, %xmm1

// CHECK: vgetexpsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0x0a]
vgetexpsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0x0a]
vgetexpsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x43,0xc9]
vgetexpsd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x43,0xc9]
vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x43,0xc9]
vgetexpsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x43,0xc9]
vgetexpsd %xmm1, %xmm1, %xmm1

// CHECK: vgetexpsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x43,0xc9]
vgetexpsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x43,0xc9]
vgetexpsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x4c,0x02,0x40]
vgetexpss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vgetexpss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x4c,0x02,0x40]
vgetexpss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x4c,0x02,0x40]
vgetexpss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vgetexpss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetexpss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetexpss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vgetexpss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vgetexpss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x8a,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096(%edx), %xmm1, %xmm1

// CHECK: vgetexpss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x8a,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x8a,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x0d,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm1, %xmm1

// CHECK: vgetexpss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x0d,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x0d,0xf0,0x1c,0xf0,0x1c]
vgetexpss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0x0a]
vgetexpss (%edx), %xmm1, %xmm1

// CHECK: vgetexpss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0x0a]
vgetexpss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0x0a]
vgetexpss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x43,0xc9]
vgetexpss {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x43,0xc9]
vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x43,0xc9]
vgetexpss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetexpss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x43,0xc9]
vgetexpss %xmm1, %xmm1, %xmm1

// CHECK: vgetexpss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x43,0xc9]
vgetexpss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetexpss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x43,0xc9]
vgetexpss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vgetmantsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetmantsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetmantsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vgetmantsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vgetmantsd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096(%edx), %xmm1, %xmm1

// CHECK: vgetmantsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm1, %xmm1

// CHECK: vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantsd $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x4c,0x02,0x40,0x00]
vgetmantsd $0, 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vgetmantsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x4c,0x02,0x40,0x00]
vgetmantsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x4c,0x02,0x40,0x00]
vgetmantsd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0x0a,0x00]
vgetmantsd $0, (%edx), %xmm1, %xmm1

// CHECK: vgetmantsd $0, (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0x0a,0x00]
vgetmantsd $0, (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0x0a,0x00]
vgetmantsd $0, (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x18,0x27,0xc9,0x00]
vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x1a,0x27,0xc9,0x00]
vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x9a,0x27,0xc9,0x00]
vgetmantsd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantsd $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x27,0xc9,0x00]
vgetmantsd $0, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x27,0xc9,0x00]
vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x27,0xc9,0x00]
vgetmantsd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x4c,0x02,0x40,0x00]
vgetmantss $0, 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vgetmantss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x4c,0x02,0x40,0x00]
vgetmantss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x4c,0x02,0x40,0x00]
vgetmantss $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vgetmantss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetmantss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vgetmantss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vgetmantss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vgetmantss $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096(%edx), %xmm1, %xmm1

// CHECK: vgetmantss $0, 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm1, %xmm1

// CHECK: vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vgetmantss $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0x0a,0x00]
vgetmantss $0, (%edx), %xmm1, %xmm1

// CHECK: vgetmantss $0, (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0x0a,0x00]
vgetmantss $0, (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0x0a,0x00]
vgetmantss $0, (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x18,0x27,0xc9,0x00]
vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x1a,0x27,0xc9,0x00]
vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x9a,0x27,0xc9,0x00]
vgetmantss $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vgetmantss $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x27,0xc9,0x00]
vgetmantss $0, %xmm1, %xmm1, %xmm1

// CHECK: vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x27,0xc9,0x00]
vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x27,0xc9,0x00]
vgetmantss $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmaxsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmaxsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vmaxsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vmaxsd 485498096, %xmm1, %xmm1

// CHECK: vmaxsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]
vmaxsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5f,0x4c,0x02,0x40]
{evex} vmaxsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vmaxsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x4c,0x02,0x40]
vmaxsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x4c,0x02,0x40]
vmaxsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5f,0x0a]
{evex} vmaxsd (%edx), %xmm1, %xmm1

// CHECK: vmaxsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0x0a]
vmaxsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0x0a]
vmaxsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5f,0xc9]
vmaxsd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5f,0xc9]
vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5f,0xc9]
vmaxsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5f,0xc9]
{evex} vmaxsd %xmm1, %xmm1, %xmm1

// CHECK: vmaxsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5f,0xc9]
vmaxsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5f,0xc9]
vmaxsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5f,0x4c,0x02,0x40]
{evex} vmaxss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vmaxss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x4c,0x02,0x40]
vmaxss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x4c,0x02,0x40]
vmaxss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmaxss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmaxss 485498096(%edx), %xmm1, %xmm1

// CHECK: vmaxss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x8a,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vmaxss 485498096, %xmm1, %xmm1

// CHECK: vmaxss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmaxss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x0d,0xf0,0x1c,0xf0,0x1c]
vmaxss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5f,0x0a]
{evex} vmaxss (%edx), %xmm1, %xmm1

// CHECK: vmaxss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0x0a]
vmaxss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmaxss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0x0a]
vmaxss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5f,0xc9]
vmaxss {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5f,0xc9]
vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5f,0xc9]
vmaxss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmaxss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5f,0xc9]
{evex} vmaxss %xmm1, %xmm1, %xmm1

// CHECK: vmaxss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5f,0xc9]
vmaxss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmaxss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5f,0xc9]
vmaxss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vminsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vminsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vminsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vminsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vminsd 485498096, %xmm1, %xmm1

// CHECK: vminsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vminsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]
vminsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5d,0x4c,0x02,0x40]
{evex} vminsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vminsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x4c,0x02,0x40]
vminsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vminsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x4c,0x02,0x40]
vminsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5d,0x0a]
{evex} vminsd (%edx), %xmm1, %xmm1

// CHECK: vminsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0x0a]
vminsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vminsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0x0a]
vminsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5d,0xc9]
vminsd {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5d,0xc9]
vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5d,0xc9]
vminsd {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5d,0xc9]
{evex} vminsd %xmm1, %xmm1, %xmm1

// CHECK: vminsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5d,0xc9]
vminsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5d,0xc9]
vminsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5d,0x4c,0x02,0x40]
{evex} vminss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vminss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x4c,0x02,0x40]
vminss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vminss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x4c,0x02,0x40]
vminss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vminss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vminss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vminss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vminss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vminss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vminss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vminss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vminss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vminss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vminss 485498096(%edx), %xmm1, %xmm1

// CHECK: vminss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]
vminss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vminss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x8a,0xf0,0x1c,0xf0,0x1c]
vminss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vminss 485498096, %xmm1, %xmm1

// CHECK: vminss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vminss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x0d,0xf0,0x1c,0xf0,0x1c]
vminss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5d,0x0a]
{evex} vminss (%edx), %xmm1, %xmm1

// CHECK: vminss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0x0a]
vminss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vminss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0x0a]
vminss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5d,0xc9]
vminss {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5d,0xc9]
vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5d,0xc9]
vminss {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vminss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5d,0xc9]
{evex} vminss %xmm1, %xmm1, %xmm1

// CHECK: vminss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5d,0xc9]
vminss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vminss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5d,0xc9]
vminss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovsd -485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmovsd -485498096(%edx,%eax,4), %xmm1

// CHECK: vmovsd 485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmovsd 485498096(%edx,%eax,4), %xmm1

// CHECK: vmovsd -485498096(%edx,%eax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmovsd -485498096(%edx,%eax,4), %xmm1 {%k2}

// CHECK: vmovsd 485498096(%edx,%eax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096(%edx,%eax,4), %xmm1 {%k2}

// CHECK: vmovsd -485498096(%edx,%eax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmovsd -485498096(%edx,%eax,4), %xmm1 {%k2} {z}

// CHECK: vmovsd 485498096(%edx,%eax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096(%edx,%eax,4), %xmm1 {%k2} {z}

// CHECK: vmovsd 485498096(%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmovsd 485498096(%edx), %xmm1

// CHECK: vmovsd 485498096(%edx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096(%edx), %xmm1 {%k2}

// CHECK: vmovsd 485498096(%edx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096(%edx), %xmm1 {%k2} {z}

// CHECK: vmovsd 485498096, %xmm1
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vmovsd 485498096, %xmm1

// CHECK: vmovsd 485498096, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm1 {%k2}

// CHECK: vmovsd 485498096, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]
vmovsd 485498096, %xmm1 {%k2} {z}

// CHECK: vmovsd 512(%edx,%eax), %xmm1
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x10,0x4c,0x02,0x40]
{evex} vmovsd 512(%edx,%eax), %xmm1

// CHECK: vmovsd 512(%edx,%eax), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x4c,0x02,0x40]
vmovsd 512(%edx,%eax), %xmm1 {%k2}

// CHECK: vmovsd 512(%edx,%eax), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x4c,0x02,0x40]
vmovsd 512(%edx,%eax), %xmm1 {%k2} {z}

// CHECK: vmovsd (%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x10,0x0a]
{evex} vmovsd (%edx), %xmm1

// CHECK: vmovsd (%edx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x10,0x0a]
vmovsd (%edx), %xmm1 {%k2}

// CHECK: vmovsd (%edx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xff,0x8a,0x10,0x0a]
vmovsd (%edx), %xmm1 {%k2} {z}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x11,0xc9]
{evex} vmovsd.s %xmm1, %xmm1, %xmm1

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x11,0xc9]
vmovsd.s %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x11,0xc9]
vmovsd.s %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovsd %xmm1, -485498096(%edx,%eax,4)
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmovsd %xmm1, -485498096(%edx,%eax,4)

// CHECK: vmovsd %xmm1, 485498096(%edx,%eax,4)
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmovsd %xmm1, 485498096(%edx,%eax,4)

// CHECK: vmovsd %xmm1, -485498096(%edx,%eax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmovsd %xmm1, -485498096(%edx,%eax,4) {%k2}

// CHECK: vmovsd %xmm1, 485498096(%edx,%eax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmovsd %xmm1, 485498096(%edx,%eax,4) {%k2}

// CHECK: vmovsd %xmm1, 485498096(%edx)
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmovsd %xmm1, 485498096(%edx)

// CHECK: vmovsd %xmm1, 485498096(%edx) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]
vmovsd %xmm1, 485498096(%edx) {%k2}

// CHECK: vmovsd %xmm1, 485498096
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x11,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vmovsd %xmm1, 485498096

// CHECK: vmovsd %xmm1, 485498096 {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x0d,0xf0,0x1c,0xf0,0x1c]
vmovsd %xmm1, 485498096 {%k2}

// CHECK: vmovsd %xmm1, 512(%edx,%eax)
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x11,0x4c,0x02,0x40]
{evex} vmovsd %xmm1, 512(%edx,%eax)

// CHECK: vmovsd %xmm1, 512(%edx,%eax) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x4c,0x02,0x40]
vmovsd %xmm1, 512(%edx,%eax) {%k2}

// CHECK: vmovsd %xmm1, (%edx)
// CHECK: encoding: [0x62,0xf1,0xff,0x08,0x11,0x0a]
{evex} vmovsd %xmm1, (%edx)

// CHECK: vmovsd %xmm1, (%edx) {%k2}
// CHECK: encoding: [0x62,0xf1,0xff,0x0a,0x11,0x0a]
vmovsd %xmm1, (%edx) {%k2}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x10,0xc9]
{evex} vmovsd %xmm1, %xmm1, %xmm1

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x10,0xc9]
vmovsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x10,0xc9]
vmovsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovss 256(%edx,%eax), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x10,0x4c,0x02,0x40]
{evex} vmovss 256(%edx,%eax), %xmm1

// CHECK: vmovss 256(%edx,%eax), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x4c,0x02,0x40]
vmovss 256(%edx,%eax), %xmm1 {%k2}

// CHECK: vmovss 256(%edx,%eax), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x4c,0x02,0x40]
vmovss 256(%edx,%eax), %xmm1 {%k2} {z}

// CHECK: vmovss -485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmovss -485498096(%edx,%eax,4), %xmm1

// CHECK: vmovss 485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmovss 485498096(%edx,%eax,4), %xmm1

// CHECK: vmovss -485498096(%edx,%eax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmovss -485498096(%edx,%eax,4), %xmm1 {%k2}

// CHECK: vmovss 485498096(%edx,%eax,4), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096(%edx,%eax,4), %xmm1 {%k2}

// CHECK: vmovss -485498096(%edx,%eax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmovss -485498096(%edx,%eax,4), %xmm1 {%k2} {z}

// CHECK: vmovss 485498096(%edx,%eax,4), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096(%edx,%eax,4), %xmm1 {%k2} {z}

// CHECK: vmovss 485498096(%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmovss 485498096(%edx), %xmm1

// CHECK: vmovss 485498096(%edx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096(%edx), %xmm1 {%k2}

// CHECK: vmovss 485498096(%edx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x8a,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096(%edx), %xmm1 {%k2} {z}

// CHECK: vmovss 485498096, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vmovss 485498096, %xmm1

// CHECK: vmovss 485498096, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm1 {%k2}

// CHECK: vmovss 485498096, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x0d,0xf0,0x1c,0xf0,0x1c]
vmovss 485498096, %xmm1 {%k2} {z}

// CHECK: vmovss (%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x10,0x0a]
{evex} vmovss (%edx), %xmm1

// CHECK: vmovss (%edx), %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x10,0x0a]
vmovss (%edx), %xmm1 {%k2}

// CHECK: vmovss (%edx), %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x7e,0x8a,0x10,0x0a]
vmovss (%edx), %xmm1 {%k2} {z}

// CHECK: vmovss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x11,0xc9]
{evex} vmovss.s %xmm1, %xmm1, %xmm1

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x11,0xc9]
vmovss.s %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x11,0xc9]
vmovss.s %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmovss %xmm1, 256(%edx,%eax)
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x11,0x4c,0x02,0x40]
{evex} vmovss %xmm1, 256(%edx,%eax)

// CHECK: vmovss %xmm1, 256(%edx,%eax) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x4c,0x02,0x40]
vmovss %xmm1, 256(%edx,%eax) {%k2}

// CHECK: vmovss %xmm1, -485498096(%edx,%eax,4)
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmovss %xmm1, -485498096(%edx,%eax,4)

// CHECK: vmovss %xmm1, 485498096(%edx,%eax,4)
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmovss %xmm1, 485498096(%edx,%eax,4)

// CHECK: vmovss %xmm1, -485498096(%edx,%eax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmovss %xmm1, -485498096(%edx,%eax,4) {%k2}

// CHECK: vmovss %xmm1, 485498096(%edx,%eax,4) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmovss %xmm1, 485498096(%edx,%eax,4) {%k2}

// CHECK: vmovss %xmm1, 485498096(%edx)
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmovss %xmm1, 485498096(%edx)

// CHECK: vmovss %xmm1, 485498096(%edx) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x8a,0xf0,0x1c,0xf0,0x1c]
vmovss %xmm1, 485498096(%edx) {%k2}

// CHECK: vmovss %xmm1, 485498096
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x11,0x0d,0xf0,0x1c,0xf0,0x1c
{evex} vmovss %xmm1, 485498096

// CHECK: vmovss %xmm1, 485498096 {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x0d,0xf0,0x1c,0xf0,0x1c]
vmovss %xmm1, 485498096 {%k2}

// CHECK: vmovss %xmm1, (%edx)
// CHECK: encoding: [0x62,0xf1,0x7e,0x08,0x11,0x0a]
{evex} vmovss %xmm1, (%edx)

// CHECK: vmovss %xmm1, (%edx) {%k2}
// CHECK: encoding: [0x62,0xf1,0x7e,0x0a,0x11,0x0a]
vmovss %xmm1, (%edx) {%k2}

// CHECK: vmovss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x10,0xc9]
{evex} vmovss %xmm1, %xmm1, %xmm1

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x10,0xc9]
vmovss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmovss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x10,0xc9]
vmovss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmulsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmulsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vmulsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vmulsd 485498096, %xmm1, %xmm1

// CHECK: vmulsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]
vmulsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x59,0x4c,0x02,0x40]
{evex} vmulsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vmulsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x4c,0x02,0x40]
vmulsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x4c,0x02,0x40]
vmulsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x59,0x0a]
{evex} vmulsd (%edx), %xmm1, %xmm1

// CHECK: vmulsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0x0a]
vmulsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmulsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0x0a]
vmulsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x59,0xc9]
vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x59,0xc9]
vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x59,0xc9]
vmulsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x59,0xc9]
vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x59,0xc9]
vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x59,0xc9]
vmulsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x59,0xc9]
vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x59,0xc9]
vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x59,0xc9]
vmulsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x59,0xc9]
vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x59,0xc9]
vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x59,0xc9]
vmulsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x59,0xc9]
{evex} vmulsd %xmm1, %xmm1, %xmm1

// CHECK: vmulsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x59,0xc9]
vmulsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x59,0xc9]
vmulsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x59,0x4c,0x02,0x40]
{evex} vmulss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vmulss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x4c,0x02,0x40]
vmulss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vmulss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x4c,0x02,0x40]
vmulss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vmulss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vmulss 485498096(%edx), %xmm1, %xmm1

// CHECK: vmulss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmulss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x8a,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vmulss 485498096, %xmm1, %xmm1

// CHECK: vmulss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vmulss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x0d,0xf0,0x1c,0xf0,0x1c]
vmulss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x59,0x0a]
{evex} vmulss (%edx), %xmm1, %xmm1

// CHECK: vmulss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0x0a]
vmulss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vmulss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0x0a]
vmulss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x59,0xc9]
vmulss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x59,0xc9]
vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x59,0xc9]
vmulss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x59,0xc9]
vmulss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x59,0xc9]
vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x59,0xc9]
vmulss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x59,0xc9]
vmulss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x59,0xc9]
vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x59,0xc9]
vmulss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x59,0xc9]
vmulss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x59,0xc9]
vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x59,0xc9]
vmulss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vmulss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x59,0xc9]
{evex} vmulss %xmm1, %xmm1, %xmm1

// CHECK: vmulss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x59,0xc9]
vmulss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vmulss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x59,0xc9]
vmulss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrcp14sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrcp14sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrcp14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrcp14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrcp14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x8a,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vrcp14sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x8a,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x8a,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x0d,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm1, %xmm1

// CHECK: vrcp14sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x0d,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x0d,0xf0,0x1c,0xf0,0x1c]
vrcp14sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x4c,0x02,0x40]
vrcp14sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vrcp14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x4c,0x02,0x40]
vrcp14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x4c,0x02,0x40]
vrcp14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0x0a]
vrcp14sd (%edx), %xmm1, %xmm1

// CHECK: vrcp14sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0x0a]
vrcp14sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0x0a]
vrcp14sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4d,0xc9]
vrcp14sd %xmm1, %xmm1, %xmm1

// CHECK: vrcp14sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4d,0xc9]
vrcp14sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4d,0xc9]
vrcp14sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x4c,0x02,0x40]
vrcp14ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vrcp14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x4c,0x02,0x40]
vrcp14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x4c,0x02,0x40]
vrcp14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrcp14ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrcp14ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrcp14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrcp14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrcp14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x8a,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vrcp14ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x8a,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x8a,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x0d,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm1, %xmm1

// CHECK: vrcp14ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x0d,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x0d,0xf0,0x1c,0xf0,0x1c]
vrcp14ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0x0a]
vrcp14ss (%edx), %xmm1, %xmm1

// CHECK: vrcp14ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0x0a]
vrcp14ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0x0a]
vrcp14ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrcp14ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4d,0xc9]
vrcp14ss %xmm1, %xmm1, %xmm1

// CHECK: vrcp14ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4d,0xc9]
vrcp14ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrcp14ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4d,0xc9]
vrcp14ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vrndscalesd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrndscalesd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrndscalesd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vrndscalesd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vrndscalesd $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096(%edx), %xmm1, %xmm1

// CHECK: vrndscalesd $0, 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm1, %xmm1

// CHECK: vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscalesd $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x4c,0x02,0x40,0x00]
vrndscalesd $0, 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vrndscalesd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x4c,0x02,0x40,0x00]
vrndscalesd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x4c,0x02,0x40,0x00]
vrndscalesd $0, 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0x0a,0x00]
vrndscalesd $0, (%edx), %xmm1, %xmm1

// CHECK: vrndscalesd $0, (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0x0a,0x00]
vrndscalesd $0, (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0x0a,0x00]
vrndscalesd $0, (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x18,0x0b,0xc9,0x00]
vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x1a,0x0b,0xc9,0x00]
vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x9a,0x0b,0xc9,0x00]
vrndscalesd $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscalesd $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0xf5,0x08,0x0b,0xc9,0x00]
vrndscalesd $0, %xmm1, %xmm1, %xmm1

// CHECK: vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xf5,0x0a,0x0b,0xc9,0x00]
vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0xf5,0x8a,0x0b,0xc9,0x00]
vrndscalesd $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x4c,0x02,0x40,0x00]
vrndscaless $0, 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vrndscaless $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x4c,0x02,0x40,0x00]
vrndscaless $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x4c,0x02,0x40,0x00]
vrndscaless $0, 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vrndscaless $0, -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrndscaless $0, 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrndscaless $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vrndscaless $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x8c,0x82,0x10,0xe3,0x0f,0xe3,0x00]
vrndscaless $0, -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x8c,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096(%edx), %xmm1, %xmm1

// CHECK: vrndscaless $0, 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x8a,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm1, %xmm1

// CHECK: vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x0d,0xf0,0x1c,0xf0,0x1c,0x00]
vrndscaless $0, 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0x0a,0x00]
vrndscaless $0, (%edx), %xmm1, %xmm1

// CHECK: vrndscaless $0, (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0x0a,0x00]
vrndscaless $0, (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0x0a,0x00]
vrndscaless $0, (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x18,0x0a,0xc9,0x00]
vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1

// CHECK: vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x1a,0x0a,0xc9,0x00]
vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x9a,0x0a,0xc9,0x00]
vrndscaless $0, {sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrndscaless $0, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf3,0x75,0x08,0x0a,0xc9,0x00]
vrndscaless $0, %xmm1, %xmm1, %xmm1

// CHECK: vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0x75,0x0a,0x0a,0xc9,0x00]
vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf3,0x75,0x8a,0x0a,0xc9,0x00]
vrndscaless $0, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrsqrt14sd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrsqrt14sd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrsqrt14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrsqrt14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrsqrt14sd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x8a,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096(%edx), %xmm1, %xmm1

// CHECK: vrsqrt14sd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x8a,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x8a,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x0d,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm1, %xmm1

// CHECK: vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x0d,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x0d,0xf0,0x1c,0xf0,0x1c]
vrsqrt14sd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x4c,0x02,0x40]
vrsqrt14sd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vrsqrt14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x4c,0x02,0x40]
vrsqrt14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x4c,0x02,0x40]
vrsqrt14sd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0x0a]
vrsqrt14sd (%edx), %xmm1, %xmm1

// CHECK: vrsqrt14sd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0x0a]
vrsqrt14sd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0x0a]
vrsqrt14sd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14sd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x4f,0xc9]
vrsqrt14sd %xmm1, %xmm1, %xmm1

// CHECK: vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x4f,0xc9]
vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x4f,0xc9]
vrsqrt14sd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x4c,0x02,0x40]
vrsqrt14ss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vrsqrt14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x4c,0x02,0x40]
vrsqrt14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x4c,0x02,0x40]
vrsqrt14ss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrsqrt14ss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrsqrt14ss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vrsqrt14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrsqrt14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vrsqrt14ss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x8a,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096(%edx), %xmm1, %xmm1

// CHECK: vrsqrt14ss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x8a,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x8a,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x0d,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm1, %xmm1

// CHECK: vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x0d,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x0d,0xf0,0x1c,0xf0,0x1c]
vrsqrt14ss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0x0a]
vrsqrt14ss (%edx), %xmm1, %xmm1

// CHECK: vrsqrt14ss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0x0a]
vrsqrt14ss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0x0a]
vrsqrt14ss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vrsqrt14ss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x4f,0xc9]
vrsqrt14ss %xmm1, %xmm1, %xmm1

// CHECK: vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x4f,0xc9]
vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x4f,0xc9]
vrsqrt14ss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vscalefsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vscalefsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vscalefsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vscalefsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vscalefsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x8a,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vscalefsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x8a,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x8a,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x0d,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm1, %xmm1

// CHECK: vscalefsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x0d,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x0d,0xf0,0x1c,0xf0,0x1c]
vscalefsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x4c,0x02,0x40]
vscalefsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vscalefsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x4c,0x02,0x40]
vscalefsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x4c,0x02,0x40]
vscalefsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0x0a]
vscalefsd (%edx), %xmm1, %xmm1

// CHECK: vscalefsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0x0a]
vscalefsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0x0a]
vscalefsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x38,0x2d,0xc9]
vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x3a,0x2d,0xc9]
vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xba,0x2d,0xc9]
vscalefsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x18,0x2d,0xc9]
vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x1a,0x2d,0xc9]
vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x9a,0x2d,0xc9]
vscalefsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x58,0x2d,0xc9]
vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x5a,0x2d,0xc9]
vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xda,0x2d,0xc9]
vscalefsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x78,0x2d,0xc9]
vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x7a,0x2d,0xc9]
vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0xfa,0x2d,0xc9]
vscalefsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0xf5,0x08,0x2d,0xc9]
vscalefsd %xmm1, %xmm1, %xmm1

// CHECK: vscalefsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0xf5,0x0a,0x2d,0xc9]
vscalefsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0xf5,0x8a,0x2d,0xc9]
vscalefsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x4c,0x02,0x40]
vscalefss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vscalefss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x4c,0x02,0x40]
vscalefss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x4c,0x02,0x40]
vscalefss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vscalefss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vscalefss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vscalefss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vscalefss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vscalefss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x8a,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096(%edx), %xmm1, %xmm1

// CHECK: vscalefss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x8a,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x8a,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x0d,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm1, %xmm1

// CHECK: vscalefss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x0d,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x0d,0xf0,0x1c,0xf0,0x1c]
vscalefss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0x0a]
vscalefss (%edx), %xmm1, %xmm1

// CHECK: vscalefss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0x0a]
vscalefss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vscalefss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0x0a]
vscalefss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x38,0x2d,0xc9]
vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x3a,0x2d,0xc9]
vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xba,0x2d,0xc9]
vscalefss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x18,0x2d,0xc9]
vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x1a,0x2d,0xc9]
vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x9a,0x2d,0xc9]
vscalefss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x58,0x2d,0xc9]
vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x5a,0x2d,0xc9]
vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xda,0x2d,0xc9]
vscalefss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x78,0x2d,0xc9]
vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x7a,0x2d,0xc9]
vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0xfa,0x2d,0xc9]
vscalefss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vscalefss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf2,0x75,0x08,0x2d,0xc9]
vscalefss %xmm1, %xmm1, %xmm1

// CHECK: vscalefss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x75,0x0a,0x2d,0xc9]
vscalefss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vscalefss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf2,0x75,0x8a,0x2d,0xc9]
vscalefss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsqrtsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vsqrtsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vsqrtsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vsqrtsd 485498096, %xmm1, %xmm1

// CHECK: vsqrtsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]
vsqrtsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x51,0x4c,0x02,0x40]
{evex} vsqrtsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vsqrtsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x4c,0x02,0x40]
vsqrtsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x4c,0x02,0x40]
vsqrtsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x51,0x0a]
{evex} vsqrtsd (%edx), %xmm1, %xmm1

// CHECK: vsqrtsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0x0a]
vsqrtsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0x0a]
vsqrtsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x51,0xc9]
vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x51,0xc9]
vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x51,0xc9]
vsqrtsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x51,0xc9]
vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x51,0xc9]
vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x51,0xc9]
vsqrtsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x51,0xc9]
vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x51,0xc9]
vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x51,0xc9]
vsqrtsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x51,0xc9]
vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x51,0xc9]
vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x51,0xc9]
vsqrtsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x51,0xc9]
{evex} vsqrtsd %xmm1, %xmm1, %xmm1

// CHECK: vsqrtsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x51,0xc9]
vsqrtsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x51,0xc9]
vsqrtsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x51,0x4c,0x02,0x40
{evex} vsqrtss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vsqrtss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x4c,0x02,0x40]
vsqrtss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x4c,0x02,0x40]
vsqrtss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsqrtss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vsqrtss 485498096(%edx), %xmm1, %xmm1

// CHECK: vsqrtss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x8a,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vsqrtss 485498096, %xmm1, %xmm1

// CHECK: vsqrtss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x0d,0xf0,0x1c,0xf0,0x1c]
vsqrtss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x51,0x0a]
{evex} vsqrtss (%edx), %xmm1, %xmm1

// CHECK: vsqrtss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0x0a]
vsqrtss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0x0a]
vsqrtss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x51,0xc9]
vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x51,0xc9]
vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x51,0xc9]
vsqrtss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x51,0xc9]
vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x51,0xc9]
vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x51,0xc9]
vsqrtss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x51,0xc9]
vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x51,0xc9]
vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x51,0xc9]
vsqrtss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x51,0xc9]
vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x51,0xc9]
vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x51,0xc9]
vsqrtss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsqrtss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x51,0xc9]
{evex} vsqrtss %xmm1, %xmm1, %xmm1

// CHECK: vsqrtss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x51,0xc9]
vsqrtss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsqrtss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x51,0xc9]
vsqrtss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsubsd -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vsubsd 485498096(%edx), %xmm1, %xmm1

// CHECK: vsubsd 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vsubsd 485498096, %xmm1, %xmm1

// CHECK: vsubsd 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]
vsubsd 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd 512(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5c,0x4c,0x02,0x40]
{evex} vsubsd 512(%edx,%eax), %xmm1, %xmm1

// CHECK: vsubsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x4c,0x02,0x40]
vsubsd 512(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x4c,0x02,0x40]
vsubsd 512(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5c,0x0a]
{evex} vsubsd (%edx), %xmm1, %xmm1

// CHECK: vsubsd (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0x0a]
vsubsd (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsubsd (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0x0a]
vsubsd (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x38,0x5c,0xc9]
vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x3a,0x5c,0xc9]
vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xba,0x5c,0xc9]
vsubsd {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x18,0x5c,0xc9]
vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x1a,0x5c,0xc9]
vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x9a,0x5c,0xc9]
vsubsd {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x58,0x5c,0xc9]
vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x5a,0x5c,0xc9]
vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xda,0x5c,0xc9]
vsubsd {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x78,0x5c,0xc9]
vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x7a,0x5c,0xc9]
vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0xfa,0x5c,0xc9]
vsubsd {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubsd %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xf7,0x08,0x5c,0xc9]
{evex} vsubsd %xmm1, %xmm1, %xmm1

// CHECK: vsubsd %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0xf7,0x0a,0x5c,0xc9]
vsubsd %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubsd %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0xf7,0x8a,0x5c,0xc9]
vsubsd %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 256(%edx,%eax), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5c,0x4c,0x02,0x40]
{evex} vsubss 256(%edx,%eax), %xmm1, %xmm1

// CHECK: vsubss 256(%edx,%eax), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x4c,0x02,0x40]
vsubss 256(%edx,%eax), %xmm1, %xmm1 {%k2}

// CHECK: vsubss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x4c,0x02,0x40]
vsubss 256(%edx,%eax), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1

// CHECK: vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2}

// CHECK: vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
vsubss -485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096(%edx,%eax,4), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 485498096(%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vsubss 485498096(%edx), %xmm1, %xmm1

// CHECK: vsubss 485498096(%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096(%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsubss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x8a,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096(%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss 485498096, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vsubss 485498096, %xmm1, %xmm1

// CHECK: vsubss 485498096, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm1, %xmm1 {%k2}

// CHECK: vsubss 485498096, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x0d,0xf0,0x1c,0xf0,0x1c]
vsubss 485498096, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss (%edx), %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5c,0x0a]
{evex} vsubss (%edx), %xmm1, %xmm1

// CHECK: vsubss (%edx), %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0x0a]
vsubss (%edx), %xmm1, %xmm1 {%k2}

// CHECK: vsubss (%edx), %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0x0a]
vsubss (%edx), %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {rd-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x38,0x5c,0xc9]
vsubss {rd-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x3a,0x5c,0xc9]
vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xba,0x5c,0xc9]
vsubss {rd-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {rn-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x18,0x5c,0xc9]
vsubss {rn-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x1a,0x5c,0xc9]
vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x9a,0x5c,0xc9]
vsubss {rn-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {ru-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x58,0x5c,0xc9]
vsubss {ru-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x5a,0x5c,0xc9]
vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xda,0x5c,0xc9]
vsubss {ru-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss {rz-sae}, %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x78,0x5c,0xc9]
vsubss {rz-sae}, %xmm1, %xmm1, %xmm1

// CHECK: vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x7a,0x5c,0xc9]
vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0xfa,0x5c,0xc9]
vsubss {rz-sae}, %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vsubss %xmm1, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x76,0x08,0x5c,0xc9]
{evex} vsubss %xmm1, %xmm1, %xmm1

// CHECK: vsubss %xmm1, %xmm1, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf1,0x76,0x0a,0x5c,0xc9]
vsubss %xmm1, %xmm1, %xmm1 {%k2}

// CHECK: vsubss %xmm1, %xmm1, %xmm1 {%k2} {z}
// CHECK: encoding: [0x62,0xf1,0x76,0x8a,0x5c,0xc9]
vsubss %xmm1, %xmm1, %xmm1 {%k2} {z}

// CHECK: vucomisd -485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vucomisd -485498096(%edx,%eax,4), %xmm1

// CHECK: vucomisd 485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vucomisd 485498096(%edx,%eax,4), %xmm1

// CHECK: vucomisd 485498096(%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2e,0x8a,0xf0,0x1c,0xf0,0x1c
{evex} vucomisd 485498096(%edx), %xmm1

// CHECK: vucomisd 485498096, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2e,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vucomisd 485498096, %xmm1

// CHECK: vucomisd 512(%edx,%eax), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2e,0x4c,0x02,0x40]
{evex} vucomisd 512(%edx,%eax), %xmm1

// CHECK: vucomisd (%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2e,0x0a]
{evex} vucomisd (%edx), %xmm1

// CHECK: vucomisd {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x18,0x2e,0xc9]
vucomisd {sae}, %xmm1, %xmm1

// CHECK: vucomisd %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0xfd,0x08,0x2e,0xc9]
{evex} vucomisd %xmm1, %xmm1

// CHECK: vucomiss 256(%edx,%eax), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2e,0x4c,0x02,0x40]
{evex} vucomiss 256(%edx,%eax), %xmm1

// CHECK: vucomiss -485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2e,0x8c,0x82,0x10,0xe3,0x0f,0xe3]
{evex} vucomiss -485498096(%edx,%eax,4), %xmm1

// CHECK: vucomiss 485498096(%edx,%eax,4), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2e,0x8c,0x82,0xf0,0x1c,0xf0,0x1c]
{evex} vucomiss 485498096(%edx,%eax,4), %xmm1

// CHECK: vucomiss 485498096(%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2e,0x8a,0xf0,0x1c,0xf0,0x1c]
{evex} vucomiss 485498096(%edx), %xmm1

// CHECK: vucomiss 485498096, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2e,0x0d,0xf0,0x1c,0xf0,0x1c]
{evex} vucomiss 485498096, %xmm1

// CHECK: vucomiss (%edx), %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2e,0x0a]
{evex} vucomiss (%edx), %xmm1

// CHECK: vucomiss {sae}, %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x18,0x2e,0xc9]
vucomiss {sae}, %xmm1, %xmm1

// CHECK: vucomiss %xmm1, %xmm1
// CHECK: encoding: [0x62,0xf1,0x7c,0x08,0x2e,0xc9]
{evex} vucomiss %xmm1, %xmm1

