// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: addps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x58,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
addps 485498096, %xmm6

// CHECK: addps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x58,0x74,0x82,0xc0]
addps -64(%rdx,%rax,4), %xmm6

// CHECK: addps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x58,0x74,0x82,0x40]
addps 64(%rdx,%rax,4), %xmm6

// CHECK: addps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x58,0x74,0x02,0x40]
addps 64(%rdx,%rax), %xmm6

// CHECK: addps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x58,0x72,0x40]
addps 64(%rdx), %xmm6

// CHECK: addps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x58,0x32]
addps (%rdx), %xmm6

// CHECK: addps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x58,0xf6]
addps %xmm6, %xmm6

// CHECK: addss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x58,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
addss 485498096, %xmm6

// CHECK: addss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x58,0x74,0x82,0xc0]
addss -64(%rdx,%rax,4), %xmm6

// CHECK: addss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x58,0x74,0x82,0x40]
addss 64(%rdx,%rax,4), %xmm6

// CHECK: addss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x58,0x74,0x02,0x40]
addss 64(%rdx,%rax), %xmm6

// CHECK: addss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x58,0x72,0x40]
addss 64(%rdx), %xmm6

// CHECK: addss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x58,0x32]
addss (%rdx), %xmm6

// CHECK: addss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x58,0xf6]
addss %xmm6, %xmm6

// CHECK: andnps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x55,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
andnps 485498096, %xmm6

// CHECK: andnps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x55,0x74,0x82,0xc0]
andnps -64(%rdx,%rax,4), %xmm6

// CHECK: andnps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x55,0x74,0x82,0x40]
andnps 64(%rdx,%rax,4), %xmm6

// CHECK: andnps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x55,0x74,0x02,0x40]
andnps 64(%rdx,%rax), %xmm6

// CHECK: andnps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x55,0x72,0x40]
andnps 64(%rdx), %xmm6

// CHECK: andnps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x55,0x32]
andnps (%rdx), %xmm6

// CHECK: andnps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x55,0xf6]
andnps %xmm6, %xmm6

// CHECK: andps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x54,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
andps 485498096, %xmm6

// CHECK: andps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x54,0x74,0x82,0xc0]
andps -64(%rdx,%rax,4), %xmm6

// CHECK: andps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x54,0x74,0x82,0x40]
andps 64(%rdx,%rax,4), %xmm6

// CHECK: andps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x54,0x74,0x02,0x40]
andps 64(%rdx,%rax), %xmm6

// CHECK: andps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x54,0x72,0x40]
andps 64(%rdx), %xmm6

// CHECK: andps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x54,0x32]
andps (%rdx), %xmm6

// CHECK: andps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x54,0xf6]
andps %xmm6, %xmm6

// CHECK: cmpeqps 485498096, %xmm6
// CHECK: encoding: [0x0f,0xc2,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
cmpeqps 485498096, %xmm6

// CHECK: cmpeqps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0xc2,0x74,0x82,0xc0,0x00]
cmpeqps -64(%rdx,%rax,4), %xmm6

// CHECK: cmpeqps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0xc2,0x74,0x82,0x40,0x00]
cmpeqps 64(%rdx,%rax,4), %xmm6

// CHECK: cmpeqps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0xc2,0x74,0x02,0x40,0x00]
cmpeqps 64(%rdx,%rax), %xmm6

// CHECK: cmpeqps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0xc2,0x72,0x40,0x00]
cmpeqps 64(%rdx), %xmm6

// CHECK: cmpeqps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0xc2,0x32,0x00]
cmpeqps (%rdx), %xmm6

// CHECK: cmpeqps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0xc2,0xf6,0x00]
cmpeqps %xmm6, %xmm6

// CHECK: cmpeqss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0xc2,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
cmpeqss 485498096, %xmm6

// CHECK: cmpeqss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0xc2,0x74,0x82,0xc0,0x00]
cmpeqss -64(%rdx,%rax,4), %xmm6

// CHECK: cmpeqss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0xc2,0x74,0x82,0x40,0x00]
cmpeqss 64(%rdx,%rax,4), %xmm6

// CHECK: cmpeqss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0xc2,0x74,0x02,0x40,0x00]
cmpeqss 64(%rdx,%rax), %xmm6

// CHECK: cmpeqss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0xc2,0x72,0x40,0x00]
cmpeqss 64(%rdx), %xmm6

// CHECK: cmpeqss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0xc2,0x32,0x00]
cmpeqss (%rdx), %xmm6

// CHECK: cmpeqss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0xc2,0xf6,0x00]
cmpeqss %xmm6, %xmm6

// CHECK: comiss 485498096, %xmm6
// CHECK: encoding: [0x0f,0x2f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
comiss 485498096, %xmm6

// CHECK: comiss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x2f,0x74,0x82,0xc0]
comiss -64(%rdx,%rax,4), %xmm6

// CHECK: comiss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x2f,0x74,0x82,0x40]
comiss 64(%rdx,%rax,4), %xmm6

// CHECK: comiss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x2f,0x74,0x02,0x40]
comiss 64(%rdx,%rax), %xmm6

// CHECK: comiss 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x2f,0x72,0x40]
comiss 64(%rdx), %xmm6

// CHECK: comiss (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x2f,0x32]
comiss (%rdx), %xmm6

// CHECK: comiss %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x2f,0xf6]
comiss %xmm6, %xmm6

// CHECK: cvtpi2ps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
cvtpi2ps 485498096, %xmm6

// CHECK: cvtpi2ps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x2a,0x74,0x82,0xc0]
cvtpi2ps -64(%rdx,%rax,4), %xmm6

// CHECK: cvtpi2ps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x2a,0x74,0x82,0x40]
cvtpi2ps 64(%rdx,%rax,4), %xmm6

// CHECK: cvtpi2ps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x2a,0x74,0x02,0x40]
cvtpi2ps 64(%rdx,%rax), %xmm6

// CHECK: cvtpi2ps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x2a,0x72,0x40]
cvtpi2ps 64(%rdx), %xmm6

// CHECK: cvtpi2ps %mm4, %xmm6
// CHECK: encoding: [0x0f,0x2a,0xf4]
cvtpi2ps %mm4, %xmm6

// CHECK: cvtpi2ps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x2a,0x32]
cvtpi2ps (%rdx), %xmm6

// CHECK: cvtps2pi 485498096, %mm4
// CHECK: encoding: [0x0f,0x2d,0x24,0x25,0xf0,0x1c,0xf0,0x1c]
cvtps2pi 485498096, %mm4

// CHECK: cvtps2pi 64(%rdx), %mm4
// CHECK: encoding: [0x0f,0x2d,0x62,0x40]
cvtps2pi 64(%rdx), %mm4

// CHECK: cvtps2pi -64(%rdx,%rax,4), %mm4
// CHECK: encoding: [0x0f,0x2d,0x64,0x82,0xc0]
cvtps2pi -64(%rdx,%rax,4), %mm4

// CHECK: cvtps2pi 64(%rdx,%rax,4), %mm4
// CHECK: encoding: [0x0f,0x2d,0x64,0x82,0x40]
cvtps2pi 64(%rdx,%rax,4), %mm4

// CHECK: cvtps2pi 64(%rdx,%rax), %mm4
// CHECK: encoding: [0x0f,0x2d,0x64,0x02,0x40]
cvtps2pi 64(%rdx,%rax), %mm4

// CHECK: cvtps2pi (%rdx), %mm4
// CHECK: encoding: [0x0f,0x2d,0x22]
cvtps2pi (%rdx), %mm4

// CHECK: cvtps2pi %xmm6, %mm4
// CHECK: encoding: [0x0f,0x2d,0xe6]
cvtps2pi %xmm6, %mm4

// CHECK: cvtsi2ssl 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
cvtsi2ssl 485498096, %xmm6

// CHECK: cvtsi2ssl -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x2a,0x74,0x82,0xc0]
cvtsi2ssl -64(%rdx,%rax,4), %xmm6

// CHECK: cvtsi2ssl 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x2a,0x74,0x82,0x40]
cvtsi2ssl 64(%rdx,%rax,4), %xmm6

// CHECK: cvtsi2ssl 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x2a,0x74,0x02,0x40]
cvtsi2ssl 64(%rdx,%rax), %xmm6

// CHECK: cvtsi2ssl 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x2a,0x72,0x40]
cvtsi2ssl 64(%rdx), %xmm6

// CHECK: cvtsi2ss %r13d, %xmm6
// CHECK: encoding: [0xf3,0x41,0x0f,0x2a,0xf5]
cvtsi2ssl %r13d, %xmm6

// CHECK: cvtsi2ssl (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x2a,0x32]
cvtsi2ssl (%rdx), %xmm6

// CHECK: cvtsi2ssq 485498096, %xmm6
// CHECK: encoding: [0xf3,0x48,0x0f,0x2a,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
cvtsi2ssq 485498096, %xmm6

// CHECK: cvtsi2ssq -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x48,0x0f,0x2a,0x74,0x82,0xc0]
cvtsi2ssq -64(%rdx,%rax,4), %xmm6

// CHECK: cvtsi2ssq 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x48,0x0f,0x2a,0x74,0x82,0x40]
cvtsi2ssq 64(%rdx,%rax,4), %xmm6

// CHECK: cvtsi2ssq 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x48,0x0f,0x2a,0x74,0x02,0x40]
cvtsi2ssq 64(%rdx,%rax), %xmm6

// CHECK: cvtsi2ssq 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x48,0x0f,0x2a,0x72,0x40]
cvtsi2ssq 64(%rdx), %xmm6

// CHECK: cvtsi2ss %r15, %xmm6
// CHECK: encoding: [0xf3,0x49,0x0f,0x2a,0xf7]
cvtsi2ssq %r15, %xmm6

// CHECK: cvtsi2ssq (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x48,0x0f,0x2a,0x32]
cvtsi2ssq (%rdx), %xmm6

// CHECK: cvtss2si 485498096, %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2d,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
cvtss2si 485498096, %r13d

// CHECK: cvtss2si 485498096, %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2d,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
cvtss2si 485498096, %r15

// CHECK: cvtss2si 64(%rdx), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2d,0x6a,0x40]
cvtss2si 64(%rdx), %r13d

// CHECK: cvtss2si 64(%rdx), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2d,0x7a,0x40]
cvtss2si 64(%rdx), %r15

// CHECK: cvtss2si -64(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2d,0x6c,0x82,0xc0]
cvtss2si -64(%rdx,%rax,4), %r13d

// CHECK: cvtss2si 64(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2d,0x6c,0x82,0x40]
cvtss2si 64(%rdx,%rax,4), %r13d

// CHECK: cvtss2si -64(%rdx,%rax,4), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2d,0x7c,0x82,0xc0]
cvtss2si -64(%rdx,%rax,4), %r15

// CHECK: cvtss2si 64(%rdx,%rax,4), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2d,0x7c,0x82,0x40]
cvtss2si 64(%rdx,%rax,4), %r15

// CHECK: cvtss2si 64(%rdx,%rax), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2d,0x6c,0x02,0x40]
cvtss2si 64(%rdx,%rax), %r13d

// CHECK: cvtss2si 64(%rdx,%rax), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2d,0x7c,0x02,0x40]
cvtss2si 64(%rdx,%rax), %r15

// CHECK: cvtss2si (%rdx), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2d,0x2a]
cvtss2si (%rdx), %r13d

// CHECK: cvtss2si (%rdx), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2d,0x3a]
cvtss2si (%rdx), %r15

// CHECK: cvtss2si %xmm6, %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2d,0xee]
cvtss2si %xmm6, %r13d

// CHECK: cvtss2si %xmm6, %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2d,0xfe]
cvtss2si %xmm6, %r15

// CHECK: cvttps2pi 485498096, %mm4
// CHECK: encoding: [0x0f,0x2c,0x24,0x25,0xf0,0x1c,0xf0,0x1c]
cvttps2pi 485498096, %mm4

// CHECK: cvttps2pi 64(%rdx), %mm4
// CHECK: encoding: [0x0f,0x2c,0x62,0x40]
cvttps2pi 64(%rdx), %mm4

// CHECK: cvttps2pi -64(%rdx,%rax,4), %mm4
// CHECK: encoding: [0x0f,0x2c,0x64,0x82,0xc0]
cvttps2pi -64(%rdx,%rax,4), %mm4

// CHECK: cvttps2pi 64(%rdx,%rax,4), %mm4
// CHECK: encoding: [0x0f,0x2c,0x64,0x82,0x40]
cvttps2pi 64(%rdx,%rax,4), %mm4

// CHECK: cvttps2pi 64(%rdx,%rax), %mm4
// CHECK: encoding: [0x0f,0x2c,0x64,0x02,0x40]
cvttps2pi 64(%rdx,%rax), %mm4

// CHECK: cvttps2pi (%rdx), %mm4
// CHECK: encoding: [0x0f,0x2c,0x22]
cvttps2pi (%rdx), %mm4

// CHECK: cvttps2pi %xmm6, %mm4
// CHECK: encoding: [0x0f,0x2c,0xe6]
cvttps2pi %xmm6, %mm4

// CHECK: cvttss2si 485498096, %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2c,0x2c,0x25,0xf0,0x1c,0xf0,0x1c]
cvttss2si 485498096, %r13d

// CHECK: cvttss2si 485498096, %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2c,0x3c,0x25,0xf0,0x1c,0xf0,0x1c]
cvttss2si 485498096, %r15

// CHECK: cvttss2si 64(%rdx), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2c,0x6a,0x40]
cvttss2si 64(%rdx), %r13d

// CHECK: cvttss2si 64(%rdx), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2c,0x7a,0x40]
cvttss2si 64(%rdx), %r15

// CHECK: cvttss2si -64(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2c,0x6c,0x82,0xc0]
cvttss2si -64(%rdx,%rax,4), %r13d

// CHECK: cvttss2si 64(%rdx,%rax,4), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2c,0x6c,0x82,0x40]
cvttss2si 64(%rdx,%rax,4), %r13d

// CHECK: cvttss2si -64(%rdx,%rax,4), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2c,0x7c,0x82,0xc0]
cvttss2si -64(%rdx,%rax,4), %r15

// CHECK: cvttss2si 64(%rdx,%rax,4), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2c,0x7c,0x82,0x40]
cvttss2si 64(%rdx,%rax,4), %r15

// CHECK: cvttss2si 64(%rdx,%rax), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2c,0x6c,0x02,0x40]
cvttss2si 64(%rdx,%rax), %r13d

// CHECK: cvttss2si 64(%rdx,%rax), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2c,0x7c,0x02,0x40]
cvttss2si 64(%rdx,%rax), %r15

// CHECK: cvttss2si (%rdx), %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2c,0x2a]
cvttss2si (%rdx), %r13d

// CHECK: cvttss2si (%rdx), %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2c,0x3a]
cvttss2si (%rdx), %r15

// CHECK: cvttss2si %xmm6, %r13d
// CHECK: encoding: [0xf3,0x44,0x0f,0x2c,0xee]
cvttss2si %xmm6, %r13d

// CHECK: cvttss2si %xmm6, %r15
// CHECK: encoding: [0xf3,0x4c,0x0f,0x2c,0xfe]
cvttss2si %xmm6, %r15

// CHECK: divps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x5e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
divps 485498096, %xmm6

// CHECK: divps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5e,0x74,0x82,0xc0]
divps -64(%rdx,%rax,4), %xmm6

// CHECK: divps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5e,0x74,0x82,0x40]
divps 64(%rdx,%rax,4), %xmm6

// CHECK: divps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x5e,0x74,0x02,0x40]
divps 64(%rdx,%rax), %xmm6

// CHECK: divps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5e,0x72,0x40]
divps 64(%rdx), %xmm6

// CHECK: divps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5e,0x32]
divps (%rdx), %xmm6

// CHECK: divps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x5e,0xf6]
divps %xmm6, %xmm6

// CHECK: divss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
divss 485498096, %xmm6

// CHECK: divss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5e,0x74,0x82,0xc0]
divss -64(%rdx,%rax,4), %xmm6

// CHECK: divss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5e,0x74,0x82,0x40]
divss 64(%rdx,%rax,4), %xmm6

// CHECK: divss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5e,0x74,0x02,0x40]
divss 64(%rdx,%rax), %xmm6

// CHECK: divss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5e,0x72,0x40]
divss 64(%rdx), %xmm6

// CHECK: divss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5e,0x32]
divss (%rdx), %xmm6

// CHECK: divss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5e,0xf6]
divss %xmm6, %xmm6

// CHECK: maxps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x5f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
maxps 485498096, %xmm6

// CHECK: maxps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5f,0x74,0x82,0xc0]
maxps -64(%rdx,%rax,4), %xmm6

// CHECK: maxps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5f,0x74,0x82,0x40]
maxps 64(%rdx,%rax,4), %xmm6

// CHECK: maxps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x5f,0x74,0x02,0x40]
maxps 64(%rdx,%rax), %xmm6

// CHECK: maxps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5f,0x72,0x40]
maxps 64(%rdx), %xmm6

// CHECK: maxps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5f,0x32]
maxps (%rdx), %xmm6

// CHECK: maxps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x5f,0xf6]
maxps %xmm6, %xmm6

// CHECK: maxss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5f,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
maxss 485498096, %xmm6

// CHECK: maxss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5f,0x74,0x82,0xc0]
maxss -64(%rdx,%rax,4), %xmm6

// CHECK: maxss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5f,0x74,0x82,0x40]
maxss 64(%rdx,%rax,4), %xmm6

// CHECK: maxss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5f,0x74,0x02,0x40]
maxss 64(%rdx,%rax), %xmm6

// CHECK: maxss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5f,0x72,0x40]
maxss 64(%rdx), %xmm6

// CHECK: maxss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5f,0x32]
maxss (%rdx), %xmm6

// CHECK: maxss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5f,0xf6]
maxss %xmm6, %xmm6

// CHECK: minps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x5d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
minps 485498096, %xmm6

// CHECK: minps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5d,0x74,0x82,0xc0]
minps -64(%rdx,%rax,4), %xmm6

// CHECK: minps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5d,0x74,0x82,0x40]
minps 64(%rdx,%rax,4), %xmm6

// CHECK: minps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x5d,0x74,0x02,0x40]
minps 64(%rdx,%rax), %xmm6

// CHECK: minps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5d,0x72,0x40]
minps 64(%rdx), %xmm6

// CHECK: minps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5d,0x32]
minps (%rdx), %xmm6

// CHECK: minps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x5d,0xf6]
minps %xmm6, %xmm6

// CHECK: minss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
minss 485498096, %xmm6

// CHECK: minss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5d,0x74,0x82,0xc0]
minss -64(%rdx,%rax,4), %xmm6

// CHECK: minss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5d,0x74,0x82,0x40]
minss 64(%rdx,%rax,4), %xmm6

// CHECK: minss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5d,0x74,0x02,0x40]
minss 64(%rdx,%rax), %xmm6

// CHECK: minss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5d,0x72,0x40]
minss 64(%rdx), %xmm6

// CHECK: minss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5d,0x32]
minss (%rdx), %xmm6

// CHECK: minss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5d,0xf6]
minss %xmm6, %xmm6

// CHECK: movaps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x28,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movaps 485498096, %xmm6

// CHECK: movaps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x28,0x74,0x82,0xc0]
movaps -64(%rdx,%rax,4), %xmm6

// CHECK: movaps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x28,0x74,0x82,0x40]
movaps 64(%rdx,%rax,4), %xmm6

// CHECK: movaps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x28,0x74,0x02,0x40]
movaps 64(%rdx,%rax), %xmm6

// CHECK: movaps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x28,0x72,0x40]
movaps 64(%rdx), %xmm6

// CHECK: movaps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x28,0x32]
movaps (%rdx), %xmm6

// CHECK: movaps %xmm6, 485498096
// CHECK: encoding: [0x0f,0x29,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movaps %xmm6, 485498096

// CHECK: movaps %xmm6, 64(%rdx)
// CHECK: encoding: [0x0f,0x29,0x72,0x40]
movaps %xmm6, 64(%rdx)

// CHECK: movaps %xmm6, 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0x29,0x74,0x02,0x40]
movaps %xmm6, 64(%rdx,%rax)

// CHECK: movaps %xmm6, -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x29,0x74,0x82,0xc0]
movaps %xmm6, -64(%rdx,%rax,4)

// CHECK: movaps %xmm6, 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x29,0x74,0x82,0x40]
movaps %xmm6, 64(%rdx,%rax,4)

// CHECK: movaps %xmm6, (%rdx)
// CHECK: encoding: [0x0f,0x29,0x32]
movaps %xmm6, (%rdx)

// CHECK: movaps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x28,0xf6]
movaps %xmm6, %xmm6

// CHECK: movhlps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x12,0xf6]
movhlps %xmm6, %xmm6

// CHECK: movhps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x16,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movhps 485498096, %xmm6

// CHECK: movhps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x16,0x74,0x82,0xc0]
movhps -64(%rdx,%rax,4), %xmm6

// CHECK: movhps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x16,0x74,0x82,0x40]
movhps 64(%rdx,%rax,4), %xmm6

// CHECK: movhps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x16,0x74,0x02,0x40]
movhps 64(%rdx,%rax), %xmm6

// CHECK: movhps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x16,0x72,0x40]
movhps 64(%rdx), %xmm6

// CHECK: movhps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x16,0x32]
movhps (%rdx), %xmm6

// CHECK: movhps %xmm6, 485498096
// CHECK: encoding: [0x0f,0x17,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movhps %xmm6, 485498096

// CHECK: movhps %xmm6, 64(%rdx)
// CHECK: encoding: [0x0f,0x17,0x72,0x40]
movhps %xmm6, 64(%rdx)

// CHECK: movhps %xmm6, 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0x17,0x74,0x02,0x40]
movhps %xmm6, 64(%rdx,%rax)

// CHECK: movhps %xmm6, -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x17,0x74,0x82,0xc0]
movhps %xmm6, -64(%rdx,%rax,4)

// CHECK: movhps %xmm6, 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x17,0x74,0x82,0x40]
movhps %xmm6, 64(%rdx,%rax,4)

// CHECK: movhps %xmm6, (%rdx)
// CHECK: encoding: [0x0f,0x17,0x32]
movhps %xmm6, (%rdx)

// CHECK: movlhps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x16,0xf6]
movlhps %xmm6, %xmm6

// CHECK: movlps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movlps 485498096, %xmm6

// CHECK: movlps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x12,0x74,0x82,0xc0]
movlps -64(%rdx,%rax,4), %xmm6

// CHECK: movlps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x12,0x74,0x82,0x40]
movlps 64(%rdx,%rax,4), %xmm6

// CHECK: movlps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x12,0x74,0x02,0x40]
movlps 64(%rdx,%rax), %xmm6

// CHECK: movlps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x12,0x72,0x40]
movlps 64(%rdx), %xmm6

// CHECK: movlps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x12,0x32]
movlps (%rdx), %xmm6

// CHECK: movlps %xmm6, 485498096
// CHECK: encoding: [0x0f,0x13,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movlps %xmm6, 485498096

// CHECK: movlps %xmm6, 64(%rdx)
// CHECK: encoding: [0x0f,0x13,0x72,0x40]
movlps %xmm6, 64(%rdx)

// CHECK: movlps %xmm6, 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0x13,0x74,0x02,0x40]
movlps %xmm6, 64(%rdx,%rax)

// CHECK: movlps %xmm6, -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x13,0x74,0x82,0xc0]
movlps %xmm6, -64(%rdx,%rax,4)

// CHECK: movlps %xmm6, 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x13,0x74,0x82,0x40]
movlps %xmm6, 64(%rdx,%rax,4)

// CHECK: movlps %xmm6, (%rdx)
// CHECK: encoding: [0x0f,0x13,0x32]
movlps %xmm6, (%rdx)

// CHECK: movmskps %xmm6, %r13d
// CHECK: encoding: [0x44,0x0f,0x50,0xee]
movmskps %xmm6, %r13d

// CHECK: movntps %xmm6, 485498096
// CHECK: encoding: [0x0f,0x2b,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movntps %xmm6, 485498096

// CHECK: movntps %xmm6, 64(%rdx)
// CHECK: encoding: [0x0f,0x2b,0x72,0x40]
movntps %xmm6, 64(%rdx)

// CHECK: movntps %xmm6, 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0x2b,0x74,0x02,0x40]
movntps %xmm6, 64(%rdx,%rax)

// CHECK: movntps %xmm6, -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x2b,0x74,0x82,0xc0]
movntps %xmm6, -64(%rdx,%rax,4)

// CHECK: movntps %xmm6, 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x2b,0x74,0x82,0x40]
movntps %xmm6, 64(%rdx,%rax,4)

// CHECK: movntps %xmm6, (%rdx)
// CHECK: encoding: [0x0f,0x2b,0x32]
movntps %xmm6, (%rdx)

// CHECK: movss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x10,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movss 485498096, %xmm6

// CHECK: movss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x10,0x74,0x82,0xc0]
movss -64(%rdx,%rax,4), %xmm6

// CHECK: movss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x10,0x74,0x82,0x40]
movss 64(%rdx,%rax,4), %xmm6

// CHECK: movss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x10,0x74,0x02,0x40]
movss 64(%rdx,%rax), %xmm6

// CHECK: movss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x10,0x72,0x40]
movss 64(%rdx), %xmm6

// CHECK: movss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x10,0x32]
movss (%rdx), %xmm6

// CHECK: movss %xmm6, 485498096
// CHECK: encoding: [0xf3,0x0f,0x11,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movss %xmm6, 485498096

// CHECK: movss %xmm6, 64(%rdx)
// CHECK: encoding: [0xf3,0x0f,0x11,0x72,0x40]
movss %xmm6, 64(%rdx)

// CHECK: movss %xmm6, 64(%rdx,%rax)
// CHECK: encoding: [0xf3,0x0f,0x11,0x74,0x02,0x40]
movss %xmm6, 64(%rdx,%rax)

// CHECK: movss %xmm6, -64(%rdx,%rax,4)
// CHECK: encoding: [0xf3,0x0f,0x11,0x74,0x82,0xc0]
movss %xmm6, -64(%rdx,%rax,4)

// CHECK: movss %xmm6, 64(%rdx,%rax,4)
// CHECK: encoding: [0xf3,0x0f,0x11,0x74,0x82,0x40]
movss %xmm6, 64(%rdx,%rax,4)

// CHECK: movss %xmm6, (%rdx)
// CHECK: encoding: [0xf3,0x0f,0x11,0x32]
movss %xmm6, (%rdx)

// CHECK: movss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x10,0xf6]
movss %xmm6, %xmm6

// CHECK: movups 485498096, %xmm6
// CHECK: encoding: [0x0f,0x10,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movups 485498096, %xmm6

// CHECK: movups -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x10,0x74,0x82,0xc0]
movups -64(%rdx,%rax,4), %xmm6

// CHECK: movups 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x10,0x74,0x82,0x40]
movups 64(%rdx,%rax,4), %xmm6

// CHECK: movups 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x10,0x74,0x02,0x40]
movups 64(%rdx,%rax), %xmm6

// CHECK: movups 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x10,0x72,0x40]
movups 64(%rdx), %xmm6

// CHECK: movups (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x10,0x32]
movups (%rdx), %xmm6

// CHECK: movups %xmm6, 485498096
// CHECK: encoding: [0x0f,0x11,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movups %xmm6, 485498096

// CHECK: movups %xmm6, 64(%rdx)
// CHECK: encoding: [0x0f,0x11,0x72,0x40]
movups %xmm6, 64(%rdx)

// CHECK: movups %xmm6, 64(%rdx,%rax)
// CHECK: encoding: [0x0f,0x11,0x74,0x02,0x40]
movups %xmm6, 64(%rdx,%rax)

// CHECK: movups %xmm6, -64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x11,0x74,0x82,0xc0]
movups %xmm6, -64(%rdx,%rax,4)

// CHECK: movups %xmm6, 64(%rdx,%rax,4)
// CHECK: encoding: [0x0f,0x11,0x74,0x82,0x40]
movups %xmm6, 64(%rdx,%rax,4)

// CHECK: movups %xmm6, (%rdx)
// CHECK: encoding: [0x0f,0x11,0x32]
movups %xmm6, (%rdx)

// CHECK: movups %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x10,0xf6]
movups %xmm6, %xmm6

// CHECK: mulps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x59,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
mulps 485498096, %xmm6

// CHECK: mulps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x59,0x74,0x82,0xc0]
mulps -64(%rdx,%rax,4), %xmm6

// CHECK: mulps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x59,0x74,0x82,0x40]
mulps 64(%rdx,%rax,4), %xmm6

// CHECK: mulps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x59,0x74,0x02,0x40]
mulps 64(%rdx,%rax), %xmm6

// CHECK: mulps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x59,0x72,0x40]
mulps 64(%rdx), %xmm6

// CHECK: mulps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x59,0x32]
mulps (%rdx), %xmm6

// CHECK: mulps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x59,0xf6]
mulps %xmm6, %xmm6

// CHECK: mulss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x59,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
mulss 485498096, %xmm6

// CHECK: mulss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x59,0x74,0x82,0xc0]
mulss -64(%rdx,%rax,4), %xmm6

// CHECK: mulss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x59,0x74,0x82,0x40]
mulss 64(%rdx,%rax,4), %xmm6

// CHECK: mulss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x59,0x74,0x02,0x40]
mulss 64(%rdx,%rax), %xmm6

// CHECK: mulss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x59,0x72,0x40]
mulss 64(%rdx), %xmm6

// CHECK: mulss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x59,0x32]
mulss (%rdx), %xmm6

// CHECK: mulss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x59,0xf6]
mulss %xmm6, %xmm6

// CHECK: orps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x56,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
orps 485498096, %xmm6

// CHECK: orps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x56,0x74,0x82,0xc0]
orps -64(%rdx,%rax,4), %xmm6

// CHECK: orps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x56,0x74,0x82,0x40]
orps 64(%rdx,%rax,4), %xmm6

// CHECK: orps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x56,0x74,0x02,0x40]
orps 64(%rdx,%rax), %xmm6

// CHECK: orps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x56,0x72,0x40]
orps 64(%rdx), %xmm6

// CHECK: orps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x56,0x32]
orps (%rdx), %xmm6

// CHECK: orps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x56,0xf6]
orps %xmm6, %xmm6

// CHECK: pextrw $0, %mm4, %r13d
// CHECK: encoding: [0x44,0x0f,0xc5,0xec,0x00]
pextrw $0, %mm4, %r13d

// CHECK: pinsrw $0, -485498096(%rdx,%rax,4), %mm4
// CHECK: encoding: [0x0f,0xc4,0xa4,0x82,0x10,0xe3,0x0f,0xe3,0x00]
pinsrw $0, -485498096(%rdx,%rax,4), %mm4

// CHECK: pinsrw $0, 485498096(%rdx,%rax,4), %mm4
// CHECK: encoding: [0x0f,0xc4,0xa4,0x82,0xf0,0x1c,0xf0,0x1c,0x00]
pinsrw $0, 485498096(%rdx,%rax,4), %mm4

// CHECK: pinsrw $0, 485498096(%rdx), %mm4
// CHECK: encoding: [0x0f,0xc4,0xa2,0xf0,0x1c,0xf0,0x1c,0x00]
pinsrw $0, 485498096(%rdx), %mm4

// CHECK: pinsrw $0, 485498096, %mm4
// CHECK: encoding: [0x0f,0xc4,0x24,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
pinsrw $0, 485498096, %mm4

// CHECK: pinsrw $0, 64(%rdx,%rax), %mm4
// CHECK: encoding: [0x0f,0xc4,0x64,0x02,0x40,0x00]
pinsrw $0, 64(%rdx,%rax), %mm4

// CHECK: pinsrw $0, (%rdx), %mm4
// CHECK: encoding: [0x0f,0xc4,0x22,0x00]
pinsrw $0, (%rdx), %mm4

// CHECK: pinsrw $0, %r13d, %mm4
// CHECK: encoding: [0x41,0x0f,0xc4,0xe5,0x00]
pinsrw $0, %r13d, %mm4

// CHECK: pmovmskb %mm4, %r13d
// CHECK: encoding: [0x44,0x0f,0xd7,0xec]
pmovmskb %mm4, %r13d

// CHECK: pmovmskb %mm4, %r13d
// CHECK: encoding: [0x44,0x0f,0xd7,0xec]
pmovmskb %mm4, %r13

// CHECK: rcpps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x53,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
rcpps 485498096, %xmm6

// CHECK: rcpps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x53,0x74,0x82,0xc0]
rcpps -64(%rdx,%rax,4), %xmm6

// CHECK: rcpps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x53,0x74,0x82,0x40]
rcpps 64(%rdx,%rax,4), %xmm6

// CHECK: rcpps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x53,0x74,0x02,0x40]
rcpps 64(%rdx,%rax), %xmm6

// CHECK: rcpps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x53,0x72,0x40]
rcpps 64(%rdx), %xmm6

// CHECK: rcpps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x53,0x32]
rcpps (%rdx), %xmm6

// CHECK: rcpps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x53,0xf6]
rcpps %xmm6, %xmm6

// CHECK: rcpss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x53,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
rcpss 485498096, %xmm6

// CHECK: rcpss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x53,0x74,0x82,0xc0]
rcpss -64(%rdx,%rax,4), %xmm6

// CHECK: rcpss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x53,0x74,0x82,0x40]
rcpss 64(%rdx,%rax,4), %xmm6

// CHECK: rcpss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x53,0x74,0x02,0x40]
rcpss 64(%rdx,%rax), %xmm6

// CHECK: rcpss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x53,0x72,0x40]
rcpss 64(%rdx), %xmm6

// CHECK: rcpss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x53,0x32]
rcpss (%rdx), %xmm6

// CHECK: rcpss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x53,0xf6]
rcpss %xmm6, %xmm6

// CHECK: rsqrtps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x52,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
rsqrtps 485498096, %xmm6

// CHECK: rsqrtps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x52,0x74,0x82,0xc0]
rsqrtps -64(%rdx,%rax,4), %xmm6

// CHECK: rsqrtps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x52,0x74,0x82,0x40]
rsqrtps 64(%rdx,%rax,4), %xmm6

// CHECK: rsqrtps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x52,0x74,0x02,0x40]
rsqrtps 64(%rdx,%rax), %xmm6

// CHECK: rsqrtps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x52,0x72,0x40]
rsqrtps 64(%rdx), %xmm6

// CHECK: rsqrtps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x52,0x32]
rsqrtps (%rdx), %xmm6

// CHECK: rsqrtps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x52,0xf6]
rsqrtps %xmm6, %xmm6

// CHECK: rsqrtss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x52,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
rsqrtss 485498096, %xmm6

// CHECK: rsqrtss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x52,0x74,0x82,0xc0]
rsqrtss -64(%rdx,%rax,4), %xmm6

// CHECK: rsqrtss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x52,0x74,0x82,0x40]
rsqrtss 64(%rdx,%rax,4), %xmm6

// CHECK: rsqrtss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x52,0x74,0x02,0x40]
rsqrtss 64(%rdx,%rax), %xmm6

// CHECK: rsqrtss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x52,0x72,0x40]
rsqrtss 64(%rdx), %xmm6

// CHECK: rsqrtss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x52,0x32]
rsqrtss (%rdx), %xmm6

// CHECK: rsqrtss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x52,0xf6]
rsqrtss %xmm6, %xmm6

// CHECK: sfence
// CHECK: encoding: [0x0f,0xae,0xf8]
sfence

// CHECK: shufps $0, 485498096, %xmm6
// CHECK: encoding: [0x0f,0xc6,0x34,0x25,0xf0,0x1c,0xf0,0x1c,0x00]
shufps $0, 485498096, %xmm6

// CHECK: shufps $0, -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0xc6,0x74,0x82,0xc0,0x00]
shufps $0, -64(%rdx,%rax,4), %xmm6

// CHECK: shufps $0, 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0xc6,0x74,0x82,0x40,0x00]
shufps $0, 64(%rdx,%rax,4), %xmm6

// CHECK: shufps $0, 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0xc6,0x74,0x02,0x40,0x00]
shufps $0, 64(%rdx,%rax), %xmm6

// CHECK: shufps $0, 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0xc6,0x72,0x40,0x00]
shufps $0, 64(%rdx), %xmm6

// CHECK: shufps $0, (%rdx), %xmm6
// CHECK: encoding: [0x0f,0xc6,0x32,0x00]
shufps $0, (%rdx), %xmm6

// CHECK: shufps $0, %xmm6, %xmm6
// CHECK: encoding: [0x0f,0xc6,0xf6,0x00]
shufps $0, %xmm6, %xmm6

// CHECK: sqrtps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x51,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
sqrtps 485498096, %xmm6

// CHECK: sqrtps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x51,0x74,0x82,0xc0]
sqrtps -64(%rdx,%rax,4), %xmm6

// CHECK: sqrtps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x51,0x74,0x82,0x40]
sqrtps 64(%rdx,%rax,4), %xmm6

// CHECK: sqrtps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x51,0x74,0x02,0x40]
sqrtps 64(%rdx,%rax), %xmm6

// CHECK: sqrtps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x51,0x72,0x40]
sqrtps 64(%rdx), %xmm6

// CHECK: sqrtps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x51,0x32]
sqrtps (%rdx), %xmm6

// CHECK: sqrtps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x51,0xf6]
sqrtps %xmm6, %xmm6

// CHECK: sqrtss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x51,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
sqrtss 485498096, %xmm6

// CHECK: sqrtss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x51,0x74,0x82,0xc0]
sqrtss -64(%rdx,%rax,4), %xmm6

// CHECK: sqrtss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x51,0x74,0x82,0x40]
sqrtss 64(%rdx,%rax,4), %xmm6

// CHECK: sqrtss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x51,0x74,0x02,0x40]
sqrtss 64(%rdx,%rax), %xmm6

// CHECK: sqrtss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x51,0x72,0x40]
sqrtss 64(%rdx), %xmm6

// CHECK: sqrtss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x51,0x32]
sqrtss (%rdx), %xmm6

// CHECK: sqrtss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x51,0xf6]
sqrtss %xmm6, %xmm6

// CHECK: subps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x5c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
subps 485498096, %xmm6

// CHECK: subps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5c,0x74,0x82,0xc0]
subps -64(%rdx,%rax,4), %xmm6

// CHECK: subps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x5c,0x74,0x82,0x40]
subps 64(%rdx,%rax,4), %xmm6

// CHECK: subps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x5c,0x74,0x02,0x40]
subps 64(%rdx,%rax), %xmm6

// CHECK: subps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5c,0x72,0x40]
subps 64(%rdx), %xmm6

// CHECK: subps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x5c,0x32]
subps (%rdx), %xmm6

// CHECK: subps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x5c,0xf6]
subps %xmm6, %xmm6

// CHECK: subss 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
subss 485498096, %xmm6

// CHECK: subss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5c,0x74,0x82,0xc0]
subss -64(%rdx,%rax,4), %xmm6

// CHECK: subss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5c,0x74,0x82,0x40]
subss 64(%rdx,%rax,4), %xmm6

// CHECK: subss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5c,0x74,0x02,0x40]
subss 64(%rdx,%rax), %xmm6

// CHECK: subss 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5c,0x72,0x40]
subss 64(%rdx), %xmm6

// CHECK: subss (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5c,0x32]
subss (%rdx), %xmm6

// CHECK: subss %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x5c,0xf6]
subss %xmm6, %xmm6

// CHECK: ucomiss 485498096, %xmm6
// CHECK: encoding: [0x0f,0x2e,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
ucomiss 485498096, %xmm6

// CHECK: ucomiss -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x2e,0x74,0x82,0xc0]
ucomiss -64(%rdx,%rax,4), %xmm6

// CHECK: ucomiss 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x2e,0x74,0x82,0x40]
ucomiss 64(%rdx,%rax,4), %xmm6

// CHECK: ucomiss 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x2e,0x74,0x02,0x40]
ucomiss 64(%rdx,%rax), %xmm6

// CHECK: ucomiss 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x2e,0x72,0x40]
ucomiss 64(%rdx), %xmm6

// CHECK: ucomiss (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x2e,0x32]
ucomiss (%rdx), %xmm6

// CHECK: ucomiss %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x2e,0xf6]
ucomiss %xmm6, %xmm6

// CHECK: unpckhps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x15,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
unpckhps 485498096, %xmm6

// CHECK: unpckhps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x15,0x74,0x82,0xc0]
unpckhps -64(%rdx,%rax,4), %xmm6

// CHECK: unpckhps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x15,0x74,0x82,0x40]
unpckhps 64(%rdx,%rax,4), %xmm6

// CHECK: unpckhps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x15,0x74,0x02,0x40]
unpckhps 64(%rdx,%rax), %xmm6

// CHECK: unpckhps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x15,0x72,0x40]
unpckhps 64(%rdx), %xmm6

// CHECK: unpckhps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x15,0x32]
unpckhps (%rdx), %xmm6

// CHECK: unpckhps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x15,0xf6]
unpckhps %xmm6, %xmm6

// CHECK: unpcklps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x14,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
unpcklps 485498096, %xmm6

// CHECK: unpcklps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x14,0x74,0x82,0xc0]
unpcklps -64(%rdx,%rax,4), %xmm6

// CHECK: unpcklps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x14,0x74,0x82,0x40]
unpcklps 64(%rdx,%rax,4), %xmm6

// CHECK: unpcklps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x14,0x74,0x02,0x40]
unpcklps 64(%rdx,%rax), %xmm6

// CHECK: unpcklps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x14,0x72,0x40]
unpcklps 64(%rdx), %xmm6

// CHECK: unpcklps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x14,0x32]
unpcklps (%rdx), %xmm6

// CHECK: unpcklps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x14,0xf6]
unpcklps %xmm6, %xmm6

// CHECK: xorps 485498096, %xmm6
// CHECK: encoding: [0x0f,0x57,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
xorps 485498096, %xmm6

// CHECK: xorps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x57,0x74,0x82,0xc0]
xorps -64(%rdx,%rax,4), %xmm6

// CHECK: xorps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x0f,0x57,0x74,0x82,0x40]
xorps 64(%rdx,%rax,4), %xmm6

// CHECK: xorps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x0f,0x57,0x74,0x02,0x40]
xorps 64(%rdx,%rax), %xmm6

// CHECK: xorps 64(%rdx), %xmm6
// CHECK: encoding: [0x0f,0x57,0x72,0x40]
xorps 64(%rdx), %xmm6

// CHECK: xorps (%rdx), %xmm6
// CHECK: encoding: [0x0f,0x57,0x32]
xorps (%rdx), %xmm6

// CHECK: xorps %xmm6, %xmm6
// CHECK: encoding: [0x0f,0x57,0xf6]
xorps %xmm6, %xmm6

