// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: addsubpd 485498096, %xmm6
// CHECK: encoding: [0x66,0x0f,0xd0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
addsubpd 485498096, %xmm6

// CHECK: addsubpd -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x66,0x0f,0xd0,0x74,0x82,0xc0]
addsubpd -64(%rdx,%rax,4), %xmm6

// CHECK: addsubpd 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x66,0x0f,0xd0,0x74,0x82,0x40]
addsubpd 64(%rdx,%rax,4), %xmm6

// CHECK: addsubpd 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x66,0x0f,0xd0,0x74,0x02,0x40]
addsubpd 64(%rdx,%rax), %xmm6

// CHECK: addsubpd 64(%rdx), %xmm6
// CHECK: encoding: [0x66,0x0f,0xd0,0x72,0x40]
addsubpd 64(%rdx), %xmm6

// CHECK: addsubpd (%rdx), %xmm6
// CHECK: encoding: [0x66,0x0f,0xd0,0x32]
addsubpd (%rdx), %xmm6

// CHECK: addsubpd %xmm6, %xmm6
// CHECK: encoding: [0x66,0x0f,0xd0,0xf6]
addsubpd %xmm6, %xmm6

// CHECK: addsubps 485498096, %xmm6
// CHECK: encoding: [0xf2,0x0f,0xd0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
addsubps 485498096, %xmm6

// CHECK: addsubps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xd0,0x74,0x82,0xc0]
addsubps -64(%rdx,%rax,4), %xmm6

// CHECK: addsubps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xd0,0x74,0x82,0x40]
addsubps 64(%rdx,%rax,4), %xmm6

// CHECK: addsubps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xd0,0x74,0x02,0x40]
addsubps 64(%rdx,%rax), %xmm6

// CHECK: addsubps 64(%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xd0,0x72,0x40]
addsubps 64(%rdx), %xmm6

// CHECK: addsubps (%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xd0,0x32]
addsubps (%rdx), %xmm6

// CHECK: addsubps %xmm6, %xmm6
// CHECK: encoding: [0xf2,0x0f,0xd0,0xf6]
addsubps %xmm6, %xmm6

// CHECK: fisttpl 485498096
// CHECK: encoding: [0xdb,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
fisttpl 485498096

// CHECK: fisttpl 64(%rdx)
// CHECK: encoding: [0xdb,0x4a,0x40]
fisttpl 64(%rdx)

// CHECK: fisttpl -64(%rdx,%rax,4)
// CHECK: encoding: [0xdb,0x4c,0x82,0xc0]
fisttpl -64(%rdx,%rax,4)

// CHECK: fisttpl 64(%rdx,%rax,4)
// CHECK: encoding: [0xdb,0x4c,0x82,0x40]
fisttpl 64(%rdx,%rax,4)

// CHECK: fisttpl 64(%rdx,%rax)
// CHECK: encoding: [0xdb,0x4c,0x02,0x40]
fisttpl 64(%rdx,%rax)

// CHECK: fisttpll 485498096
// CHECK: encoding: [0xdd,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
fisttpll 485498096

// CHECK: fisttpll 64(%rdx)
// CHECK: encoding: [0xdd,0x4a,0x40]
fisttpll 64(%rdx)

// CHECK: fisttpll -64(%rdx,%rax,4)
// CHECK: encoding: [0xdd,0x4c,0x82,0xc0]
fisttpll -64(%rdx,%rax,4)

// CHECK: fisttpll 64(%rdx,%rax,4)
// CHECK: encoding: [0xdd,0x4c,0x82,0x40]
fisttpll 64(%rdx,%rax,4)

// CHECK: fisttpll 64(%rdx,%rax)
// CHECK: encoding: [0xdd,0x4c,0x02,0x40]
fisttpll 64(%rdx,%rax)

// CHECK: fisttpll (%rdx)
// CHECK: encoding: [0xdd,0x0a]
fisttpll (%rdx)

// CHECK: fisttpl (%rdx)
// CHECK: encoding: [0xdb,0x0a]
fisttpl (%rdx)

// CHECK: fisttps 485498096
// CHECK: encoding: [0xdf,0x0c,0x25,0xf0,0x1c,0xf0,0x1c]
fisttps 485498096

// CHECK: fisttps 64(%rdx)
// CHECK: encoding: [0xdf,0x4a,0x40]
fisttps 64(%rdx)

// CHECK: fisttps -64(%rdx,%rax,4)
// CHECK: encoding: [0xdf,0x4c,0x82,0xc0]
fisttps -64(%rdx,%rax,4)

// CHECK: fisttps 64(%rdx,%rax,4)
// CHECK: encoding: [0xdf,0x4c,0x82,0x40]
fisttps 64(%rdx,%rax,4)

// CHECK: fisttps 64(%rdx,%rax)
// CHECK: encoding: [0xdf,0x4c,0x02,0x40]
fisttps 64(%rdx,%rax)

// CHECK: fisttps (%rdx)
// CHECK: encoding: [0xdf,0x0a]
fisttps (%rdx)

// CHECK: haddpd 485498096, %xmm6
// CHECK: encoding: [0x66,0x0f,0x7c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
haddpd 485498096, %xmm6

// CHECK: haddpd -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7c,0x74,0x82,0xc0]
haddpd -64(%rdx,%rax,4), %xmm6

// CHECK: haddpd 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7c,0x74,0x82,0x40]
haddpd 64(%rdx,%rax,4), %xmm6

// CHECK: haddpd 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7c,0x74,0x02,0x40]
haddpd 64(%rdx,%rax), %xmm6

// CHECK: haddpd 64(%rdx), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7c,0x72,0x40]
haddpd 64(%rdx), %xmm6

// CHECK: haddpd (%rdx), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7c,0x32]
haddpd (%rdx), %xmm6

// CHECK: haddpd %xmm6, %xmm6
// CHECK: encoding: [0x66,0x0f,0x7c,0xf6]
haddpd %xmm6, %xmm6

// CHECK: haddps 485498096, %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7c,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
haddps 485498096, %xmm6

// CHECK: haddps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7c,0x74,0x82,0xc0]
haddps -64(%rdx,%rax,4), %xmm6

// CHECK: haddps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7c,0x74,0x82,0x40]
haddps 64(%rdx,%rax,4), %xmm6

// CHECK: haddps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7c,0x74,0x02,0x40]
haddps 64(%rdx,%rax), %xmm6

// CHECK: haddps 64(%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7c,0x72,0x40]
haddps 64(%rdx), %xmm6

// CHECK: haddps (%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7c,0x32]
haddps (%rdx), %xmm6

// CHECK: haddps %xmm6, %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7c,0xf6]
haddps %xmm6, %xmm6

// CHECK: hsubpd 485498096, %xmm6
// CHECK: encoding: [0x66,0x0f,0x7d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
hsubpd 485498096, %xmm6

// CHECK: hsubpd -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7d,0x74,0x82,0xc0]
hsubpd -64(%rdx,%rax,4), %xmm6

// CHECK: hsubpd 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7d,0x74,0x82,0x40]
hsubpd 64(%rdx,%rax,4), %xmm6

// CHECK: hsubpd 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7d,0x74,0x02,0x40]
hsubpd 64(%rdx,%rax), %xmm6

// CHECK: hsubpd 64(%rdx), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7d,0x72,0x40]
hsubpd 64(%rdx), %xmm6

// CHECK: hsubpd (%rdx), %xmm6
// CHECK: encoding: [0x66,0x0f,0x7d,0x32]
hsubpd (%rdx), %xmm6

// CHECK: hsubpd %xmm6, %xmm6
// CHECK: encoding: [0x66,0x0f,0x7d,0xf6]
hsubpd %xmm6, %xmm6

// CHECK: hsubps 485498096, %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7d,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
hsubps 485498096, %xmm6

// CHECK: hsubps -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7d,0x74,0x82,0xc0]
hsubps -64(%rdx,%rax,4), %xmm6

// CHECK: hsubps 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7d,0x74,0x82,0x40]
hsubps 64(%rdx,%rax,4), %xmm6

// CHECK: hsubps 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7d,0x74,0x02,0x40]
hsubps 64(%rdx,%rax), %xmm6

// CHECK: hsubps 64(%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7d,0x72,0x40]
hsubps 64(%rdx), %xmm6

// CHECK: hsubps (%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7d,0x32]
hsubps (%rdx), %xmm6

// CHECK: hsubps %xmm6, %xmm6
// CHECK: encoding: [0xf2,0x0f,0x7d,0xf6]
hsubps %xmm6, %xmm6

// CHECK: lddqu 485498096, %xmm6
// CHECK: encoding: [0xf2,0x0f,0xf0,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
lddqu 485498096, %xmm6

// CHECK: lddqu -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xf0,0x74,0x82,0xc0]
lddqu -64(%rdx,%rax,4), %xmm6

// CHECK: lddqu 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xf0,0x74,0x82,0x40]
lddqu 64(%rdx,%rax,4), %xmm6

// CHECK: lddqu 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xf0,0x74,0x02,0x40]
lddqu 64(%rdx,%rax), %xmm6

// CHECK: lddqu 64(%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xf0,0x72,0x40]
lddqu 64(%rdx), %xmm6

// CHECK: lddqu (%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0xf0,0x32]
lddqu (%rdx), %xmm6

// CHECK: monitor
// CHECK: encoding: [0x0f,0x01,0xc8]
monitor

// CHECK: movddup 485498096, %xmm6
// CHECK: encoding: [0xf2,0x0f,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movddup 485498096, %xmm6

// CHECK: movddup -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x12,0x74,0x82,0xc0]
movddup -64(%rdx,%rax,4), %xmm6

// CHECK: movddup 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x12,0x74,0x82,0x40]
movddup 64(%rdx,%rax,4), %xmm6

// CHECK: movddup 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x12,0x74,0x02,0x40]
movddup 64(%rdx,%rax), %xmm6

// CHECK: movddup 64(%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x12,0x72,0x40]
movddup 64(%rdx), %xmm6

// CHECK: movddup (%rdx), %xmm6
// CHECK: encoding: [0xf2,0x0f,0x12,0x32]
movddup (%rdx), %xmm6

// CHECK: movddup %xmm6, %xmm6
// CHECK: encoding: [0xf2,0x0f,0x12,0xf6]
movddup %xmm6, %xmm6

// CHECK: movshdup 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x16,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movshdup 485498096, %xmm6

// CHECK: movshdup -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x16,0x74,0x82,0xc0]
movshdup -64(%rdx,%rax,4), %xmm6

// CHECK: movshdup 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x16,0x74,0x82,0x40]
movshdup 64(%rdx,%rax,4), %xmm6

// CHECK: movshdup 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x16,0x74,0x02,0x40]
movshdup 64(%rdx,%rax), %xmm6

// CHECK: movshdup 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x16,0x72,0x40]
movshdup 64(%rdx), %xmm6

// CHECK: movshdup (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x16,0x32]
movshdup (%rdx), %xmm6

// CHECK: movshdup %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x16,0xf6]
movshdup %xmm6, %xmm6

// CHECK: movsldup 485498096, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x12,0x34,0x25,0xf0,0x1c,0xf0,0x1c]
movsldup 485498096, %xmm6

// CHECK: movsldup -64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x12,0x74,0x82,0xc0]
movsldup -64(%rdx,%rax,4), %xmm6

// CHECK: movsldup 64(%rdx,%rax,4), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x12,0x74,0x82,0x40]
movsldup 64(%rdx,%rax,4), %xmm6

// CHECK: movsldup 64(%rdx,%rax), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x12,0x74,0x02,0x40]
movsldup 64(%rdx,%rax), %xmm6

// CHECK: movsldup 64(%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x12,0x72,0x40]
movsldup 64(%rdx), %xmm6

// CHECK: movsldup (%rdx), %xmm6
// CHECK: encoding: [0xf3,0x0f,0x12,0x32]
movsldup (%rdx), %xmm6

// CHECK: movsldup %xmm6, %xmm6
// CHECK: encoding: [0xf3,0x0f,0x12,0xf6]
movsldup %xmm6, %xmm6

// CHECK: mwait
// CHECK: encoding: [0x0f,0x01,0xc9]
mwait

