# RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

## sha1msg1

# CHECK: {evex}	sha1msg1	xmm12, xmm13
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xd9,0xe5]
         {evex}	sha1msg1	xmm12, xmm13

# CHECK: {evex}	sha1msg1	xmm12, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xd9,0x64,0x98,0x7b]
         {evex}	sha1msg1	xmm12, xmmword ptr [rax + 4*rbx + 123]

# CHECK: sha1msg1	xmm12, xmm13
# CHECK: encoding: [0x45,0x0f,0x38,0xc9,0xe5]
         sha1msg1	xmm12, xmm13

# CHECK: sha1msg1	xmm12, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd9,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1msg1	xmm12, xmmword ptr [r28 + 4*r29 + 291]

## sha1msg2

# CHECK: {evex}	sha1msg2	xmm12, xmm13
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xda,0xe5]
         {evex}	sha1msg2	xmm12, xmm13

# CHECK: {evex}	sha1msg2	xmm12, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xda,0x64,0x98,0x7b]
         {evex}	sha1msg2	xmm12, xmmword ptr [rax + 4*rbx + 123]

# CHECK: sha1msg2	xmm12, xmm13
# CHECK: encoding: [0x45,0x0f,0x38,0xca,0xe5]
         sha1msg2	xmm12, xmm13

# CHECK: sha1msg2	xmm12, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xda,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1msg2	xmm12, xmmword ptr [r28 + 4*r29 + 291]

## sha1nexte

# CHECK: {evex}	sha1nexte	xmm12, xmm13
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xd8,0xe5]
         {evex}	sha1nexte	xmm12, xmm13

# CHECK: {evex}	sha1nexte	xmm12, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xd8,0x64,0x98,0x7b]
         {evex}	sha1nexte	xmm12, xmmword ptr [rax + 4*rbx + 123]

# CHECK: sha1nexte	xmm12, xmm13
# CHECK: encoding: [0x45,0x0f,0x38,0xc8,0xe5]
         sha1nexte	xmm12, xmm13

# CHECK: sha1nexte	xmm12, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd8,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1nexte	xmm12, xmmword ptr [r28 + 4*r29 + 291]

## sha1rnds4

# CHECK: {evex}	sha1rnds4	xmm12, xmm13, 123
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xd4,0xe5,0x7b]
         {evex}	sha1rnds4	xmm12, xmm13, 123

# CHECK: {evex}	sha1rnds4	xmm12, xmmword ptr [rax + 4*rbx + 123], 123
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xd4,0x64,0x98,0x7b,0x7b]
         {evex}	sha1rnds4	xmm12, xmmword ptr [rax + 4*rbx + 123], 123

# CHECK: sha1rnds4	xmm12, xmm13, 123
# CHECK: encoding: [0x45,0x0f,0x3a,0xcc,0xe5,0x7b]
         sha1rnds4	xmm12, xmm13, 123

# CHECK: sha1rnds4	xmm12, xmmword ptr [r28 + 4*r29 + 291], 123
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd4,0xa4,0xac,0x23,0x01,0x00,0x00,0x7b]
         sha1rnds4	xmm12, xmmword ptr [r28 + 4*r29 + 291], 123

## sha256msg1

# CHECK: {evex}	sha256msg1	xmm12, xmm13
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xdc,0xe5]
         {evex}	sha256msg1	xmm12, xmm13

# CHECK: {evex}	sha256msg1	xmm12, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xdc,0x64,0x98,0x7b]
         {evex}	sha256msg1	xmm12, xmmword ptr [rax + 4*rbx + 123]

# CHECK: sha256msg1	xmm12, xmm13
# CHECK: encoding: [0x45,0x0f,0x38,0xcc,0xe5]
         sha256msg1	xmm12, xmm13

# CHECK: sha256msg1	xmm12, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xdc,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha256msg1	xmm12, xmmword ptr [r28 + 4*r29 + 291]

## sha256msg2

# CHECK: {evex}	sha256msg2	xmm12, xmm13
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xdd,0xe5]
         {evex}	sha256msg2	xmm12, xmm13

# CHECK: {evex}	sha256msg2	xmm12, xmmword ptr [rax + 4*rbx + 123]
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xdd,0x64,0x98,0x7b]
         {evex}	sha256msg2	xmm12, xmmword ptr [rax + 4*rbx + 123]

# CHECK: sha256msg2	xmm12, xmm13
# CHECK: encoding: [0x45,0x0f,0x38,0xcd,0xe5]
         sha256msg2	xmm12, xmm13

# CHECK: sha256msg2	xmm12, xmmword ptr [r28 + 4*r29 + 291]
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xdd,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha256msg2	xmm12, xmmword ptr [r28 + 4*r29 + 291]

## sha256rnds2

# CHECK: {evex}	sha256rnds2	xmm12, xmmword ptr [rax + 4*rbx + 123], xmm0
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xdb,0x64,0x98,0x7b]
         {evex}	sha256rnds2	xmm12, xmmword ptr [rax + 4*rbx + 123], xmm0

# CHECK: sha256rnds2	xmm12, xmm13, xmm0
# CHECK: encoding: [0x45,0x0f,0x38,0xcb,0xe5]
         sha256rnds2	xmm12, xmm13, xmm0

# CHECK: sha256rnds2	xmm12, xmmword ptr [r28 + 4*r29 + 291], xmm0
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xdb,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha256rnds2	xmm12, xmmword ptr [r28 + 4*r29 + 291], xmm0
