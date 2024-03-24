# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-27: error:
# ERROR-NOT: error:

## sha1msg1

# CHECK: {evex}	sha1msg1	%xmm13, %xmm12
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xd9,0xe5]
         {evex}	sha1msg1	%xmm13, %xmm12

# CHECK: {evex}	sha1msg1	123(%rax,%rbx,4), %xmm12
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xd9,0x64,0x98,0x7b]
         {evex}	sha1msg1	123(%rax,%rbx,4), %xmm12

# CHECK: sha1msg1	%xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xc9,0xe5]
         sha1msg1	%xmm13, %xmm12

# CHECK: sha1msg1	291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd9,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1msg1	291(%r28,%r29,4), %xmm12

## sha1msg2

# CHECK: {evex}	sha1msg2	%xmm13, %xmm12
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xda,0xe5]
         {evex}	sha1msg2	%xmm13, %xmm12

# CHECK: {evex}	sha1msg2	123(%rax,%rbx,4), %xmm12
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xda,0x64,0x98,0x7b]
         {evex}	sha1msg2	123(%rax,%rbx,4), %xmm12

# CHECK: sha1msg2	%xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xca,0xe5]
         sha1msg2	%xmm13, %xmm12

# CHECK: sha1msg2	291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xda,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1msg2	291(%r28,%r29,4), %xmm12

## sha1nexte

# CHECK: {evex}	sha1nexte	%xmm13, %xmm12
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xd8,0xe5]
         {evex}	sha1nexte	%xmm13, %xmm12

# CHECK: {evex}	sha1nexte	123(%rax,%rbx,4), %xmm12
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xd8,0x64,0x98,0x7b]
         {evex}	sha1nexte	123(%rax,%rbx,4), %xmm12

# CHECK: sha1nexte	%xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xc8,0xe5]
         sha1nexte	%xmm13, %xmm12

# CHECK: sha1nexte	291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd8,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha1nexte	291(%r28,%r29,4), %xmm12

## sha1rnds4

# CHECK: {evex}	sha1rnds4	$123, %xmm13, %xmm12
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xd4,0xe5,0x7b]
         {evex}	sha1rnds4	$123, %xmm13, %xmm12

# CHECK: {evex}	sha1rnds4	$123, 123(%rax,%rbx,4), %xmm12
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xd4,0x64,0x98,0x7b,0x7b]
         {evex}	sha1rnds4	$123, 123(%rax,%rbx,4), %xmm12

# CHECK: sha1rnds4	$123, %xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x3a,0xcc,0xe5,0x7b]
         sha1rnds4	$123, %xmm13, %xmm12

# CHECK: sha1rnds4	$123, 291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xd4,0xa4,0xac,0x23,0x01,0x00,0x00,0x7b]
         sha1rnds4	$123, 291(%r28,%r29,4), %xmm12

## sha256msg1

# CHECK: {evex}	sha256msg1	%xmm13, %xmm12
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xdc,0xe5]
         {evex}	sha256msg1	%xmm13, %xmm12

# CHECK: {evex}	sha256msg1	123(%rax,%rbx,4), %xmm12
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xdc,0x64,0x98,0x7b]
         {evex}	sha256msg1	123(%rax,%rbx,4), %xmm12

# CHECK: sha256msg1	%xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xcc,0xe5]
         sha256msg1	%xmm13, %xmm12

# CHECK: sha256msg1	291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xdc,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha256msg1	291(%r28,%r29,4), %xmm12

## sha256msg2

# CHECK: {evex}	sha256msg2	%xmm13, %xmm12
# CHECK: encoding: [0x62,0x54,0x7c,0x08,0xdd,0xe5]
         {evex}	sha256msg2	%xmm13, %xmm12

# CHECK: {evex}	sha256msg2	123(%rax,%rbx,4), %xmm12
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xdd,0x64,0x98,0x7b]
         {evex}	sha256msg2	123(%rax,%rbx,4), %xmm12

# CHECK: sha256msg2	%xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xcd,0xe5]
         sha256msg2	%xmm13, %xmm12

# CHECK: sha256msg2	291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xdd,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha256msg2	291(%r28,%r29,4), %xmm12

## sha256rnds2

# CHECK: {evex}	sha256rnds2	%xmm0, 123(%rax,%rbx,4), %xmm12
# CHECK: encoding: [0x62,0x74,0x7c,0x08,0xdb,0x64,0x98,0x7b]
         {evex}	sha256rnds2	%xmm0, 123(%rax,%rbx,4), %xmm12

# CHECK: sha256rnds2	%xmm0, %xmm13, %xmm12
# CHECK: encoding: [0x45,0x0f,0x38,0xcb,0xe5]
         sha256rnds2	%xmm0, %xmm13, %xmm12

# CHECK: sha256rnds2	%xmm0, 291(%r28,%r29,4), %xmm12
# CHECK: encoding: [0x62,0x1c,0x78,0x08,0xdb,0xa4,0xac,0x23,0x01,0x00,0x00]
         sha256rnds2	%xmm0, 291(%r28,%r29,4), %xmm12
