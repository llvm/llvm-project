# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-12: error:
# ERROR-NOT: error:
# CHECK: {nf}	blsmskl	%ecx, %edx
# CHECK: encoding: [0x62,0xf2,0x6c,0x0c,0xf3,0xd1]
         {nf}	blsmskl	%ecx, %edx

# CHECK: {evex}	blsmskl	%ecx, %edx
# CHECK: encoding: [0x62,0xf2,0x6c,0x08,0xf3,0xd1]
         {evex}	blsmskl	%ecx, %edx

# CHECK: {nf}	blsmskq	%r9, %r15
# CHECK: encoding: [0x62,0xd2,0x84,0x0c,0xf3,0xd1]
         {nf}	blsmskq	%r9, %r15

# CHECK: {evex}	blsmskq	%r9, %r15
# CHECK: encoding: [0x62,0xd2,0x84,0x08,0xf3,0xd1]
         {evex}	blsmskq	%r9, %r15

# CHECK: {nf}	blsmskl	123(%rax,%rbx,4), %ecx
# CHECK: encoding: [0x62,0xf2,0x74,0x0c,0xf3,0x54,0x98,0x7b]
         {nf}	blsmskl	123(%rax,%rbx,4), %ecx

# CHECK: {evex}	blsmskl	123(%rax,%rbx,4), %ecx
# CHECK: encoding: [0x62,0xf2,0x74,0x08,0xf3,0x54,0x98,0x7b]
         {evex}	blsmskl	123(%rax,%rbx,4), %ecx

# CHECK: {nf}	blsmskq	123(%rax,%rbx,4), %r9
# CHECK: encoding: [0x62,0xf2,0xb4,0x0c,0xf3,0x54,0x98,0x7b]
         {nf}	blsmskq	123(%rax,%rbx,4), %r9

# CHECK: {evex}	blsmskq	123(%rax,%rbx,4), %r9
# CHECK: encoding: [0x62,0xf2,0xb4,0x08,0xf3,0x54,0x98,0x7b]
         {evex}	blsmskq	123(%rax,%rbx,4), %r9

# CHECK: blsmskl	%r18d, %r22d
# CHECK: encoding: [0x62,0xfa,0x4c,0x00,0xf3,0xd2]
         blsmskl	%r18d, %r22d

# CHECK: blsmskq	%r19, %r23
# CHECK: encoding: [0x62,0xfa,0xc4,0x00,0xf3,0xd3]
         blsmskq	%r19, %r23

# CHECK: blsmskl	291(%r28,%r29,4), %r18d
# CHECK: encoding: [0x62,0x9a,0x68,0x00,0xf3,0x94,0xac,0x23,0x01,0x00,0x00]
         blsmskl	291(%r28,%r29,4), %r18d

# CHECK: blsmskq	291(%r28,%r29,4), %r19
# CHECK: encoding: [0x62,0x9a,0xe0,0x00,0xf3,0x94,0xac,0x23,0x01,0x00,0x00]
         blsmskq	291(%r28,%r29,4), %r19
