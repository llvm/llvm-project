# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-12: error:
# ERROR-NOT: error:
# CHECK: {nf}	andnl	%ecx, %edx, %r10d
# CHECK: encoding: [0x62,0x72,0x6c,0x0c,0xf2,0xd1]
         {nf}	andnl	%ecx, %edx, %r10d

# CHECK: {evex}	andnl	%ecx, %edx, %r10d
# CHECK: encoding: [0x62,0x72,0x6c,0x08,0xf2,0xd1]
         {evex}	andnl	%ecx, %edx, %r10d

# CHECK: {nf}	andnq	%r9, %r15, %r11
# CHECK: encoding: [0x62,0x52,0x84,0x0c,0xf2,0xd9]
         {nf}	andnq	%r9, %r15, %r11

# CHECK: {evex}	andnq	%r9, %r15, %r11
# CHECK: encoding: [0x62,0x52,0x84,0x08,0xf2,0xd9]
         {evex}	andnq	%r9, %r15, %r11

# CHECK: {nf}	andnl	123(%rax,%rbx,4), %ecx, %edx
# CHECK: encoding: [0x62,0xf2,0x74,0x0c,0xf2,0x54,0x98,0x7b]
         {nf}	andnl	123(%rax,%rbx,4), %ecx, %edx

# CHECK: {evex}	andnl	123(%rax,%rbx,4), %ecx, %edx
# CHECK: encoding: [0x62,0xf2,0x74,0x08,0xf2,0x54,0x98,0x7b]
         {evex}	andnl	123(%rax,%rbx,4), %ecx, %edx

# CHECK: {nf}	andnq	123(%rax,%rbx,4), %r9, %r15
# CHECK: encoding: [0x62,0x72,0xb4,0x0c,0xf2,0x7c,0x98,0x7b]
         {nf}	andnq	123(%rax,%rbx,4), %r9, %r15

# CHECK: {evex}	andnq	123(%rax,%rbx,4), %r9, %r15
# CHECK: encoding: [0x62,0x72,0xb4,0x08,0xf2,0x7c,0x98,0x7b]
         {evex}	andnq	123(%rax,%rbx,4), %r9, %r15

# CHECK: andnl	%r18d, %r22d, %r26d
# CHECK: encoding: [0x62,0x6a,0x4c,0x00,0xf2,0xd2]
         andnl	%r18d, %r22d, %r26d

# CHECK: andnq	%r19, %r23, %r27
# CHECK: encoding: [0x62,0x6a,0xc4,0x00,0xf2,0xdb]
         andnq	%r19, %r23, %r27

# CHECK: andnl	291(%r28,%r29,4), %r18d, %r22d
# CHECK: encoding: [0x62,0x8a,0x68,0x00,0xf2,0xb4,0xac,0x23,0x01,0x00,0x00]
         andnl	291(%r28,%r29,4), %r18d, %r22d

# CHECK: andnq	291(%r28,%r29,4), %r19, %r23
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf2,0xbc,0xac,0x23,0x01,0x00,0x00]
         andnq	291(%r28,%r29,4), %r19, %r23
