# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-12: error:
# ERROR-NOT: error:
# CHECK: {nf}	bzhil	%ecx, %edx, %r10d
# CHECK: encoding: [0x62,0x72,0x74,0x0c,0xf5,0xd2]
         {nf}	bzhil	%ecx, %edx, %r10d

# CHECK: {evex}	bzhil	%ecx, %edx, %r10d
# CHECK: encoding: [0x62,0x72,0x74,0x08,0xf5,0xd2]
         {evex}	bzhil	%ecx, %edx, %r10d

# CHECK: {nf}	bzhil	%ecx, 123(%rax,%rbx,4), %edx
# CHECK: encoding: [0x62,0xf2,0x74,0x0c,0xf5,0x54,0x98,0x7b]
         {nf}	bzhil	%ecx, 123(%rax,%rbx,4), %edx

# CHECK: {evex}	bzhil	%ecx, 123(%rax,%rbx,4), %edx
# CHECK: encoding: [0x62,0xf2,0x74,0x08,0xf5,0x54,0x98,0x7b]
         {evex}	bzhil	%ecx, 123(%rax,%rbx,4), %edx

# CHECK: {nf}	bzhiq	%r9, %r15, %r11
# CHECK: encoding: [0x62,0x52,0xb4,0x0c,0xf5,0xdf]
         {nf}	bzhiq	%r9, %r15, %r11

# CHECK: {evex}	bzhiq	%r9, %r15, %r11
# CHECK: encoding: [0x62,0x52,0xb4,0x08,0xf5,0xdf]
         {evex}	bzhiq	%r9, %r15, %r11

# CHECK: {nf}	bzhiq	%r9, 123(%rax,%rbx,4), %r15
# CHECK: encoding: [0x62,0x72,0xb4,0x0c,0xf5,0x7c,0x98,0x7b]
         {nf}	bzhiq	%r9, 123(%rax,%rbx,4), %r15

# CHECK: {evex}	bzhiq	%r9, 123(%rax,%rbx,4), %r15
# CHECK: encoding: [0x62,0x72,0xb4,0x08,0xf5,0x7c,0x98,0x7b]
         {evex}	bzhiq	%r9, 123(%rax,%rbx,4), %r15

# CHECK: bzhil	%r18d, %r22d, %r26d
# CHECK: encoding: [0x62,0x6a,0x6c,0x00,0xf5,0xd6]
         bzhil	%r18d, %r22d, %r26d

# CHECK: bzhil	%r18d, 291(%r28,%r29,4), %r22d
# CHECK: encoding: [0x62,0x8a,0x68,0x00,0xf5,0xb4,0xac,0x23,0x01,0x00,0x00]
         bzhil	%r18d, 291(%r28,%r29,4), %r22d

# CHECK: bzhiq	%r19, %r23, %r27
# CHECK: encoding: [0x62,0x6a,0xe4,0x00,0xf5,0xdf]
         bzhiq	%r19, %r23, %r27

# CHECK: bzhiq	%r19, 291(%r28,%r29,4), %r23
# CHECK: encoding: [0x62,0x8a,0xe0,0x00,0xf5,0xbc,0xac,0x23,0x01,0x00,0x00]
         bzhiq	%r19, 291(%r28,%r29,4), %r23
