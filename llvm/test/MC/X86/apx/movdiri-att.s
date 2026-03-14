# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-4: error:
# ERROR-NOT: error:
# CHECK: {evex}	movdiri	%ecx, 123(%eax,%ebx,4)
# CHECK: encoding: [0x67,0x62,0xf4,0x7c,0x08,0xf9,0x4c,0x98,0x7b]
         {evex}	movdiri	%ecx, 123(%eax,%ebx,4)

# CHECK: {evex}	movdiri	%r9, 123(%rax,%rbx,4)
# CHECK: encoding: [0x62,0x74,0xfc,0x08,0xf9,0x4c,0x98,0x7b]
         {evex}	movdiri	%r9, 123(%rax,%rbx,4)

# CHECK: movdiri	%r18d, 291(%r28,%r29,4)
# CHECK: encoding: [0x62,0x8c,0x78,0x08,0xf9,0x94,0xac,0x23,0x01,0x00,0x00]
         movdiri	%r18d, 291(%r28,%r29,4)

# CHECK: movdiri	%r19, 291(%r28,%r29,4)
# CHECK: encoding: [0x62,0x8c,0xf8,0x08,0xf9,0x9c,0xac,0x23,0x01,0x00,0x00]
         movdiri	%r19, 291(%r28,%r29,4)
