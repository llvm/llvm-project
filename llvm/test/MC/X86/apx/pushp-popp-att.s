# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-8: error:
# ERROR-NOT: error:

# CHECK: pushp	%rax
# CHECK: encoding: [0xd5,0x08,0x50]
         pushp	%rax
# CHECK: pushp	%rbx
# CHECK: encoding: [0xd5,0x08,0x53]
         pushp	%rbx
# CHECK: pushp	%r15
# CHECK: encoding: [0xd5,0x09,0x57]
         pushp	%r15
# CHECK: pushp	%r16
# CHECK: encoding: [0xd5,0x18,0x50]
         pushp	%r16

# CHECK: popp	%rax
# CHECK: encoding: [0xd5,0x08,0x58]
         popp	%rax
# CHECK: popp	%rbx
# CHECK: encoding: [0xd5,0x08,0x5b]
         popp	%rbx
# CHECK: popp	%r15
# CHECK: encoding: [0xd5,0x09,0x5f]
         popp	%r15
# CHECK: popp	%r16
# CHECK: encoding: [0xd5,0x18,0x58]
         popp	%r16
