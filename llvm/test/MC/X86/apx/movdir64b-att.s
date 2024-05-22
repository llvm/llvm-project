# RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple i386 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR-COUNT-2: error:
# ERROR-NOT: error:
# CHECK: movdir64b	291(%r28d,%r29d,4), %r18d
# CHECK: encoding: [0x67,0x62,0x8c,0x79,0x08,0xf8,0x94,0xac,0x23,0x01,0x00,0x00]
         movdir64b	291(%r28d,%r29d,4), %r18d

# CHECK: movdir64b	291(%r28,%r29,4), %r19
# CHECK: encoding: [0x62,0x8c,0x79,0x08,0xf8,0x9c,0xac,0x23,0x01,0x00,0x00]
         movdir64b	291(%r28,%r29,4), %r19
