# RUN: llvm-mc -filetype=obj --mattr=+experimental-xqcibi -triple=riscv32 %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcibi -d -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+experimental-xqcibi %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcibi -dr -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST-RELAX %s

       .text
       .type   test,@function

test:

# CHECK-INST:         qc.beqi     a0, 0xa, 0x8
# CHECK-INST-NEXT:    jal     zero, 0x1458
# CHECK-INST-RELAX:         qc.beqi     a0, 0xa, 0x8
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.bnei a0, 10, .L1
.fill 1300, 4, 0
.L1:
   ret

# CHECK-INST:         qc.bnei     a0, 0x6, 0x1462
# CHECK-INST-NEXT:    jal     zero, 0x28b2
# CHECK-INST-RELAX:         qc.bnei     a0, 0x6, 0x1462
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.beqi a0, 6, .L2
.fill 1300, 4, 0
.L2:
   ret

# CHECK-INST:         qc.bgei     a0, 0xd, 0x28bc
# CHECK-INST-NEXT:    jal     zero, 0x3d0c
# CHECK-INST-RELAX:         qc.bgei     a0, 0xd, 0x28bc
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.blti a0, 13, .L3
.fill 1300, 4, 0
.L3:
   ret

# CHECK-INST:         qc.blti     a0, 0x1, 0x3d16
# CHECK-INST-NEXT:    jal     zero, 0x5166
# CHECK-INST-RELAX:         qc.blti     a0, 0x1, 0x3d16
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.bgei a0, 1, .L4
.fill 1300, 4, 0
.L4:
   ret

# CHECK-INST:         qc.bgeui    a0, 0x5, 0x5170
# CHECK-INST-NEXT:    jal     zero, 0x65c0
# CHECK-INST-RELAX:         qc.bgeui    a0, 0x5, 0x5170
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.bltui a0, 5, .L5
.fill 1300, 4, 0
.L5:
   ret

# CHECK-INST:         qc.bltui    a0, 0xc, 0x65ca
# CHECK-INST-NEXT:    jal     zero, 0x7a1a
# CHECK-INST-RELAX:         qc.bltui    a0, 0xc, 0x65ca
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.bgeui a0, 12, .L6
.fill 1300, 4, 0
.L6:
   ret

# CHECK-INST:         qc.e.beqi    a0, 0x51, 0x7a26
# CHECK-INST-NEXT:    jal     zero, 0x8e76
# CHECK-INST-RELAX:         qc.e.beqi    a0, 0x51, 0x7a26
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bnei a0, 81, .L7
.fill 1300, 4, 0
.L7:
   ret

# CHECK-INST:         qc.e.bnei    a0, 0x3e, 0x8e82
# CHECK-INST-NEXT:    jal     zero, 0xa2d2
# CHECK-INST-RELAX:         qc.e.bnei    a0, 0x3e, 0x8e82
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.beqi a0, 62, .L8
.fill 1300, 4, 0
.L8:
   ret

# CHECK-INST:         qc.e.bgei    a0, 0x5d, 0xa2de
# CHECK-INST-NEXT:    jal     zero, 0xb72e
# CHECK-INST-RELAX:         qc.e.bgei    a0, 0x5d, 0xa2de
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.blti a0, 93, .L9
.fill 1300, 4, 0
.L9:
   ret

# CHECK-INST:         qc.e.blti    a0, 0x2c, 0xb73a
# CHECK-INST-NEXT:    jal     zero, 0xcb8a
# CHECK-INST-RELAX:         qc.e.blti    a0, 0x2c, 0xb73a
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bgei a0, 44, .L10
.fill 1300, 4, 0
.L10:
   ret

# CHECK-INST:         qc.e.bgeui    a0, 0x37, 0xcb96
# CHECK-INST-NEXT:    jal     zero, 0xdfe6
# CHECK-INST-RELAX:         qc.e.bgeui    a0, 0x37, 0xcb96
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bltui a0, 55, .L11
.fill 1300, 4, 0
.L11:
   ret

# CHECK-INST:         qc.e.bltui    a0, 0x24, 0xdff2
# CHECK-INST-NEXT:    jal     zero, 0xf442
# CHECK-INST-RELAX:         qc.e.bltui    a0, 0x24, 0xdff2
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bgeui a0, 36, .L12
.fill 1300, 4, 0
.L12:
   ret

# Check that instructions are first compressed and then relaxed

# CHECK-INST:         qc.beqi    a0, 0xa, 0xf44c
# CHECK-INST-NEXT:    jal     zero, 0x1089c
# CHECK-INST-RELAX:         qc.beqi    a0, 0xa, 0xf44c
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bnei a0, 10, .L13
.fill 1300, 4, 0
.L13:
   ret

# CHECK-INST:         qc.bnei    a0, 0xa, 0x108a6
# CHECK-INST-NEXT:    jal     zero, 0x11cf6
# CHECK-INST-RELAX:         qc.bnei    a0, 0xa, 0x108a6
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.beqi a0, 10, .L14
.fill 1300, 4, 0
.L14:
   ret

# CHECK-INST:         qc.bgei    a0, 0xa, 0x11d00
# CHECK-INST-NEXT:    jal     zero, 0x13150
# CHECK-INST-RELAX:         qc.bgei    a0, 0xa, 0x11d00
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.blti a0, 10, .L15
.fill 1300, 4, 0
.L15:
   ret

# CHECK-INST:         qc.blti    a0, 0xa, 0x1315a
# CHECK-INST-NEXT:    jal     zero, 0x145aa
# CHECK-INST-RELAX:         qc.blti    a0, 0xa, 0x1315a
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bgei a0, 10, .L16
.fill 1300, 4, 0
.L16:
   ret

# CHECK-INST:         qc.bgeui    a0, 0xa, 0x145b4
# CHECK-INST-NEXT:    jal     zero, 0x15a04
# CHECK-INST-RELAX:         qc.bgeui    a0, 0xa, 0x145b4
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bltui a0, 10, .L17
.fill 1300, 4, 0
.L17:
   ret

# CHECK-INST:         qc.bltui    a0, 0xa, 0x15a0e
# CHECK-INST-NEXT:    jal     zero, 0x16e5e
# CHECK-INST-RELAX:         qc.bltui    a0, 0xa, 0x15a0e
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   qc.e.bgeui a0, 10, .L18
.fill 1300, 4, 0
.L18:
   ret

.Lfunc_end0:
       .size   test, .Lfunc_end0-test
