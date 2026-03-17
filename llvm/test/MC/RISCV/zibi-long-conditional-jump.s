# RUN: llvm-mc -filetype=obj --mattr=+experimental-zibi -triple=riscv32 %s \
# RUN:     | llvm-objdump --mattr=+experimental-zibi -d -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+experimental-zibi %s \
# RUN:     | llvm-objdump --mattr=+experimental-zibi -dr -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST-RELAX %s

       .text
       .type   test,@function

test:

# CHECK-INST:         beqi     a0, 0xa, 0x8
# CHECK-INST-NEXT:    jal     zero, 0x1458
# CHECK-INST-RELAX:         beqi     a0, 0xa, 0x8
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   bnei a0, 10, .L1
.fill 1300, 4, 0
.L1:
   ret

# CHECK-INST:         bnei     a0, 0x6, 0x1464
# CHECK-INST-NEXT:    jal     zero, 0x28b4
# CHECK-INST-RELAX:         bnei     a0, 0x6, 0x1464
# CHECK-INST-RELAX-NEXT:    jal     zero, {{.*}}
   beqi a0, 6, .L2
.fill 1300, 4, 0
.L2:
   ret

.Lfunc_end0:
       .size   test, .Lfunc_end0-test
