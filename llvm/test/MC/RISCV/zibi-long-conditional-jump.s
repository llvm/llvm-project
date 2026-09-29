# RUN: llvm-mc -filetype=obj --mattr=+experimental-zibi -triple=riscv32 %s \
# RUN:     | llvm-objdump --mattr=+experimental-zibi -d -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax,+experimental-zibi %s \
# RUN:     | llvm-objdump --mattr=+experimental-zibi -dr -M no-aliases - \
# RUN:     | FileCheck --check-prefix=CHECK-INST-RELAX %s

# Tests that are forward in range.

# CHECK-INST-LABEL: beqi_in_range_forward
# CHECK-INST:       beqi a0, 0xa, {{.*}}

# CHECK-INST-RELAX-LABEL: beqi_in_range_forward
# CHECK-INST-RELAX:       beqi a0, 0xa, {{.*}}
beqi_in_range_forward:
   beqi a0, 10, .L1
   .fill 1000, 4, 0
.L1:
   ret

# CHECK-INST-LABEL: bnei_in_range_forward
# CHECK-INST:       bnei a0, 0xa, {{.*}}

# CHECK-INST-RELAX-LABEL: bnei_in_range_forward
# CHECK-INST-RELAX:       bnei a0, 0xa, {{.*}}
bnei_in_range_forward:
   bnei a0, 10, .L2
   .fill 1000, 4, 0
.L2:
   ret

# Tests that are backward in range.

# CHECK-INST-LABEL: beqi_in_range_backward
# CHECK-INST:       beqi a0, 0xa, {{.*}}

# CHECK-INST-RELAX-LABEL: beqi_in_range_backward
# CHECK-INST-RELAX:       beqi a0, 0xa, {{.*}}
beqi_in_range_backward:
.L3:
   .fill 1000, 4, 0
   beqi a0, 10, .L3
   ret

# CHECK-INST-LABEL: bnei_in_range_backward
# CHECK-INST:       bnei a0, 0xa, {{.*}}

# CHECK-INST-RELAX-LABEL: bnei_in_range_backward
# CHECK-INST-RELAX:       bnei a0, 0xa, {{.*}}
bnei_in_range_backward:
.L4:
   .fill 1000, 4, 0
   bnei a0, 10, .L4
   ret

# Tests that are forward out of range.

# CHECK-INST-LABEL: beqi_out_of_range_forward
# CHECK-INST:       bnei a0, 0xa, {{.*}}+0x8
# CHECK-INST-NEXT:  jal zero, {{.*}}

# CHECK-INST-RELAX-LABEL: beqi_out_of_range_forward
# CHECK-INST-RELAX:       bnei a0, 0xa, {{.*}}+0x8
# CHECK-INST-RELAX-NEXT:  jal zero, {{.*}}
beqi_out_of_range_forward:
   beqi a0, 10, .L5
   .fill 1300, 4, 0
.L5:
   ret

# CHECK-INST-LABEL: bnei_out_of_range_forward
# CHECK-INST:       beqi a0, 0xa, {{.*}}+0x8
# CHECK-INST-NEXT:  jal zero, {{.*}}

# CHECK-INST-RELAX-LABEL: bnei_out_of_range_forward
# CHECK-INST-RELAX:       beqi a0, 0xa, {{.*}}+0x8
# CHECK-INST-RELAX-NEXT:  jal zero, {{.*}}
bnei_out_of_range_forward:
   bnei a0, 10, .L6
   .fill 1300, 4, 0
.L6:
   ret

# Tests that are backward out of range.

# CHECK-INST-LABEL: beqi_out_of_range_backward
# CHECK-INST:       bnei a0, 0xa, {{.*}}+0x1458
# CHECK-INST-NEXT:  jal zero, {{.*}}

# CHECK-INST-RELAX-LABEL: beqi_out_of_range_backward
# CHECK-INST-RELAX:       bnei a0, 0xa, {{.*}}+0x1458
# CHECK-INST-RELAX-NEXT:  jal zero, {{.*}}
beqi_out_of_range_backward:
.L7:
   .fill 1300, 4, 0
   beqi a0, 10, .L7
   ret

# CHECK-INST-LABEL: bnei_out_of_range_backward
# CHECK-INST:       beqi a0, 0xa, {{.*}}+0x1458
# CHECK-INST-NEXT:  jal zero, {{.*}}

# CHECK-INST-RELAX-LABEL: bnei_out_of_range_backward
# CHECK-INST-RELAX:       beqi a0, 0xa, {{.*}}+0x1458
# CHECK-INST-RELAX-NEXT:  jal zero, {{.*}}
bnei_out_of_range_backward:
.L8:
   .fill 1300, 4, 0
   bnei a0, 10, .L8
   ret
