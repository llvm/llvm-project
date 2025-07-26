# RUN: llvm-mc --triple=riscv32 -mattr=+relax,+experimental-xqcilb \
# RUN:    %s -filetype=obj -o - -riscv-add-build-attributes \
# RUN:    | llvm-objdump -dr - \
# RUN:    | FileCheck %s

.global foo

bar:
  jal x1, foo
# CHECK: qc.e.jal 0x0 <bar>
# CHECK-NEXT: R_RISCV_VENDOR QUALCOMM
# CHECK-NEXT: R_RISCV_CUSTOM195 foo
# CHECK-NEXT: R_RISCV_RELAX *ABS*
  bne a0, a1, bar
# CHECK-NEXT: bne a0, a1, 0x6 <bar+0x6>
# CHECK-NEXT: R_RISCV_BRANCH bar
  ret
# CHECK-NEXT: ret
