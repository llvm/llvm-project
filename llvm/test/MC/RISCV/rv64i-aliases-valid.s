# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefix=CHECK-ALIAS %s

# TODO ld
# TODO sd

# CHECK-INST: subw t6, zero, ra
# CHECK-ALIAS: negw t6, ra
negw x31, x1
# CHECK-INST: addiw t6, ra, 0
# CHECK-ALIAS: sext.w t6, ra
sext.w x31, x1
