# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvbi %s \
# RUN:   | llvm-objdump -dr -M no-aliases - \
# RUN:   | FileCheck %s

# cv.beqimm / cv.bneimm encode a 13-bit signed PC-relative offset
# (+/-4094 bytes). A branch whose target is out of that range must be
# relaxed by the assembler into an inverted short branch over a JAL
# trampoline. An in-range branch stays a single instruction.

.text

# CHECK-LABEL: <far_beqimm>:
# CHECK:        cv.bneimm a0, 0x3, 0x{{[0-9a-f]+}}
# CHECK-NEXT:   jal zero, 0x{{[0-9a-f]+}} <target1>
far_beqimm:
  cv.beqimm a0, 3, target1
  .space 8192
target1:
  ret

# CHECK-LABEL: <far_bneimm>:
# CHECK:        cv.beqimm a1, -0x5, 0x{{[0-9a-f]+}}
# CHECK-NEXT:   jal zero, 0x{{[0-9a-f]+}} <target2>
far_bneimm:
  cv.bneimm a1, -5, target2
  .space 8192
target2:
  ret

# An in-range branch is not relaxed: a single cv.beqimm to the target.
# CHECK-LABEL: <near_beqimm>:
# CHECK:        cv.beqimm a0, 0x1, 0x{{[0-9a-f]+}} <target3>
# CHECK-NOT:    jal zero
near_beqimm:
  cv.beqimm a0, 1, target3
target3:
  ret
