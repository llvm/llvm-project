# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

# Test coverage for HexagonAsmParser directives: .falign and .subsection.

# CHECK-LABEL: <test_falign>:
# .falign inserts NOP padding to align the following packet to a fetch boundary.
# CHECK: nop
# CHECK: jumpr r31
.globl test_falign
test_falign:
  nop
  .falign
  jumpr lr

# .subsection directive switches to a numbered subsection.
.subsection 1
# CHECK-LABEL: <test_subsection>:
# CHECK: jumpr r31
.globl test_subsection
test_subsection:
  nop
  jumpr lr
