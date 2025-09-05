# RUN: llvm-mc -filetype=obj -triple riscv32  < %s \
# RUN:     --mattr=+experimental-xqcili,+experimental-xqcilb,+experimental-xqcibi \
# RUN:     -riscv-add-build-attributes \
# RUN:     | llvm-objdump --no-print-imm-hex -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INSTR %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s \
# RUN:     --mattr=+experimental-xqcili,+experimental-xqcilb,+experimental-xqcibi \
# RUN:     | llvm-readobj -r - | FileCheck %s -check-prefix=CHECK-REL

## This checks that, if the assembler can resolve the qc fixup, that the fixup
## is applied correctly to the instruction.

.L0:
# CHECK-INSTR: qc.e.beqi a0, 64, 0x0
qc.e.beqi a0, 64, .L0
# CHECK-INSTR: qc.e.j 0x10000016
qc.e.j func
# CHECK-INSTR: qc.e.li a0, 8
qc.e.li a0, abs_sym
# CHECK-INSTR: qc.li a0, 8
qc.li a0, %qc.abs20(abs_sym)



# This has to come after the instructions that use it or it will
# be evaluated at parse-time (avoiding fixups)
abs_sym = 8


.space 0x10000000
func:
  ret

## All these fixups should be resolved by the assembler without emitting
## relocations.
# CHECK-REL-NOT: R_RISCV
