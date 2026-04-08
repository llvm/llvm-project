# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s --defsym LATE=1 -o %t1
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases %t1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o %t0
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases %t0 | FileCheck %s --check-prefix=CHECK0

# CHECK:            4: 00 00 01 00 .word 0x00010000
# CHECK-EMPTY:
# CHECK:            8: 78 56 34 12 .word 0x12345678
# CHECK-NEXT:       c: 00 00 00 00 .word 0x00000000
# CHECK:           10: auipc   ra, 0x0
# CHECK-NEXT:                R_RISCV_CALL_PLT     foo
# CHECK-NEXT:                R_RISCV_RELAX        *ABS*
# CHECK:           18: c.nop
# CHECK-NEXT:                R_RISCV_ALIGN        *ABS*+0x6

## Alignment directives in a lower-numbered subsection may be conservatively treated as linker-relaxable.
# CHECK0:           4: 00 00 01 00 .word 0x00010000
# CHECK0-NEXT:               000000006: R_RISCV_ALIGN        *ABS*+0x6
# CHECK0-NEXT:      8: 13 00 00 00 .word 0x00000013
# CHECK0:          14: auipc   ra, 0x0
# CHECK0:          1c: c.nop
# CHECK0-NEXT:               R_RISCV_ALIGN        *ABS*+0x6

.text 2
.option push
.option norelax
## R_RISCV_ALIGN is required even if norelax, because it is after a linker-relaxable instruction.
.balign 8
l2:
  .word 0x12345678
.option pop

.text 1
  .org .+1
  .org .+3
.ifdef LATE
  .org .+0
.endif
  call foo

.text 0
_start:
  .space 6
.option push
.option norelax
.balign 8
l0:
  .word 0x12345678
.option pop
