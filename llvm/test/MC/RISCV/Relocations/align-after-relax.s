# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s --defsym LATE=1 -o %t1
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases %t1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o %t0
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases %t0 | FileCheck %s --check-prefix=CHECK0

# CHECK:            6: c.nop
# CHECK-EMPTY:
# CHECK:           10: auipc   ra, 0x0
# CHECK-NEXT:                R_RISCV_CALL_PLT     foo
# CHECK-NEXT:                R_RISCV_RELAX        *ABS*
# CHECK:           20: c.nop
# CHECK-NEXT:                R_RISCV_ALIGN        *ABS*+0x6

## Alignment directives in a smaller-number subsection might be conservatively treated as linker-relaxable.
# CHECK0:           6: c.nop
# CHECK0-NEXT:               R_RISCV_ALIGN        *ABS*+0x6
# CHECK0:          20: c.nop
# CHECK0-NEXT:               R_RISCV_ALIGN        *ABS*+0x6

.text 2
.option push
.option norelax
## R_RISCV_ALIGN is required even if norelax, because there is a preceding linker-relaxable instruction.
.balign 8
l2:
  .word 0x12345678
.option pop

.text 1
  .space 4
.ifdef LATE
  .space 0
.endif
  call foo
.ifdef LATE
  .space 8
.else
  .space 4
.endif

.text 0
_start:
  .space 6
.option push
.option norelax
.balign 8
l0:
  .word 0x12345678
.option pop
