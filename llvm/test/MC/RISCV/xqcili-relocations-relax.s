# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcili %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcili %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump -dr --mattr=+experimental-xqcili - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we emit the right relocations for Xqcili
## immediates, when relaxations are enabled. These don't require
## relocations, but the behaviour is different depending on whether
## there is a target-specific symbol specifier or not (there is 
## one for `qc.li`).

## It might be more obvious if both of these did the same thing,
## either both emitting relocations, or both not emitting relocations.

# These are required to turn off autocompression, but to re-enable
# linker relaxation.
.option exact
.option relax

.set abs_symbol, 0x0

# ASM-LABEL: this_section:
# OBJ-LABEL: <this_section>:
this_section:


# ASM: qc.li a1, %qc.abs20(0)
# OBJ: qc.li a1, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM192 *ABS*{{$}}
# OBJ-NOT: R_RISCV
qc.li a1, %qc.abs20(abs_symbol)

# ASM: qc.e.li s1, 0
# OBJ: qc.e.li s1, 0x0
# OBJ-NOT: R_RISCV
qc.e.li s1, abs_symbol

# ASM-LABEL: same_section:
# OBJ-LABEL: <same_section>:
same_section:
  nop

.section .text.other, "ax", @progbits

# ASM-LABEL: other_section:
# OBJ-LABEL: <other_section>:
other_section:
  nop
