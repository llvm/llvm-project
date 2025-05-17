# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcilb %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcilb %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump -dr --mattr=+experimental-xqcilb - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we emit the right relocations for Xqcilb
## relative jumps. These can be resolved within the same section
## (when relaxations are disabled) but otherwise require a
## vendor-specific relocation pair.

# This is required so that the conditional jumps are not compressed
# by the assembler
.option exact

# ASM-LABEL: this_section:
# OBJ-LABEL: <this_section>:
this_section:

# ASM: qc.e.j undef
# OBJ: qc.e.j 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM195 undef{{$}}
# OBJ-NOT: R_RISCV
qc.e.j undef

# ASM: qc.e.jal undef
# OBJ: qc.e.jal 0x6
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM195 undef{{$}}
# OBJ-NOT: R_RISCV
qc.e.jal undef


# ASM: qc.e.j same_section
# OBJ: qc.e.j 0x24
# OBJ-NOT: R_RISCV
qc.e.j same_section

# ASM: qc.e.jal same_section
# OBJ: qc.e.jal 0x24
# OBJ-NOT: R_RISCV
qc.e.jal same_section


# ASM: qc.e.j other_section
# OBJ: qc.e.j 0x18
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM195 other_section{{$}}
# OBJ-NOT: R_RISCV
qc.e.j other_section

# ASM: qc.e.jal other_section
# OBJ: qc.e.jal 0x1e
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM195 other_section{{$}}
# OBJ-NOT: R_RISCV
qc.e.jal other_section


# ASM-LABEL: same_section:
# OBJ-LABEL: <same_section>:
same_section:
  nop

.section .text.other, "ax", @progbits

# ASM-LABEL: other_section:
# OBJ-LABEL: <other_section>:
other_section:
  nop
