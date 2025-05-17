# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcilb %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcilb %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump -dr --mattr=+experimental-xqcilb - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we emit the right relocations for Xqcilb
## relative jumps when relocations are enabled. These require a
## vendor-specific relocation pair.

# These are required to turn off autocompression, but to re-enable
# linker relaxation.
.option exact
.option relax

# ASM-LABEL: this_section:
# OBJ-LABEL: <this_section>:
this_section:

# ASM: qc.e.j same_section
# OBJ: qc.e.j 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM195 same_section{{$}}
# OBJ-NOT: R_RISCV
qc.e.j same_section

# ASM: qc.e.jal same_section
# OBJ: qc.e.jal 0x6
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM195 same_section{{$}}
# OBJ-NOT: R_RISCV
qc.e.jal same_section

# ASM-LABEL: same_section:
# OBJ-LABEL: <same_section>:
same_section:
nop
