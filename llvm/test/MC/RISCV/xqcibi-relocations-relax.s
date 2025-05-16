# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump -dr --mattr=+experimental-xqcibi - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we emit the right relocations for Xqcibi
## relative branches, when relaxations are enabled. These require a relocation.
## The QC.E.B<op>I instructions also require a vendor relocation.

# These are required to turn off autocompression, but to re-enable
# linker relaxation.
.option exact
.option relax

# ASM-LABEL: this_section:
# OBJ-LABEL: <this_section>:
this_section:

# ASM: qc.bnei t2, 11, same_section
# OBJ: qc.bnei t2, 0xb, 0x0 <this_section>
# OBJ-NEXT: R_RISCV_BRANCH same_section{{$}}
# OBJ-NOT: R_RISCV
qc.bnei t2, 11, same_section

# ASM: qc.e.bgeui s1, 21, same_section
# OBJ: qc.e.bgeui s1, 0x15, 0x4 <this_section+0x4>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM193 same_section{{$}}
# OBJ-NOT: R_RISCV
qc.e.bgeui s1, 21, same_section

# ASM-LABEL: same_section:
# OBJ-LABEL: <same_section>:
same_section:
nop

