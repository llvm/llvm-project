# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump -dr --mattr=+experimental-xqcibi - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we emit the right relocations for Xqcibi
## relative branches. These can be resolved within the same section
## (when relaxations are disabled) but otherwise require a relocation.
## The QC.E.B<op>I instructions also require a vendor relocation.

# This is required so that the conditional branches requiring relocations
# are not converted into inverted branches with long jumps by the assembler.
.option exact

# ASM-LABEL: this_section:
# OBJ-LABEL: <this_section>:
this_section:

# ASM: qc.bnei t1, 10, undef
# OBJ: qc.bnei t1, 0xa, 0x0 <this_section>
# OBJ-NEXT: R_RISCV_BRANCH undef{{$}}
# OBJ-NOT: R_RISCV
qc.bnei t1, 10, undef

# ASM: qc.e.bgeui s0, 20, undef
# OBJ: qc.e.bgeui s0, 0x14, 0x4 <this_section+0x4>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM193 undef{{$}}
# OBJ-NOT: R_RISCV
qc.e.bgeui s0, 20, undef


# ASM: qc.bnei t2, 11, same_section
# OBJ: qc.bnei t2, 0xb, 0x1e <same_section>
# OBJ-NOT: R_RISCV
qc.bnei t2, 11, same_section

# ASM: qc.e.bgeui s1, 21, same_section
# OBJ: qc.e.bgeui s1, 0x15, 0x1e <same_section>
# OBJ-NOT: R_RISCV
qc.e.bgeui s1, 21, same_section


# ASM: qc.bnei t3, 12, other_section
# OBJ: qc.bnei t3, 0xc, 0x14 <this_section+0x14>
# OBJ-NEXT: R_RISCV_BRANCH other_section{{$}}
# OBJ-NOT: R_RISCV
qc.bnei t3, 12, other_section

# ASM: qc.e.bgeui s2, 22, other_section
# OBJ: qc.e.bgeui s2, 0x16, 0x18 <this_section+0x18>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM193 other_section{{$}}
# OBJ-NOT: R_RISCV
qc.e.bgeui s2, 22, other_section


# ASM-LABEL: same_section:
# OBJ-LABEL: <same_section>:
same_section:
  nop

.section .text.second, "ax", @progbits

# ASM-LABEL: other_section:
# OBJ-LABEL: <other_section>:
other_section:
  nop
