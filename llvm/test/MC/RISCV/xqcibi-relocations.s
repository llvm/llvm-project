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
qc.bnei t1, 10, undef

# ASM: qc.e.bgeui s0, 20, undef
# OBJ-NEXT: qc.e.bgeui s0, 0x14, 0x4 <this_section+0x4>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM193 undef{{$}}
qc.e.bgeui s0, 20, undef


# ASM: qc.bnei t2, 11, same_section
# OBJ-NEXT: qc.bnei t2, 0xb, 0x28 <same_section>
qc.bnei t2, 11, same_section

# ASM: qc.e.bgeui s1, 21, same_section
# OBJ-NEXT: qc.e.bgeui s1, 0x15, 0x28 <same_section>
qc.e.bgeui s1, 21, same_section


# ASM: qc.bnei t2, 12, same_section_extern
# OBJ-NEXT: qc.bnei t2, 0xc, 0x14 <this_section+0x14>
# OBJ-NEXT: R_RISCV_BRANCH same_section_extern{{$}}
qc.bnei t2, 12, same_section_extern

# ASM: qc.e.bgeui s1, 22, same_section_extern
# OBJ-NEXT: qc.e.bgeui s1, 0x16, 0x18 <this_section+0x18>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM193 same_section_extern{{$}}
qc.e.bgeui s1, 22, same_section_extern


# ASM: qc.bnei t3, 13, other_section
# OBJ-NEXT: qc.bnei t3, 0xd, 0x1e <this_section+0x1e>
# OBJ-NEXT: R_RISCV_BRANCH other_section{{$}}
qc.bnei t3, 13, other_section

# ASM: qc.e.bgeui s2, 23, other_section
# OBJ-NEXT: qc.e.bgeui s2, 0x17, 0x22 <this_section+0x22>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM193 other_section{{$}}
qc.e.bgeui s2, 23, other_section


# ASM-LABEL: same_section:
# OBJ-LABEL: <same_section>:
same_section:
  nop

# ASM-LABEL: same_section_extern:
# OBJ-LABEL: <same_section_extern>:
  .global same_section_extern
same_section_extern:
  nop

.option relax

# ASM: qc.bnei t3, 14, same_section
# OBJ: qc.bnei t3, 0xe, 0x28 <same_section>
qc.bnei t3, 14, same_section

# ASM: qc.e.bgeui s2, 24, same_section
# OBJ-NEXT: qc.e.bgeui s2, 0x18, 0x28 <same_section>
qc.e.bgeui s2, 24, same_section

## Enable compression/relaxation to check how symbols are handled.
.option noexact

# ASM: qc.bnei t1, 10, undef
# OBJ: qc.beqi t1, 0xa, 0x42 <same_section_extern+0x16>
# OBJ-NEXT: j 0x3e <same_section_extern+0x12>
# OBJ-NEXT: R_RISCV_JAL undef{{$}}
qc.bnei t1, 10, undef

# ASM: qc.e.bgeui s0, 40, undef
# OBJ-NEXT: qc.e.bltui s0, 0x28, 0x4c <same_section_extern+0x20>
# OBJ-NEXT: j 0x48 <same_section_extern+0x1c>
# OBJ-NEXT: R_RISCV_JAL undef{{$}}
qc.e.bgeui s0, 40, undef

.section .text.second, "ax", @progbits

# ASM-LABEL: other_section:
# OBJ-LABEL: <other_section>:
other_section:
  nop
