# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcili %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcili %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump -dr --mattr=+experimental-xqcili - \
# RUN:     | FileCheck -check-prefix=OBJ %s

## This test checks that we emit the right relocations for Xqcili
## immediates. These always require a relocation pair, unless the
## target is absolute.

# This is required so that the conditional branches requiring relocations
# are not converted into inverted branches with long jumps by the assembler.
.option exact

.set abs_symbol, 0x0

# ASM-LABEL: this_section:
# OBJ-LABEL: <this_section>:
this_section:

# ASM: qc.li a0, %qc.abs20(undef)
# OBJ: qc.li a0, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM192 undef{{$}}
qc.li a0, %qc.abs20(undef)

# ASM: qc.e.li s0, undef
# OBJ-NEXT: qc.e.li s0, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM194 undef{{$}}
qc.e.li s0, undef


# ASM: qc.li a1, %qc.abs20(0)
# OBJ-NEXT: qc.li a1, 0x0
qc.li a1, %qc.abs20(abs_symbol)

# ASM: qc.e.li s1, 0
# OBJ-NEXT: qc.e.li s1, 0x0
qc.e.li s1, abs_symbol


# ASM: qc.li a2, %qc.abs20(same_section)
# OBJ-NEXT: qc.li a2, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM192 same_section{{$}}
qc.li a2, %qc.abs20(same_section)

# ASM: qc.e.li s2, same_section
# OBJ-NEXT: qc.e.li s2, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM194 same_section{{$}}
qc.e.li s2, same_section

# ASM: qc.li a3, %qc.abs20(other_section)
# OBJ-NEXT: qc.li a3, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM192 other_section{{$}}
qc.li a3, %qc.abs20(other_section)

# ASM: qc.e.li s3, other_section
# OBJ-NEXT: qc.e.li s3, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM194 other_section{{$}}
qc.e.li s3, other_section

# ASM-LABEL: same_section:
# OBJ-LABEL: <same_section>:
same_section:
  nop

.option relax

# ASM: qc.li a1, %qc.abs20(0)
# OBJ: qc.li a1, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM192 *ABS*{{$}}
# OBJ-NEXT: R_RISCV_RELAX
qc.li a1, %qc.abs20(abs_symbol)

# ASM: qc.e.li s1, 0
# OBJ-NEXT: qc.e.li s1, 0x0
qc.e.li s1, abs_symbol

# ASM: qc.li a1, %qc.abs20(undef)
# OBJ-NEXT: qc.li a1, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM192 undef{{$}}
# OBJ-NEXT: R_RISCV_RELAX
qc.li a1, %qc.abs20(undef)

# ASM: qc.e.li s1, undef
# OBJ-NEXT: qc.e.li s1, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM194 undef{{$}}
# OBJ-NEXT: R_RISCV_RELAX
qc.e.li s1, undef

## Enable compression/relaxation to check how symbols are handled.
.option noexact

# ASM: qc.li a1, %qc.abs20(undef)
# OBJ-NEXT: qc.li a1, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM192 undef{{$}}
# OBJ-NEXT: R_RISCV_RELAX
qc.li a1, %qc.abs20(undef)

# ASM: qc.e.li a2, undef
# OBJ-NEXT: qc.e.li a2, 0x0
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_CUSTOM194 undef{{$}}
# OBJ-NEXT: R_RISCV_RELAX
qc.e.li a2, undef

.section .text.other, "ax", @progbits

# ASM-LABEL: other_section:
# OBJ-LABEL: <other_section>:
other_section:
  nop
