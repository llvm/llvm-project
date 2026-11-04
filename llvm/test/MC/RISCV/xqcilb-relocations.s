# RUN: llvm-mc -triple riscv32 -mattr=+xqcilb %s \
# RUN:     | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple riscv32 -mattr=+xqcilb %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-objdump -dr --mattr=+xqcilb - \
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
# OBJ: qc.e.j 0x0 <this_section>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT undef{{$}}
qc.e.j undef

# ASM: qc.e.jal undef
# OBJ-NEXT: qc.e.jal 0x6 <this_section+0x6>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT undef{{$}}
qc.e.jal undef


# ASM: qc.e.j same_section
# OBJ-NEXT: qc.e.j 0x30 <same_section>
qc.e.j same_section

# ASM: qc.e.jal same_section
# OBJ-NEXT: qc.e.jal 0x30 <same_section>
qc.e.jal same_section

# ASM: qc.e.j same_section_extern
# OBJ-NEXT: qc.e.j 0x18 <this_section+0x18>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT same_section_extern{{$}}
qc.e.j same_section_extern

# ASM: qc.e.jal same_section_extern
# OBJ-NEXT: qc.e.jal 0x1e <this_section+0x1e>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT same_section_extern{{$}}
qc.e.jal same_section_extern


# ASM: qc.e.j other_section
# OBJ-NEXT: qc.e.j 0x24 <this_section+0x24>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT other_section{{$}}
qc.e.j other_section

# ASM: qc.e.jal other_section
# OBJ-NEXT: qc.e.jal 0x2a <this_section+0x2a>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT other_section{{$}}
qc.e.jal other_section


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

# ASM: qc.e.j same_section
# OBJ: qc.e.j 0x38 <same_section_extern+0x4>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT same_section{{$}}
# OBJ-NEXT: R_RISCV_RELAX
qc.e.j same_section

# ASM: qc.e.jal same_section
# OBJ-NEXT: qc.e.jal 0x3e <same_section_extern+0xa>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT same_section{{$}}
# OBJ-NEXT: R_RISCV_RELAX
qc.e.jal same_section

## Enable compression/relaxation to check how symbols are handled.
.option noexact

qc.e.j undef
# ASM: j undef
# OBJ: qc.e.j 0x44 <same_section_extern+0x10>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT undef{{$}}
# OBJ-NEXT: R_RISCV_RELAX

qc.e.jal undef
# ASM: jal undef
# OBJ: qc.e.jal 0x4a <same_section_extern+0x16>
# OBJ-NEXT: R_RISCV_VENDOR QUALCOMM{{$}}
# OBJ-NEXT: R_RISCV_QC_E_CALL_PLT undef{{$}}
# OBJ-NEXT: R_RISCV_RELAX

.section .text.other, "ax", @progbits

# ASM-LABEL: other_section:
# OBJ-LABEL: <other_section>:
other_section:
  nop
