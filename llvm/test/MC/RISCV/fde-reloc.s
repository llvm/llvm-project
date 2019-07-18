# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELAX-RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=NORELAX-RELOC %s

func:
	.cfi_startproc
  ret
	.cfi_endproc

# TODO: Should produce R_RISCV_32_PCREL for the FDE pc relocation. Many of the
# ADD32/SUB32 relocations also can be safely resolved even with linker
# relaxation enabled. This test is written to capture current behaviour, in
# preparation for follow-on patches to fix it.

# RELAX-RELOC:   Section (4) .rela.eh_frame {
# RELAX-RELOC-NEXT:   0x0 R_RISCV_ADD32 - 0xFFFFFFFC
# RELAX-RELOC-NEXT:   0x0 R_RISCV_SUB32 - 0x0
# RELAX-RELOC-NEXT:   0x14 R_RISCV_ADD32 - 0x0
# RELAX-RELOC-NEXT:   0x14 R_RISCV_SUB32 - 0x0
# RELAX-RELOC-NEXT:   0x18 R_RISCV_ADD32 - 0x0
# RELAX-RELOC-NEXT:   0x18 R_RISCV_SUB32 - 0x0
# RELAX-RELOC-NEXT:   0x1C R_RISCV_ADD32 - 0x0
# RELAX-RELOC-NEXT:   0x1C R_RISCV_SUB32 - 0x0
# RELAX-RELOC-NEXT:   0x20 R_RISCV_ADD32 - 0x0
# RELAX-RELOC-NEXT:   0x20 R_RISCV_SUB32 - 0x0
# RELAX-RELOC-NEXT: }

# NORELAX-RELOC:        Section (4) .rela.eh_frame {
# NORELAX-RELOC-NEXT:    0x1C R_RISCV_32_PCREL - 0x0
# NORELAX-RELOC-NEXT:  }
