# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck %s

# Ensure that the eh_frame records the symbolic difference with the paired
# relocations always.

func:
	.cfi_startproc
  ret
	.cfi_endproc

# CHECK:   Section (4) .rela.eh_frame {
# CHECK-NEXT:   0x1C R_RISCV_32_PCREL - 0x0
# CHECK-NEXT:   0x20 R_RISCV_ADD32 - 0x0
# CHECK-NEXT:   0x20 R_RISCV_SUB32 - 0x0
# CHECK-NEXT: }
