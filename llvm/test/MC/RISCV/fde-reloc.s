# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s \
# RUN:     | llvm-readobj -r -x .eh_frame - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=-relax < %s \
# RUN:     | llvm-readobj -r -x .eh_frame - | FileCheck %s

# Ensure that the eh_frame records the symbolic difference with the paired
# relocations always.

func:
	.cfi_startproc
  ret
	.cfi_endproc

# CHECK:   Section (4) .rela.eh_frame {
# CHECK-NEXT:   0x1C R_RISCV_32_PCREL .L0  0x0
# CHECK-NEXT: }
# CHECK:      Hex dump of section '.eh_frame':
# CHECK-NEXT: 0x00000000 10000000 00000000 017a5200 017c0101
# CHECK-NEXT: 0x00000010 1b0c0200 10000000 18000000 00000000
# CHECK-NEXT: 0x00000020 04000000 00000000
#                        ^ address_range
