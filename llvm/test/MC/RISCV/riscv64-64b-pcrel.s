# RUN: llvm-mc -triple riscv64-unknown-linux-gnu -filetype obj -o - %s \
# RUN:   | llvm-readobj -r - | FileCheck %s

# CHECK: Relocations [
# CHECK:  .relasx {
# CHECK-NEXT:    0x0 R_RISCV_ADD64 y 0x0
# CHECK-NEXT:    0x0 R_RISCV_SUB64 x 0x0
# CHECK:  }
# CHECK:  .relasy {
# CHECK-NEXT:    0x0 R_RISCV_ADD64 x 0x0
# CHECK-NEXT:    0x0 R_RISCV_SUB64 y 0x0
# CHECK:  }
# CHECK:  .relasz {
# CHECK-NEXT:    0x0 R_RISCV_ADD64 z 0x0
# CHECK-NEXT:    0x0 R_RISCV_SUB64 a 0x0
# CHECK:  }
# CHECK:  .relasa {
# CHECK-NEXT:    0x0 R_RISCV_ADD64 a 0x0
# CHECK-NEXT:    0x0 R_RISCV_SUB64 z 0x0
# CHECK:  }
# CHECK: ]

	.section	sx,"aw",@progbits
x:
	.quad y-x

	.section	sy,"aw",@progbits
y:
	.quad x-y

	.section	sz
z:
	.quad z-a

	.section	sa
a:
	.quad a-z
