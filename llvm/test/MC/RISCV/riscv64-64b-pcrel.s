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
# CHECK-NEXT: ]

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

## .apple_names/.apple_types are fixed-size and do not need fixups.
## llvm-dwarfdump --apple-names does not process R_RISCV_{ADD,SUB}32 in them.
## See llvm/test/DebugInfo/Generic/accel-table-hash-collisions.ll
	.section	.apple_types
        .word 0
        .word .Ltypes0-.apple_types
.Ltypes0:
