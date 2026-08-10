# RUN: llvm-mc -triple riscv64 -filetype obj -o - %s | llvm-readobj -r - | FileCheck %s

	.section __jump_table,"aw",@progbits
	.p2align 3
.Ltmp0:
	.quad	(function+128)-.Ltmp0

# CHECK: .rela__jump_table {
# CHECK:   0x0 R_RISCV_ADD64 function 0x80
# CHECK-NEXT:   0x0 R_RISCV_SUB64 .Ltmp0 0x0
# CHECK: }
