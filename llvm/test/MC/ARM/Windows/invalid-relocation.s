# RUN: not llvm-mc -triple thumbv7-windows -incremental-linker-compatible -filetype obj -o /dev/null 2>&1 %s \
# RUN:     | FileCheck %s

	.def invalid_relocation
		.type 32
		.scl 2
	.endef
	.global invalid_relocation
	.thumb_func
	adr r0, invalid_relocation+1

# CHECK: 10:2: error: unsupported relocation type
