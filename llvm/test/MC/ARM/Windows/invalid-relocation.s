# RUN: not llvm-mc -triple thumbv7-windows -filetype obj -o /dev/null 2>&1 %s \
# RUN:     | FileCheck %s

	.def invalid_relocation
		.type 32
		.scl 2
	.endef
	.global invalid_relocation
	.thumb_func
invalid_relocation:
	adr r0, invalid_relocation+1

# CHECK: LLVM ERROR: unsupported relocation type: fixup_t2_adr_pcrel_12

