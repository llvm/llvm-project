# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s

	.def invalid_type_range
		.type 65536
	.endef

