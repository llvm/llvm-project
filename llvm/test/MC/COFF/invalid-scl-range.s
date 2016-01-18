# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s

	.def storage_class_range
		.scl 1337
	.endef

