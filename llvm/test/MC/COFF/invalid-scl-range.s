# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s

	.def storage_class_range
# CHECK: storage class value '1337' out of range
# CHECK: storage class value '9001' out of range
		.scl 1337
		.scl 9001
	.endef

