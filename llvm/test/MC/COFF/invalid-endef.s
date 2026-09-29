# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s 2>&1 | FileCheck %s

# CHECK: ending symbol definition without starting one
# CHECK: ending symbol definition without starting one
	.endef
	.endef

