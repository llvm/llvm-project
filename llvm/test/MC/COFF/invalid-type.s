# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s

	.type 65536

