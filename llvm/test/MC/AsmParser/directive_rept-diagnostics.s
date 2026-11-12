# RUN: not llvm-mc -triple i686-elf -filetype asm -o /dev/null %s 2>&1 \
# RUN:   | FileCheck %s

	.data

	.global invalid_expression
	.type invalid_expression,@object
invalid_expression:
	.rept *

# CHECK: error: invalid token in expression
# CHECK: 	.rept *
# CHECK:              ^

	.global bad_token
	.type bad_token,@object
bad_token:
	.rept bad_token

# CHECK: error: unexpected token in '.rept' directive
# CHECK: 	.rept bad_token
# CHECK:              ^

	.global negative
	.type negative,@object
negative:
	.rept -32

# CHECK: error: Count is negative
# CHECK: 	.rept -32
# CHECK:              ^

	.global trailer
	.type trailer,@object
trailer:
	.rep 0 trailer

# CHECK: :[[#@LINE-2]]:9: error: expected newline
# CHECK: 	.rep 0 trailer
# CHECK:               ^

