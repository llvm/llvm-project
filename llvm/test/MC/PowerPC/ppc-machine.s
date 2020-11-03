# RUN: llvm-mc -triple powerpc-unknown-unknown %s
# RUN: llvm-mc -triple powerpc64-unknown-unknown %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown -filetype=null %s

# For now, the only thing we check is that the .machine directive
# is accepted without syntax error.

	.machine push
	.machine any
	.machine pop

	.machine "push"
	.machine "any"
	.machine "pop"

	.machine ppc64
