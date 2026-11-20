# RUN: not llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown %s | not llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o /dev/null 2>&1 | FileCheck %s

# An undefined global is imported, so the value given here would be lost.

# CHECK: error: WebAssembly global 'g' has an initial value but is not defined
	.globaltype	g, i32
	.globalinit	g, i32.const 5

# Also for a value equal to the type's default.
# CHECK: error: WebAssembly global 'z' has an initial value but is not defined
	.globaltype	z, i32
	.globalinit	z, i32.const 0

	.functype	use () -> (i32)
	.globl	use
use:
	.functype	use () -> (i32)
	global.get	g
	global.get	z
	i32.add
	end_function
