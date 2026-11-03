# RUN: not llvm-mc -triple=wasm32-unknown-unknown -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s

# The object writer also rejects a v128 global that comes from assembly.

# CHECK: error: WebAssembly global 'g' has a type that cannot be initialized
	.globaltype	g, v128
	.globl	g
g:
