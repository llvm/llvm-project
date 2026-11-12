# RUN: not llvm-mc -triple=wasm32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

# Re-declaring a global as another kind of symbol would drop its value.

	.globaltype	fn, i32
# CHECK: :[[@LINE+1]]:14: error: .globalinit is only valid for a global: fn
	.globalinit	fn, i32.const 1
	.functype	fn () -> ()
	.globl	fn
fn:
	.functype	fn () -> ()
	end_function

	.globaltype	obj, i32
# CHECK: :[[@LINE+1]]:14: error: .globalinit is only valid for a global: obj
	.globalinit	obj, i32.const 2
	.type	obj,@object
	.section	.data.obj,"",@
	.globl	obj
obj:
	.int32	0
	.size	obj, 4

	.globaltype	tbl, i32
# CHECK: :[[@LINE+1]]:14: error: .globalinit is only valid for a global: tbl
	.globalinit	tbl, i32.const 3
	.tabletype	tbl, externref

	.globaltype	tag, i32
# CHECK: :[[@LINE+1]]:14: error: .globalinit is only valid for a global: tag
	.globalinit	tag, i32.const 4
	.tagtype	tag i32
