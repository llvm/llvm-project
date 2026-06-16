# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=aarch64_be-unknown-linux-gnu -filetype=obj \
# RUN:   -o %t/elf_aarch64_be.o %s
# RUN: llvm-jitlink -noexec -abs external_data=0xdeadbeef \
# RUN:   -check %s %t/elf_aarch64_be.o

# Test that AArch64 big-endian (BE8) objects link: data fixups are written
# in target (big) endianness while instruction fixups stay little-endian
# (the A64 ISA is word-invariant, so instructions are always LE-encoded).

	.text
	.globl	main
	.p2align	2
	.type	main,@function
main:
	ret
	.size	main, .-main

# Instruction fixup (R_AARCH64_CALL26): the BL immediate is patched into an
# LE-encoded instruction word, even in a big-endian object.
# jitlink-check: decode_operand(local_func_call26, 0)[25:0] = (local_func - local_func_call26)[27:2]
	.globl	local_func
	.p2align	2
	.type	local_func,@function
local_func:
	ret
	.size	local_func, .-local_func

	.globl	local_func_call26
	.p2align	2
local_func_call26:
	bl	local_func
	.size	local_func_call26, .-local_func_call26

# Data fixups: absolute addresses are stored big-endian.
# jitlink-check: *{8}(local_func_addr_quad) = local_func
# jitlink-check: *{4}(external_data_addr_word) = external_data
	.data
	.globl	local_func_addr_quad
	.p2align	3
local_func_addr_quad:
	.quad	local_func

	.globl	external_data_addr_word
	.p2align	2
external_data_addr_word:
	.word	external_data
