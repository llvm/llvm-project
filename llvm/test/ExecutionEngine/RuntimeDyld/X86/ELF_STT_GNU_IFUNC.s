# REQUIRES: x86_64-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: split-file %s %t
# RUN: llvm-mc -triple=x86_64-unknown-linux-gnu -filetype=obj -o %t/test_runner.o %t/test_runner.s
# RUN: llvm-mc -triple=x86_64-unknown-linux-gnu -filetype=obj -o %t/func_defs.o %t/func_defs.s
# RUN: llvm-rtdyld -triple=x86_64-unknown-linux-gnu -verify -check=%s %t/test_runner.o %t/func_defs.o
# RUN: llvm-rtdyld -triple=x86_64-unknown-linux-gnu -execute %t/test_runner.o %t/func_defs.o

#--- test_runner.s

# The _main function of this file contains calls to the two external functions
# "indirect_func" and "normal_func" that are not yet defined. They are called via
# the PLT to simulate how a compiler would emit a call to an external function.
# Eventually, indirect_func will resolve to a STT_GNU_IFUNC and normal_func to a
# regular function. We include calls to both types of functions in this test to
# test that both types of functions are executed correctly when their types are
# not known initially.
# It also contains a call to a locally defined indirect function. As RuntimeDyld
# treats local functions a bit differently than external functions, we also test
# that.
# Verify that the functions return the excpeted value. If the external indirect
# function call fails, this returns the error code 1. If the external normal
# function call fails, it's the error code 2. If the call to the locally
# defined indirect function fails, return the error code 3.

local_real_func:
	mov $0x56, %eax
	ret

local_indirect_func_resolver:
	lea local_real_func(%rip), %rax
	ret

	.type local_indirect_func, @gnu_indirect_function
	.set local_indirect_func, local_indirect_func_resolver

	.global _main
_main:
	call indirect_func@plt
	cmp $0x12, %eax
	je 1f
	mov $1, %eax
	ret
1:

	call normal_func@plt
	cmp $0x34, %eax
	je 1f
	mov $2, %eax
	ret
1:

	call local_indirect_func@plt
	cmp $0x56, %eax
	je 1f
	mov $3, %eax
	ret
1:

	xor %eax, %eax
	ret

# Test that the indirect functions have the same addresses in both calls.
# rtdyld-check: decode_operand(test_indirect_func_address_1, 4) + next_pc(test_indirect_func_address_1) = decode_operand(test_indirect_func_address_2, 4) + next_pc(test_indirect_func_address_2)
test_indirect_func_address_1:
	lea indirect_func(%rip), %rax

test_indirect_func_address_2:
	lea indirect_func(%rip), %rax

# rtdyld-check: decode_operand(test_local_indirect_func_address_1, 4) + next_pc(test_indirect_func_address_1) = decode_operand(test_local_indirect_func_address_2, 4) + next_pc(test_indirect_func_address_2)
test_local_indirect_func_address_1:
	lea local_indirect_func(%rip), %rax

test_local_indirect_func_address_2:
	lea local_indirect_func(%rip), %rax

#--- func_defs.s

# This file contains the external functions that are called above. The type of
# the indirect function is set to @gnu_indirect_function and its value is set
# to the value of ifunc_resolver. This is what gcc emits when using
# __attribute__((ifunc("ifunc_resolver"))) in C. The resolver function just
# returns the address of the real function "real_func".
# To test that everyting works correctly, the indirect function returns 0x12
# and the direct function returns 0x23. This is verified in the _main function
# above.

real_func:
	mov $0x12, %eax
	ret

ifunc_resolver:
	lea real_func(%rip), %rax
	ret

	.global indirect_func
	.type indirect_func, @gnu_indirect_function
	.set indirect_func, ifunc_resolver

	.global normal_func
normal_func:
	mov $0x34, %eax
	ret

# Test that the address of the indirect function is equal even when it is
# defined in another object file.
# rtdyld-check: decode_operand(test_indirect_func_address_1, 4) + next_pc(test_indirect_func_address_1) = decode_operand(test_indirect_func_address_3, 4) + next_pc(test_indirect_func_address_3)
test_indirect_func_address_3:
	lea indirect_func(%rip), %rax
