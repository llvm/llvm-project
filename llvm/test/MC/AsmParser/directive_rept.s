# RUN: llvm-mc -triple i686-elf -filetype asm -o - %s | FileCheck %s

	.data

	.global two_bad_calls
	.type two_bad_calls,@function
two_bad_calls:
	.rept 2
	.long 0xbadca11
	.endr

# CHECK-LABEL: two_bad_calls
# CHECK: .long	195938833
# CHECK: .long	195938833

	.global half_a_dozen_daffodils
	.type half_a_dozen_daffodils,@function
half_a_dozen_daffodils:
	.rep 6
	.long 0xdaff0d11
	.endr

# CHECK-LABEL: half_a_dozen_daffodils
# CHECK: .long	3674148113
# CHECK: .long	3674148113
# CHECK: .long	3674148113
# CHECK: .long	3674148113
# CHECK: .long	3674148113
# CHECK: .long	3674148113

