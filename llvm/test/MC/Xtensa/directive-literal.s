# RUN: llvm-mc -triple=xtensa -filetype obj -o - %s \
# RUN:   | llvm-readobj -S --sd - \
# RUN:   | FileCheck -check-prefix=CHECK-LITERAL %s

# RUN: llvm-mc %s -triple=xtensa  -show-encoding \
# RUN:   | FileCheck -check-prefix=CHECK-INST %s

	.text
	.literal_position
	.literal .LCPI0_0, 305419896
	.literal .LCPI1_0, ext_var
	.global	test_literal
	.p2align	2
	.type	test_literal,@function
test_literal:
	l32r	a2, .LCPI0_0
	l32r	a3, .LCPI1_0
	movi    a4, 30000
	movi    a5, 1000
	ret

# CHECK-LITERAL: Section {
# CHECK-LITERAL:   Name: .literal
# CHECK-LITERAL:   SectionData (
# CHECK-LITERAL:     0000: 78563412 00000000 30750000
# CHECK-LITERAL:   )
# CHECK-LITERAL: }

# CHECK-INST: .literal_position
# CHECK-INST: .literal .LCPI0_0, 305419896
# CHECK-INST: .literal .LCPI1_0, ext_var
# CHECK-INST: .global test_literal
# CHECK-INST: .p2align 2
# CHECK-INST: .type test_literal,@function
# CHECK-INST: test_literal:
# CHECK-INST: l32r a2, .LCPI0_0
# CHECK-INST: l32r a3, .LCPI1_0
# CHECK-INST: .literal .Ltmp0, 30000
# CHECK-INST: l32r a4, .Ltmp0
# CHECK-INST: movi a5, 1000
# CHECK-INST: ret

