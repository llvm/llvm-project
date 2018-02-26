# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin10.0 %s | llvm-objdump -unwind-info - | FileCheck %s

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 10

# Check that we emit compact-unwind info with UNWIND_X86_MODE_STACK_IND encoding

# CHECK: Contents of __compact_unwind section:
# CHECK-NEXT:   Entry at offset 0x0:
# CHECK-NEXT:     start:                0x0 _test0
# CHECK-NEXT:     length:               0x15
# CHECK-NEXT:     compact encoding:     0x03056804
	.globl	_test0
_test0:                                  ## @test0
	.cfi_startproc
## %bb.0:                               ## %entry
	pushq	%rbp
Ltmp0:
	.cfi_def_cfa_offset 16
	pushq	%rbx
Ltmp1:
	.cfi_def_cfa_offset 24
	subq	$14408, %rsp            ## imm = 0x3848
Ltmp2:
	.cfi_def_cfa_offset 14432
Ltmp3:
	.cfi_offset %rbx, -24
Ltmp4:
	.cfi_offset %rbp, -16
	xorl	%eax, %eax
	addq	$14408, %rsp            ## imm = 0x3848
	popq	%rbx
	popq	%rbp
	retq
	.cfi_endproc

# Check that we emit compact-unwind info with UNWIND_X86_MODE_STACK_IMMD encoding

# CHECK:   Entry at offset 0x20:
# CHECK-NEXT:     start:                0x15 _test1
# CHECK-NEXT:     length:               0x15
# CHECK-NEXT:     compact encoding:     0x02360804
	.globl	_test1
_test1:                                  ## @test1
	.cfi_startproc
## %bb.0:                               ## %entry
	pushq	%rbp
Ltmp10:
	.cfi_def_cfa_offset 16
	pushq	%rbx
Ltmp11:
	.cfi_def_cfa_offset 24
	subq	$408, %rsp              ## imm = 0x198
Ltmp12:
	.cfi_def_cfa_offset 432
Ltmp13:
	.cfi_offset %rbx, -24
Ltmp14:
	.cfi_offset %rbp, -16
	xorl	%eax, %eax
	addq	$408, %rsp              ## imm = 0x198
	popq	%rbx
	popq	%rbp
	retq
	.cfi_endproc

	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"%d\n"


.subsections_via_symbols
