# RUN: llvm-mc -triple=x86_64-unknown-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -phony-externals -noexec %t
#
# Check that symbol scope is demoted to local when external symbols are
# converted to absolutes. This is demotion is necessary to avoid "unexpected
# definition" errors.
#
# The reference to _GLOBAL_OFFSET_TABLE_ will trigger creation of an external
# _GLOBAL_OFFSET_TABLE_ symbol, and the GOTOFF relocation will force creation
# of a GOT symbol without actually introducing any GOT entries. Together these
# should cause the external _GLOBAL_OFFSET_TABLE_ symbol to be converted to an
# absolute symbol with address zero. If the scope is not demoted correctly this
# will trigger an "unexpected definition" error.

        .text
	.file	"ELF_external_to_absolute_conversion.s"
	.globl  main
	.p2align	4, 0x90
	.type	main,@function
main:
.L0$pb:
	leaq	.L0$pb(%rip), %rax
	movabsq	$_GLOBAL_OFFSET_TABLE_-.L0$pb, %rcx
        movabsq $_foo@GOTOFF, %rax
        xorq    %rax, %rax
        retq
