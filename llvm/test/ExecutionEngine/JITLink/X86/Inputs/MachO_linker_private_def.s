# Supplies a linker private definition, "l_foo".

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14
	.p2align	4, 0x90
l_foo:
	xorl	%eax, %eax
	retq

.subsections_via_symbols
