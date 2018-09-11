# RUN: llvm-mc -triple=i686-windows-msvc %s -filetype=obj -o %t.obj
# RUN: llvm-readobj -codeview %t.obj | FileCheck %s

# The .cv_string directive mainly exists as a convenience for manually writing
# FPO data in assembler. Test that we can write FPO data using this directive,
# and that the string comes out in the dumper.

# void g(int);
# void f(int x) {
#   g(x+1);
# }

# CHECK: FrameFunc [
# CHECK-NEXT: abc =
# CHECK-NEXT: def =
# CHECK-NEXT: ghi =
# CHECK-NEXT: ]

	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 1
	.def	 _f;
	.scl	2;
	.type	32;
	.endef
	.globl	_f                      # -- Begin function f
	.p2align	4, 0x90
_f:                                     # @f
Lfunc_begin0:
# %bb.0:                                # %entry
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	movl	8(%ebp), %eax
	movl	8(%ebp), %ecx
	addl	$1, %ecx
	movl	%ecx, (%esp)
	movl	%eax, -4(%ebp)          # 4-byte Spill
	calll	_g
	addl	$8, %esp
	popl	%ebp
	retl
Lfunc_end0:
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic

	# Open coded frame data
	.long	245
	.long	Lfoo_fpo_end-Lfoo_fpo_begin           # Subsection size
Lfoo_fpo_begin:
	.long _f
	.long 0
	.long Lfunc_end0-Lfunc_begin0
	.long 24 # LocalSize
	.long 0 # ParamSize
	.long 0 # MaxStackSize
	.cv_string "abc = def = ghi = "
	.short 0 # PrologSize
	.short 0 # SavedRegSize
	.long 0x4 # Flags
Lfoo_fpo_end:
	.p2align	2
	.cv_stringtable                 # String table
