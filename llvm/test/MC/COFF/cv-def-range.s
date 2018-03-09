# RUN: llvm-mc -triple=i686-pc-win32 -filetype=obj < %s | llvm-readobj -codeview -codeview-subsection-bytes | FileCheck %s
	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 1
	.def	 _g;
	.scl	2;
	.type	32;
	.endef
	.globl	_g
	.p2align	4, 0x90
_g:                                     # @g
Lfunc_begin0:
	.cv_file	1 "\\usr\\local\\google\\home\\majnemer\\llvm\\src\\<stdin>"
	.cv_func_id 0
	.cv_loc	0 1 3 0 is_stmt 0       # <stdin>:3:0
# %bb.0:                                # %entry
	pushl	%ebp
	movl	%esp, %ebp
	subl	$8, %esp
	leal	-4(%ebp), %eax
Lvar_begin0:
	#DEBUG_VALUE: g:x <- %eax
	.cv_loc	0 1 4 7                 # <stdin>:4:7
	movl	$0, -4(%ebp)
	.cv_loc	0 1 5 3                 # <stdin>:5:3
	movl	%eax, (%esp)
	calll	_f
	.cv_loc	0 1 6 1                 # <stdin>:6:1
Lvar_end0:
	addl	$8, %esp
	popl	%ebp
	retl
Lfunc_end0:

	.section	.debug$T,"dr"
	.long	4                       # Debug section magic
	.short	6                       # Type record length
	.short	4609                    # Leaf type: LF_ARGLIST
	.long	0                       # Number of arguments
	.short	14                      # Type record length
	.short	4104                    # Leaf type: LF_PROCEDURE
	.long	3                       # Return type index
	.byte	0                       # Calling convention
	.byte	0                       # Function options
	.short	0                       # # of parameters
	.long	4096                    # Argument list type index
	.short	12                      # Type record length
	.short	5633                    # Leaf type: LF_FUNC_ID
	.long	0                       # Scope type index
	.long	4097                    # Function type
	.asciz	"g"                     # Function name
	.section	.debug$S,"dr"
	.long	4                       # Debug section magic
	.long	241                     # Symbol subsection for g
	.long	Ltmp1-Ltmp0             # Subsection size
Ltmp0:
	.short	Ltmp3-Ltmp2             # Record length
Ltmp2:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	Lfunc_end0-_g           # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	0                       # Function type index
	.secrel32	_g              # Function section relative address
	.secidx	_g                      # Function section index
	.byte	0                       # Flags
	.asciz	"g"                     # Function name
Ltmp3:
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
	.cv_def_range	Lvar_begin0 Lvar_end0, "\102\021\374\377\377\377"

# CHECK:    DefRangeFramePointerRelSym {
# CHECK:      Offset: -4
# CHECK:      LocalVariableAddrRange {
# CHECK:        OffsetStart: .text+0x9
# CHECK:        ISectStart: 0x0
# CHECK:        Range: 0xF
# CHECK:      }
# CHECK:      BlockRelocations [
# CHECK:        0x4 IMAGE_REL_I386_SECREL .text
# CHECK:        0x8 IMAGE_REL_I386_SECTION .text
# CHECK:      ]
# CHECK:    }

Ltmp1:
	.p2align	2
	.cv_linetable	0, _g, Lfunc_end0
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table

