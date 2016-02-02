# RUN: llvm-mc -triple=i686-pc-win32 -filetype=obj < %s | llvm-readobj -codeview | FileCheck %s
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
	.cv_loc	0 1 7 0 is_stmt 0       # <stdin>:7:0
# BB#0:                                 # %entry
	pushl	%ebp
	movl	%esp, %ebp
	.cv_loc	1 1 4 3                 # <stdin>:4:3
	movl	_x, %eax
	addl	$1, %eax
	movl	%eax, _x
Lfunc_end0:

	.comm	_x,4,2                  # @x
	.section	.debug$T,"dr"
	.long	4
	.short	6
	.short	4609
	.long	0
	.short	14
	.short	4104
	.asciz	"\003\000\000\000\000\000\000\000\000\020\000"
	.short	12
	.short	5633
	.asciz	"\000\000\000\000\001\020\000"
	.byte	103
	.byte	0
	.short	12
	.short	5633
	.asciz	"\000\000\000\000\001\020\000"
	.byte	102
	.byte	0
	.section	.debug$S,"dr"
	.long	4
	.long	246                     # Inlinee lines subsection
	.long	Ltmp1-Ltmp0
Ltmp0:
	.long	0
	.long	4099                    # Inlined function f starts at <stdin>:3
	.long	0
	.long	3
Ltmp1:
	.long	241                     # Symbol subsection for g
	.long	Ltmp3-Ltmp2
Ltmp2:
	.short	Ltmp5-Ltmp4
Ltmp4:
	.short	4423
	.zero	12
	.long	Lfunc_end0-_g
	.zero	12
	.secrel32	_g
	.secidx	_g
	.byte	0
	.byte	103
	.byte	0
Ltmp5:
	.short	Ltmp7-Ltmp6
Ltmp6:
	.short	4429
	.asciz	"\000\000\000\000\000\000\000\000\003\020\000"
	.cv_inline_linetable	1 1 3 Lfunc_begin0 Lfunc_end0
# CHECK:    InlineSite {
# CHECK:      PtrParent: 0x0
# CHECK:      PtrEnd: 0x0
# CHECK:      Inlinee: f (0x1003)
# CHECK:      BinaryAnnotations [
# CHECK:        ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x3, LineOffset: 1}
# CHECK:        ChangeCodeLength: 0xD
# CHECK:      ]
# CHECK:    }
Ltmp7:
	.short	2
	.short	4430
# CHECK:    InlineSiteEnd {
# CHECK:    }
	.short	2
	.short	4431
Ltmp3:
	.p2align	2
	.cv_linetable	0, _g, Lfunc_end0
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table
