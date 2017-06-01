# RUN: llvm-mc -filetype=obj -triple i686-pc-win32 < %s | llvm-readobj -codeview - | FileCheck %s
	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 1
	.def	 _f;
	.scl	2;
	.type	32;
	.endef
	.globl	_f
	.p2align	4, 0x90
_f:                                     # @f
Lfunc_begin0:
# BB#0:                                 # %entry
	.cv_file	1 "cv-empty-linetable.s"
	.cv_func_id 1
	.cv_loc	1 1 3 15 is_stmt 0
	jmp	_g                      # TAILCALL
Lfunc_end0:

	.section	.debug$T,"dr"
	.long	4
	.short	6
	.short	4609
	.long	0
	.short	14
	.short	4104
	.asciz	"\003\000\000\000\000\000\000\000\000\020\000"
	.short	14
	.short	5633
	.asciz	"\000\000\000\000\001\020\000"
	.ascii	"fn1"
	.byte	0
	.short	38
	.short	5633
	.asciz	"\000\000\000\000\001\020\000"
	.ascii	"??__Fa@?1??fn1@@YAXXZ@YAXXZ"
	.byte	0
	.short	26
	.short	5633
	.asciz	"\000\000\000\000\001\020\000"
	.ascii	"vector::~vector"
	.byte	0
	.section	.debug$S,"dr"
	.long	4
	.long	241                     # Symbol subsection for f
	.long	Ltmp1-Ltmp0
Ltmp0:
	.short	Ltmp3-Ltmp2
Ltmp2:
	.short	4423
	.zero	12
	.long	Lfunc_end0-_f
	.zero	12
	.secrel32	_f
	.secidx	_f
	.byte	0
	.byte	102
	.byte	0
Ltmp3:
	.short	Ltmp5-Ltmp4
Ltmp4:
	.short	4429
	.asciz	"\000\000\000\000\000\000\000\000\004\020\000"
Ltmp5:
	.short	2
	.short	4430
	.short	2
	.short	4431
Ltmp1:
	.zero	3
	.cv_linetable	0, _f, Lfunc_end0
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table

# CHECK:  FunctionLineTable [
# CHECK:    LinkageName: _f
# CHECK:    Flags: 0x0
# CHECK:    CodeSize: 0x5
# CHECK:  ]
