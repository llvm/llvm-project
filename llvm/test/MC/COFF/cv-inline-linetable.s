# RUN: llvm-mc -triple=i686-pc-win32 -filetype=obj < %s | llvm-readobj -codeview | FileCheck %s
	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 1
	.def	 "?baz@@YAXXZ";
	.scl	2;
	.type	32;
	.endef
	.globl	"?baz@@YAXXZ"
	.p2align	4, 0x90
"?baz@@YAXXZ":                          # @"\01?baz@@YAXXZ"
Lfunc_begin0:
	.cv_file	1 "D:\\src\\llvm\\build\\t.cpp"
	.cv_func_id 0
	.cv_inline_site_id 1 within 0 inlined_at 1 15 3
	.cv_inline_site_id 2 within 1 inlined_at 1 10 3
	.cv_loc	0 1 13 0 is_stmt 0      # t.cpp:13:0
# BB#0:                                 # %entry
	pushl	%eax
	.cv_loc	0 1 14 5                # t.cpp:14:5
	addl	$6, "?x@@3HC"
	.cv_loc	1 1 9 5                 # t.cpp:9:5
	addl	$4, "?x@@3HC"
	.cv_loc	2 1 3 7                 # t.cpp:3:7
	movl	$1, (%esp)
	leal	(%esp), %eax
	.cv_loc	2 1 4 5                 # t.cpp:4:5
	addl	%eax, "?x@@3HC"
	.cv_loc	2 1 5 5                 # t.cpp:5:5
	addl	$2, "?x@@3HC"
	.cv_loc	2 1 6 5                 # t.cpp:6:5
	addl	$3, "?x@@3HC"
	.cv_loc	1 1 11 5                # t.cpp:11:5
	addl	$5, "?x@@3HC"
	.cv_loc	0 1 16 5                # t.cpp:16:5
	addl	$7, "?x@@3HC"
	.cv_loc	0 1 17 1                # t.cpp:17:1
	popl	%eax
	retl
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
	.ascii	"baz"
	.byte	0
	.short	14
	.short	5633
	.asciz	"\000\000\000\000\001\020\000"
	.ascii	"bar"
	.byte	0
	.short	14
	.short	5633
	.asciz	"\000\000\000\000\001\020\000"
	.ascii	"foo"
	.byte	0
	.section	.debug$S,"dr"
	.long	4
	.long	241                     # Symbol subsection for baz
	.long	Ltmp1-Ltmp0
Ltmp0:
	.short	Ltmp3-Ltmp2
Ltmp2:
	.short	4423
	.zero	12
	.long	Lfunc_end0-"?baz@@YAXXZ"
	.zero	12
	.secrel32	"?baz@@YAXXZ"
	.secidx	"?baz@@YAXXZ"
	.byte	0
	.ascii	"baz"
	.byte	0
Ltmp3:
	.short	Ltmp5-Ltmp4
Ltmp4:
	.short	4429
	.asciz	"\000\000\000\000\000\000\000\000\003\020\000"
	.cv_inline_linetable	1 1 9 Lfunc_begin0 Lfunc_end0
# CHECK:    InlineSiteSym {
# CHECK:      PtrParent: 0x0
# CHECK:      PtrEnd: 0x0
# CHECK:      Inlinee: bar (0x1003)
# CHECK:      BinaryAnnotations [
# CHECK-NEXT:   ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x8, LineOffset: 0}
# CHECK-NEXT:   ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x7, LineOffset: 1}
# CHECK-NEXT:   ChangeLineOffset: 1
# CHECK-NEXT:   ChangeCodeOffset: 0x1E
# CHECK-NEXT:   ChangeCodeLength: 0x7
# CHECK-NEXT: ]
# CHECK:    }
Ltmp5:
	.short	Ltmp7-Ltmp6
Ltmp6:
	.short	4429
	.asciz	"\000\000\000\000\000\000\000\000\004\020\000"
	.cv_inline_linetable	2 1 3 Lfunc_begin0 Lfunc_end0
# CHECK:    InlineSiteSym {
# CHECK:      PtrParent: 0x0
# CHECK:      PtrEnd: 0x0
# CHECK:      Inlinee: foo (0x1004)
# CHECK:      BinaryAnnotations [
# CHECK-NEXT:   ChangeCodeOffsetAndLineOffset: {CodeOffset: 0xF, LineOffset: 0}
# CHECK-NEXT:   ChangeCodeOffsetAndLineOffset: {CodeOffset: 0xA, LineOffset: 1}
# CHECK-NEXT:   ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x6, LineOffset: 1}
# CHECK-NEXT:   ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x7, LineOffset: 1}
# CHECK-NEXT:   ChangeCodeLength: 0x7
# CHECK-NEXT: ]
# CHECK:    }
Ltmp7:
	.short	2
	.short	4430
# CHECK:    InlineSiteEnd {
# CHECK:    }
	.short	2
	.short	4430
# CHECK:    InlineSiteEnd {
# CHECK:    }
	.short	2
	.short	4431
Ltmp1:
	.p2align 2
	.cv_linetable	0, "?baz@@YAXXZ", Lfunc_end0
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table

