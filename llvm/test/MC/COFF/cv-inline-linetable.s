# RUN: llvm-mc -triple=i686-pc-win32 -filetype=obj %s -o %t.o
# RUN: llvm-readobj --codeview %t.o | FileCheck %s
# RUN: llvm-objdump --no-print-imm-hex -d %t.o | FileCheck %s --check-prefix=ASM
# RUN: llvm-pdbutil dump -symbols %t.o | FileCheck %s --check-prefix=PDB
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
# %bb.0:                                # %entry
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

# Check the disassembly so we have accurate instruction offsets in hex.
# ASM-LABEL: <?baz@@YAXXZ>:
# ASM-NEXT:       0: {{.*}} pushl   %eax
# ASM-NEXT:       1: {{.*}} addl    $6, 0
# ASM-NEXT:       8: {{.*}} addl    $4, 0
# ASM-NEXT:       f: {{.*}} movl    $1, (%esp)
# ASM-NEXT:      16: {{.*}} leal    (%esp), %eax
# ASM-NEXT:      19: {{.*}} addl    %eax, 0
# ASM-NEXT:      1f: {{.*}} addl    $2, 0
# ASM-NEXT:      26: {{.*}} addl    $3, 0
# ASM-NEXT:      2d: {{.*}} addl    $5, 0
# ASM-NEXT:      34: {{.*}} addl    $7, 0
# ASM-NEXT:      3b: {{.*}} popl    %eax
# ASM-NEXT:      3c: {{.*}} retl

# PDB: S_GPROC32_ID {{.*}} `baz`
# PDB: S_INLINESITE
# PDB-NEXT: inlinee = 0x1003 (bar), parent = 0, end = 0
# PDB-NEXT:   0B08      code 0x8 (+0x8) line 0 (-0)
# PDB-NEXT:   0B27      code 0xF (+0x7) line 1 (+1)
# PDB-NEXT:   0602      line 2 (+1)
# PDB-NEXT:   031E      code 0x2D (+0x1E)
# PDB-NEXT:   0407      code end 0x34 (+0x7)
# PDB: S_INLINESITE
# PDB-NEXT: inlinee = 0x1004 (foo), parent = 0, end = 0
# PDB-NEXT:    0B0F      code 0xF (+0xF) line 0 (-0)
# PDB-NEXT:    0B2A      code 0x19 (+0xA) line 1 (+1)
# PDB-NEXT:    0B26      code 0x1F (+0x6) line 2 (+1)
# PDB-NEXT:    0B27      code 0x26 (+0x7) line 3 (+1)
# PDB-NEXT:    0407      code end 0x2D (+0x7)
# PDB: S_INLINESITE_END
# PDB: S_INLINESITE_END
# PDB: S_PROC_ID_END

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
	.long 0 # parent
	.long 0 # end
	.long 0x1003 # inlinee, bar
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

# CHECK-LABEL:  FunctionLineTable [
# CHECK:    LinkageName: ?baz@@YAXXZ
# CHECK:    Flags: 0x1
# CHECK:    CodeSize: 0x3D
# CHECK:    FilenameSegment [
# CHECK:      Filename: D:\src\llvm\build\t.cpp (0x0)
# CHECK:      +0x0 [
# CHECK:        LineNumberStart: 13
# CHECK:      ]
# CHECK:      +0x1 [
# CHECK:        LineNumberStart: 14
# CHECK:      ]
# CHECK:      +0x8 [
# CHECK:        LineNumberStart: 15
# CHECK:      ]
#	There shouldn't be any other line number entries because all the other
#	.cv_locs are on line 15 where the top-level inline call site is.
# CHECK-NOT: LineNumberStart
# CHECK:      +0x34 [
# CHECK:        LineNumberStart: 16
# CHECK:      ]
# CHECK:      +0x3B [
# CHECK:        LineNumberStart: 17
# CHECK:      ]
# CHECK:    ]
# CHECK:  ]
