# RUN: llvm-mc -triple=x86_64-pc-win32 -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s

# CHECK:    InlineSiteSym {
# CHECK:      BinaryAnnotations [
# CHECK:        ChangeLineOffset: 1
# CHECK:        ChangeCodeLength: 0x2
# CHECK:      ]
# CHECK:    }

	.text
	.cv_file	1 "D:\\src\\llvm\\build\\t.c"

	.def	 infloop;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,infloop
	.globl	infloop
	.p2align	4, 0x90
infloop:                                    # @infloop
.Lfunc_begin1:
	.cv_func_id 0
	.cv_inline_site_id 2 within 0 inlined_at 1 1 1
	.cv_loc	2 1 3 7                 # t.c:3:7
	jmp	.Lfunc_begin1
.Lfunc_end1:

	.def	 afterinfloop;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,afterinfloop
	.globl	afterinfloop
	.p2align	4, 0x90
afterinfloop:                                    # @afterinfloop
	.cv_func_id 3
	.cv_loc	3 1 13 0                # t.c:13:0
	retq

	.section	.debug$S,"dr"
	.long 4
	.long	241                     # Symbol subsection for infloop
	.long	.Ltmp17-.Ltmp16         # Subsection size
.Ltmp16:
	.short	.Ltmp19-.Ltmp18         # Record length
.Ltmp18:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	.Lfunc_end1-infloop         # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	0                       # Function type index
	.secrel32	infloop             # Function section relative address
	.secidx	infloop                     # Function section index
	.byte	0                       # Flags
	.asciz	"infloop"                   # Function name
.Ltmp19:
	.short	.Ltmp21-.Ltmp20         # Record length
.Ltmp20:
	.short	4429                    # Record kind: S_INLINESITE
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	4098                    # Inlinee type index
	.cv_inline_linetable	2 1 2 .Lfunc_begin1 .Lfunc_end1
.Ltmp21:
	.short	2                       # Record length
	.short	4430                    # Record kind: S_INLINESITE_END
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
.Ltmp17:
	.p2align	2
	.cv_linetable	1, infloop, .Lfunc_end1
