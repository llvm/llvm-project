# RUN: llvm-mc -triple=i686-windows-msvc -filetype=obj < %s | llvm-readobj -codeview | FileCheck %s

# Test for .cv_fpo_stackalign. We should generate FPO data that restores CSRs
# at each instruction, and in the last FrameData we should use the '@'
# alignment operator to define $T0, the vframe value.

# Based on this C code:
# void usevals(int, int, double*);
# int realign_with_csrs() {
#   int a = getval();
#   int b = getval();
#   double __declspec(align(8)) force_alignment = 0.42;
#   usevals(a, b, &force_alignment);
#   return a + b;
# }

# CHECK: Subsection [
# CHECK:   SubSectionType: Symbols (0xF1)
# CHECK:   Compile3Sym {
# CHECK:     Kind: S_COMPILE3 (0x113C)
# CHECK:   }
# CHECK: ]
# CHECK: Subsection [
# CHECK:   SubSectionType: FrameData (0xF5)
# CHECK:   FrameData {
# CHECK:     FrameFunc [
# CHECK:       $T0 .raSearch =
# CHECK:       $eip $T0 ^ =
# CHECK:       $esp $T0 4 + =
# CHECK:     ]
# CHECK:   }
# CHECK:   FrameData {
# CHECK:     FrameFunc [
# CHECK:       $T0 .raSearch =
# CHECK:       $eip $T0 ^ =
# CHECK:       $esp $T0 4 + =
# CHECK:       $ebp $T0 4 - ^ =
# CHECK:     ]
# CHECK:   }
# CHECK:   FrameData {
# CHECK:     FrameFunc [
# CHECK:       $T0 $ebp 4 + =
# CHECK:       $eip $T0 ^ =
# CHECK:       $esp $T0 4 + =
# CHECK:       $ebp $T0 4 - ^ =
# CHECK:     ]
# CHECK:   }
# CHECK:   FrameData {
# CHECK:     FrameFunc [
# CHECK:       $T0 $ebp 4 + =
# CHECK:       $eip $T0 ^ =
# CHECK:       $esp $T0 4 + =
# CHECK:       $ebp $T0 4 - ^ =
# CHECK:       $edi $T0 8 - ^ =
# CHECK:     ]
# CHECK:   }
# CHECK:   FrameData {
# CHECK:     FrameFunc [
# CHECK:       $T0 $ebp 4 + =
# CHECK:       $eip $T0 ^ =
# CHECK:       $esp $T0 4 + =
# CHECK:       $ebp $T0 4 - ^ =
# CHECK:       $edi $T0 8 - ^ =
# CHECK:       $esi $T0 12 - ^ =
# CHECK:     ]
# CHECK:   }
# CHECK:   FrameData {
# CHECK:     FrameFunc [
# CHECK:       $T1 $ebp 4 + =
# CHECK:       $T0 $T1 12 - 8 @ =
# CHECK:       $eip $T1 ^ =
# CHECK:       $esp $T1 4 + =
# CHECK:       $ebp $T1 4 - ^ =
# CHECK:       $edi $T1 8 - ^ =
# CHECK:       $esi $T1 12 - ^ =
# CHECK:     ]
# CHECK:   }
# CHECK: ]
# CHECK: Subsection [
# CHECK:   SubSectionType: Symbols (0xF1)
# CHECK: ]
# CHECK: Subsection [
# CHECK:   SubSectionType: FileChecksums (0xF4)
# CHECK: ]
# CHECK: Subsection [
# CHECK:   SubSectionType: StringTable (0xF3)
# CHECK: ]

	.text
	.def	 _realign_with_csrs; .scl	2; .type	32; .endef
	.globl	_realign_with_csrs      # -- Begin function realign_with_csrs
_realign_with_csrs:                     # @realign_with_csrs
Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "C:\\src\\llvm-project\\build\\t.c" "2A4F9B6BBBF7845521201755D1B14ACC" 1
	.cv_loc	0 1 4 0                 # t.c:4:0
	.cv_fpo_proc	_realign_with_csrs 0
# %bb.0:                                # %entry
	pushl	%ebp
	.cv_fpo_pushreg	%ebp
	movl	%esp, %ebp
	.cv_fpo_setframe	%ebp
Ltmp0:
	pushl	%edi
	.cv_fpo_pushreg	%edi
	pushl	%esi
	.cv_fpo_pushreg	%esi
	andl	$-8, %esp
	.cv_fpo_stackalign	8
	subl	$8, %esp
	.cv_fpo_stackalloc	8
	.cv_fpo_endprologue
	.cv_loc	0 1 5 0                 # t.c:5:0
	calll	_getval
	movl	%eax, %esi
	.cv_loc	0 1 6 0                 # t.c:6:0
	calll	_getval
	movl	%eax, %edi
	movl	%esp, %eax
	.cv_loc	0 1 7 0                 # t.c:7:0
	movl	$1071309127, 4(%esp)    # imm = 0x3FDAE147
	movl	$-1374389535, (%esp)    # imm = 0xAE147AE1
	.cv_loc	0 1 8 0                 # t.c:8:0
	pushl	%eax
	pushl	%edi
	pushl	%esi
	calll	_usevals
	addl	$12, %esp
	.cv_loc	0 1 9 0                 # t.c:9:0
	addl	%esi, %edi
	movl	%edi, %eax
	leal	-8(%ebp), %esp
	popl	%esi
	popl	%edi
	popl	%ebp
	retl
Ltmp1:
	.cv_fpo_endproc
Lfunc_end0:
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.long	241
	.long	Ltmp3-Ltmp2             # Subsection size
Ltmp2:
	.short	Ltmp5-Ltmp4             # Record length
Ltmp4:
	.short	4412                    # Record kind: S_COMPILE3
	.long	0                       # Flags and language
	.short	7                       # CPUType
	.short	8                       # Frontend version
	.short	0
	.short	0
	.short	0
	.short	8000                    # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 8.0.0 "  # Null-terminated compiler version string
Ltmp5:
Ltmp3:
	.p2align 2
	.cv_fpo_data	_realign_with_csrs
	.long	241                     # Symbol subsection for realign_with_csrs
	.long	Ltmp7-Ltmp6             # Subsection size
Ltmp6:
	.short	Ltmp9-Ltmp8             # Record length
Ltmp8:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	Lfunc_end0-_realign_with_csrs # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	0                    # Function type index
	.secrel32	_realign_with_csrs # Function section relative address
	.secidx	_realign_with_csrs      # Function section index
	.byte	0                       # Flags
	.asciz	"realign_with_csrs"     # Function name
Ltmp9:
	.short	Ltmp11-Ltmp10           # Record length
Ltmp10:
	.short	4114                    # Record kind: S_FRAMEPROC
	.long	12                      # FrameSize
	.long	0                       # Padding
	.long	0                       # Offset of padding
	.long	8                       # Bytes of callee saved registers
	.long	0                       # Exception handler offset
	.short	0                       # Exception handler section
	.long	1196032                 # Flags (defines frame register)
Ltmp11:
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
Ltmp7:
	.p2align	2
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table
