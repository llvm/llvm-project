# RUN: llvm-mc -triple x86_64-windows-msvc %s -filetype=obj -o %t.o
# RUN: llvm-pdbutil dump -symbols %t.o | FileCheck %s

# We used to have a label flushing bug down below by the "BUG" comments that
# would cause the S_DEFRANGE_FRAMEPOINTER_REL records to appear missing. In
# practice, the label would extend past the def range, so it would appear that
# every local was optimized out or had no def ranges.

# CHECK: S_GPROC32_ID {{.*}} `max`
# CHECK: S_LOCAL [size = {{.*}}] `a`
# CHECK: S_DEFRANGE_FRAMEPOINTER_REL
# CHECK: S_LOCAL [size = {{.*}}] `b`
# CHECK: S_DEFRANGE_FRAMEPOINTER_REL

	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.def	 max;
	.scl	2;
	.type	32;
	.endef
	.globl	max                     # -- Begin function max
	.p2align	4, 0x90
max:                                    # @max
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "C:\\src\\llvm-project\\build\\t.c" "44649E6EBC4FC8880991A1AF1F2D2990" 1
	.cv_loc	0 1 1 0                 # t.c:1:0
.seh_proc max
# %bb.0:                                # %entry
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movl	%edx, 4(%rsp)
	movl	%ecx, (%rsp)
.Ltmp0:
	.cv_loc	0 1 2 0                 # t.c:2:0
	movl	(%rsp), %eax
	cmpl	4(%rsp), %eax
	jle	.LBB0_2
# %bb.1:                                # %cond.true
	movl	(%rsp), %eax
	jmp	.LBB0_3
.LBB0_2:                                # %cond.false
	movl	4(%rsp), %eax
.LBB0_3:                                # %cond.end
	popq	%rcx
	retq
.Ltmp1:
.Lfunc_end0:
	.seh_handlerdata
	.text
	.seh_endproc
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2
	.long 4
	.long	241                     # Symbol subsection for max
	.long	.Ltmp7-.Ltmp6           # Subsection size
.Ltmp6:
	.short	.Ltmp9-.Ltmp8           # Record length
.Ltmp8:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	.Lfunc_end0-max         # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	4098                    # Function type index
	.secrel32	max             # Function section relative address
	.secidx	max                     # Function section index
	.byte	0                       # Flags
	.asciz	"max"                   # Function name
.Ltmp9:
	.short	.Ltmp11-.Ltmp10         # Record length
.Ltmp10:
	.short	4114                    # Record kind: S_FRAMEPROC
	.long	8                       # FrameSize
	.long	0                       # Padding
	.long	0                       # Offset of padding
	.long	0                       # Bytes of callee saved registers
	.long	0                       # Exception handler offset
	.short	0                       # Exception handler section
	.long	81920                   # Flags (defines frame register)
.Ltmp11:
	.short	.Ltmp13-.Ltmp12         # Record length
.Ltmp12:
	.short	4414                    # Record kind: S_LOCAL
	.long	18                      # TypeIndex
	.short	1                       # Flags
	.asciz	"a"
	# BUG
	.p2align 2
.Ltmp13:
	.cv_def_range	 .Ltmp0 .Ltmp1, "B\021\000\000\000\000"
	.short	.Ltmp15-.Ltmp14         # Record length
.Ltmp14:
	.short	4414                    # Record kind: S_LOCAL
	.long	18                      # TypeIndex
	.short	1                       # Flags
	.asciz	"b"
	# BUG
	.p2align 2
.Ltmp15:
	.cv_def_range	 .Ltmp0 .Ltmp1, "B\021\004\000\000\000"
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
.Ltmp7:
	.p2align	2
	.cv_linetable	0, max, .Lfunc_end0
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table
	.long	241
	.long	.Ltmp17-.Ltmp16         # Subsection size
.Ltmp16:
.Ltmp17:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	# ArgList (0x1000) {
	#   TypeLeafKind: LF_ARGLIST (0x1201)
	#   NumArgs: 2
	#   Arguments [
	#     ArgType: long (0x12)
	#     ArgType: long (0x12)
	#   ]
	# }
	.byte	0x0e, 0x00, 0x01, 0x12
	.byte	0x02, 0x00, 0x00, 0x00
	.byte	0x12, 0x00, 0x00, 0x00
	.byte	0x12, 0x00, 0x00, 0x00
	# Procedure (0x1001) {
	#   TypeLeafKind: LF_PROCEDURE (0x1008)
	#   ReturnType: long (0x12)
	#   CallingConvention: NearC (0x0)
	#   FunctionOptions [ (0x0)
	#   ]
	#   NumParameters: 2
	#   ArgListType: (long, long) (0x1000)
	# }
	.byte	0x0e, 0x00, 0x08, 0x10
	.byte	0x12, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0x02, 0x00
	.byte	0x00, 0x10, 0x00, 0x00
	# FuncId (0x1002) {
	#   TypeLeafKind: LF_FUNC_ID (0x1601)
	#   ParentScope: 0x0
	#   FunctionType: long (long, long) (0x1001)
	#   Name: max
	# }
	.byte	0x0e, 0x00, 0x01, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x01, 0x10, 0x00, 0x00
	.byte	0x6d, 0x61, 0x78, 0x00

