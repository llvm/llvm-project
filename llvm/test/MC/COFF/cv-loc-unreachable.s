# RUN: llvm-mc < %s -triple=i686-pc-win32 -filetype=obj | llvm-readobj - --codeview | FileCheck %s

# Original source, slightly modified with an extra .cv_loc directive (at EXTRA
# below) that was causing assertions:
#
# void __declspec(noreturn) __declspec(dllimport) exit(int);
# int unlikely();
# static inline void do_exit() {
#   if (unlikely()) {
#     exit(32);
#   }
# }
# void callit() {
#   do_exit();
# }

# CHECK-LABEL: InlineeSourceLine {
# CHECK:   Inlinee: do_exit (0x1002)
# CHECK:   FileID: C:\src\llvm-project\build\t.cpp (0x0)
# CHECK:   SourceLineNum: 3
# CHECK: }

# CHECK-LABEL: InlineSiteSym {
# CHECK:   Kind: S_INLINESITE (0x114D)
# CHECK:   Inlinee: do_exit (0x1002)
# CHECK:   BinaryAnnotations [
# CHECK-NEXT:     ChangeLineOffset: 1
# CHECK-NEXT:     ChangeCodeLength: 0x9
# CHECK-NEXT:     ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x1, LineOffset: 1}
# CHECK-NEXT:     ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x8, LineOffset: 1}
# CHECK-NEXT:     ChangeCodeLength: 0x0
# CHECK-NEXT:   ]
# CHECK: }

	.text
	.def	 _callit; .scl	2; .type	32; .endef
	.globl	_callit                 # -- Begin function callit
_callit:                                # @callit
Lfunc_begin0:
	.cv_func_id 0
	.cv_fpo_proc	_callit 0
# %bb.0:                                # %entry
	.cv_file	1 "C:\\src\\llvm-project\\build\\t.cpp" "0BC092F354CE14FDC2FA78F8EDE7426E" 1
	.cv_inline_site_id 1 within 0 inlined_at 1 9 0
	.cv_loc	1 1 4 0 is_stmt 0       # t.cpp:4:0
	calll	_unlikely
	testl	%eax, %eax
	jne	LBB0_1
Ltmp0:
# %bb.2:                                # %do_exit.exit
	.cv_loc	0 1 10 0                # t.cpp:10:0
	retl
LBB0_1:                                 # %if.then.i
Ltmp1:
	.cv_loc	1 1 5 0                 # t.cpp:5:0
	pushl	$32
	calll	*__imp__exit
	# EXTRA
	.cv_loc	1 1 6 0                 # t.cpp:6:0
Ltmp2:
	.cv_fpo_endproc
Lfunc_end0:
                                        # -- End function

	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.long	241
	.long	Ltmp4-Ltmp3             # Subsection size
Ltmp3:
	.short	Ltmp6-Ltmp5             # Record length
Ltmp5:
	.short	4412                    # Record kind: S_COMPILE3
	.long	0                       # Flags and language
	.short	7                       # CPUType
	.short	7                       # Frontend version
	.short	0
	.short	0
	.short	0
	.short	7000                    # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 7.0.0 "  # Null-terminated compiler version string
Ltmp6:
Ltmp4:
	.p2align	2
	.long	246                     # Inlinee lines subsection
	.long	Ltmp8-Ltmp7             # Subsection size
Ltmp7:
	.long	0                       # Inlinee lines signature

                                        # Inlined function do_exit starts at t.cpp:3
	.long	4098                    # Type index of inlined function
	.cv_filechecksumoffset	1       # Offset into filechecksum table
	.long	3                       # Starting line number
Ltmp8:
	.p2align	2
	.cv_fpo_data	_callit
	.long	241                     # Symbol subsection for callit
	.long	Ltmp10-Ltmp9            # Subsection size
Ltmp9:
	.short	Ltmp12-Ltmp11           # Record length
Ltmp11:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	Lfunc_end0-_callit      # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	4099                    # Function type index
	.secrel32	_callit         # Function section relative address
	.secidx	_callit                 # Function section index
	.byte	0                       # Flags
	.asciz	"callit"                # Function name
Ltmp12:
	.short	Ltmp14-Ltmp13           # Record length
Ltmp13:
	.short	4429                    # Record kind: S_INLINESITE
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	4098                    # Inlinee type index
	.cv_inline_linetable	1 1 3 Lfunc_begin0 Lfunc_end0
Ltmp14:
	.short	2                       # Record length
	.short	4430                    # Record kind: S_INLINESITE_END
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
Ltmp10:
	.p2align	2
	.cv_linetable	0, _callit, Lfunc_end0
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	# ArgList (0x1000) {
	#   TypeLeafKind: LF_ARGLIST (0x1201)
	#   NumArgs: 0
	#   Arguments [
	#   ]
	# }
	.byte	0x06, 0x00, 0x01, 0x12
	.byte	0x00, 0x00, 0x00, 0x00
	# Procedure (0x1001) {
	#   TypeLeafKind: LF_PROCEDURE (0x1008)
	#   ReturnType: void (0x3)
	#   CallingConvention: NearC (0x0)
	#   FunctionOptions [ (0x0)
	#   ]
	#   NumParameters: 0
	#   ArgListType: () (0x1000)
	# }
	.byte	0x0e, 0x00, 0x08, 0x10
	.byte	0x03, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x00, 0x10, 0x00, 0x00
	# FuncId (0x1002) {
	#   TypeLeafKind: LF_FUNC_ID (0x1601)
	#   ParentScope: 0x0
	#   FunctionType: void () (0x1001)
	#   Name: do_exit
	# }
	.byte	0x12, 0x00, 0x01, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x01, 0x10, 0x00, 0x00
	.byte	0x64, 0x6f, 0x5f, 0x65
	.byte	0x78, 0x69, 0x74, 0x00
	# FuncId (0x1003) {
	#   TypeLeafKind: LF_FUNC_ID (0x1601)
	#   ParentScope: 0x0
	#   FunctionType: void () (0x1001)
	#   Name: callit
	# }
	.byte	0x12, 0x00, 0x01, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x01, 0x10, 0x00, 0x00
	.byte	0x63, 0x61, 0x6c, 0x6c
	.byte	0x69, 0x74, 0x00, 0xf1

