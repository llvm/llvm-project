# RUN: llvm-mc -triple=x86_64-windows -filetype=obj < %s -o %t.obj
# RUN: llvm-objdump -d %t.obj | FileCheck %s --check-prefix=ASM
# RUN: llvm-pdbutil dump -symbols %t.obj | FileCheck %s --check-prefix=CODEVIEW

# C source to generate the assembly:
# volatile int unlikely_cond = 0;
# extern void __declspec(noreturn) abort();
# __forceinline void f() {
#   if (unlikely_cond)
#     abort();
# }
# void g() {
#   unlikely_cond = 0;
#   f();
#   unlikely_cond = 0;
# }

# This test is interesting because the inlined instructions are discontiguous.
# LLVM's block layout algorithms will put the 'abort' call last, as it is
# considered highly unlikely to execute. This is similar to what it does for
# calls to __asan_report*, for which it is very important to have an accurate
# stack trace.

# ASM:      0000000000000000 <g>:
# ASM-NEXT:        0: 48 83 ec 28                   subq    $40, %rsp
# ASM-NEXT:        4: c7 05 fc ff ff ff 00 00 00 00 movl    $0, -4(%rip)
#  Begin inline loc (matches cv_loc below)
# ASM-NEXT:        e: 83 3d ff ff ff ff 00          cmpl    $0, -1(%rip)
# ASM-NEXT:       15: 75 0f                         jne     0x26 <g+0x26>
#  End inline loc
# ASM-NEXT:       17: c7 05 fc ff ff ff 00 00 00 00 movl    $0, -4(%rip)
# ASM-NEXT:       21: 48 83 c4 28                   addq    $40, %rsp
# ASM-NEXT:       25: c3                            retq
#  Begin inline loc (matches cv_loc below)
# ASM-NEXT:       26: e8 00 00 00 00                callq   0x2b <g+0x2b>
# ASM-NEXT:       2b: 0f 0b                         ud2
#  End inline loc

# CODEVIEW:      S_INLINESITE [size = 24]
# CODEVIEW-NEXT: inlinee = 0x1002 (f), parent = 0, end = 0
# CODEVIEW-NEXT:   0B2E      code 0xE (+0xE) line 1 (+1)
# CODEVIEW-NEXT:   0409      code end 0x17 (+0x9)
# CODEVIEW-NEXT:   0B2F      code 0x26 (+0xF) line 2 (+1)
# CODEVIEW-NEXT:   0407      code end 0x2D (+0x7)

	.text
	.globl	g
g:                                      # @g
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "C:\\src\\llvm\\build\\t.cpp"
	.cv_loc	0 1 7 0 is_stmt 0       # t.cpp:7:0
.seh_proc g
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	.cv_loc	0 1 8 17                # t.cpp:8:17
	movl	$0, unlikely_cond(%rip)
	.cv_inline_site_id 1 within 0 inlined_at 1 9 3
	.cv_loc	1 1 4 7                 # t.cpp:4:7
	cmpl	$0, unlikely_cond(%rip)
	jne	.LBB0_1
	.cv_loc	0 1 10 17               # t.cpp:10:17
	movl	$0, unlikely_cond(%rip)
	.cv_loc	0 1 11 1                # t.cpp:11:1
	addq	$40, %rsp
	retq

.LBB0_1:                                # %if.then.i
	.cv_loc	1 1 5 5                 # t.cpp:5:5
	callq	abort
	ud2
.Lfunc_end0:
	.seh_handlerdata
	.text
	.seh_endproc

	.bss
	.globl	unlikely_cond           # @unlikely_cond
	.p2align	2
unlikely_cond:
	.long	0                       # 0x0

	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.long	246                     # Inlinee lines subsection
	.long	.Ltmp9-.Ltmp8           # Subsection size
.Ltmp8:
	.long	0                       # Inlinee lines signature

                                        # Inlined function f starts at t.cpp:3
	.long	4098                    # Type index of inlined function
	.long	0                       # Offset into filechecksum table
	.long	3                       # Starting line number
.Ltmp9:
	.p2align	2
	.long	241                     # Symbol subsection for g
	.long	.Ltmp11-.Ltmp10         # Subsection size
.Ltmp10:
	.short	.Ltmp13-.Ltmp12         # Record length
.Ltmp12:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	.Lfunc_end0-g           # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	4099                    # Function type index
	.secrel32	g               # Function section relative address
	.secidx	g                       # Function section index
	.byte	0                       # Flags
	.asciz	"g"                     # Function name
.Ltmp13:
	.short	.Ltmp15-.Ltmp14         # Record length
.Ltmp14:
	.short	4429                    # Record kind: S_INLINESITE
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	4098                    # Inlinee type index
	.cv_inline_linetable	1 1 3 .Lfunc_begin0 .Lfunc_end0
.Ltmp15:
	.short	2                       # Record length
	.short	4430                    # Record kind: S_INLINESITE_END
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
.Ltmp11:
	.p2align	2
	.cv_linetable	0, g, .Lfunc_end0
	.long	241                     # Symbol subsection for globals
	.long	.Ltmp17-.Ltmp16         # Subsection size
.Ltmp16:
	.short	.Ltmp19-.Ltmp18         # Record length
.Ltmp18:
	.short	4365                    # Record kind: S_GDATA32
	.long	4100                    # Type
	.secrel32	unlikely_cond   # DataOffset
	.secidx	unlikely_cond           # Segment
	.asciz	"unlikely_cond"         # Name
.Ltmp19:
.Ltmp17:
	.p2align	2
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
	#   Name: f
	# }
	.byte	0x0e, 0x00, 0x01, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x01, 0x10, 0x00, 0x00
	.byte	0x66, 0x00, 0xf2, 0xf1
	# FuncId (0x1003) {
	#   TypeLeafKind: LF_FUNC_ID (0x1601)
	#   ParentScope: 0x0
	#   FunctionType: void () (0x1001)
	#   Name: g
	# }
	.byte	0x0e, 0x00, 0x01, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x01, 0x10, 0x00, 0x00
	.byte	0x67, 0x00, 0xf2, 0xf1
	# Modifier (0x1004) {
	#   TypeLeafKind: LF_MODIFIER (0x1001)
	#   ModifiedType: int (0x74)
	#   Modifiers [ (0x2)
	#     Volatile (0x2)
	#   ]
	# }
	.byte	0x0a, 0x00, 0x01, 0x10
	.byte	0x74, 0x00, 0x00, 0x00
	.byte	0x02, 0x00, 0xf2, 0xf1

