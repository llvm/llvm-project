// clang-format off
// REQUIRES: lld, x86

// Test block range is set.
// RUN: llvm-mc -triple=x86_64-windows-msvc --filetype=obj %s > %t.obj
// RUN: lld-link /debug:full /nodefaultlib /entry:main %t.obj /out:%t.exe /base:0x140000000
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb %t.exe -o "image lookup -a 0x140001014 -v" -o "exit" | FileCheck %s

// CHECK:      Function: id = {{.*}}, name = "main", range = [0x0000000140001000-0x0000000140001044)
// CHECK-NEXT: FuncType: id = {{.*}}, byte-size = 0, compiler_type = "int (void)"
// CHECK-NEXT:   Blocks: id = {{.*}}, range = [0x140001000-0x140001044)
// CHECK-NEXT:           id = {{.*}}, range = [0x140001014-0x14000103b)


	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.intel_syntax noprefix
	.file	"blocks.cpp"
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "/tmp/blocks.cpp" "4CC0785E17ACF09C657F740775661395" 1
	.cv_loc	0 1 1 0                         # blocks.cpp:1:0
.seh_proc main
# %bb.0:                                # %entry
	sub	rsp, 16
	.seh_stackalloc 16
	.seh_endprologue
	mov	dword ptr [rsp + 12], 0
.Ltmp0:
	.cv_loc	0 1 2 0                         # blocks.cpp:2:0
	mov	dword ptr [rsp + 8], 0
.Ltmp1:
	.cv_loc	0 1 3 0                         # blocks.cpp:3:0
	mov	dword ptr [rsp + 4], 0
.LBB0_1:                                # %for.cond
                                        # =>This Inner Loop Header: Depth=1
	cmp	dword ptr [rsp + 4], 3
	jge	.LBB0_4
# %bb.2:                                # %for.body
                                        #   in Loop: Header=BB0_1 Depth=1
.Ltmp2:
	.cv_loc	0 1 4 0                         # blocks.cpp:4:0
	mov	eax, dword ptr [rsp + 8]
	add	eax, 1
	mov	dword ptr [rsp + 8], eax
.Ltmp3:
# %bb.3:                                # %for.inc
                                        #   in Loop: Header=BB0_1 Depth=1
	.cv_loc	0 1 3 0                         # blocks.cpp:3:0
	mov	eax, dword ptr [rsp + 4]
	add	eax, 1
	mov	dword ptr [rsp + 4], eax
	jmp	.LBB0_1
.Ltmp4:
.LBB0_4:                                # %for.end
	.cv_loc	0 1 6 0                         # blocks.cpp:6:0
	mov	eax, dword ptr [rsp + 8]
	add	rsp, 16
	ret
.Ltmp5:
.Lfunc_end0:
	.seh_endproc
                                        # -- End function
	.section	.drectve,"yn"
	.ascii	" /DEFAULTLIB:libcmt.lib"
	.ascii	" /DEFAULTLIB:oldnames.lib"
	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp7-.Ltmp6                   # Subsection size
.Ltmp6:
	.short	.Ltmp9-.Ltmp8                   # Record length
.Ltmp8:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.asciz	"/tmp/blocks.obj"               # Object name
	.p2align	2, 0x0
.Ltmp9:
	.short	.Ltmp11-.Ltmp10                 # Record length
.Ltmp10:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	208                             # CPUType
	.short	16                              # Frontend version
	.short	0
	.short	0
	.short	0
	.short	16000                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 16.0.0"          # Null-terminated compiler version string
	.p2align	2, 0x0
.Ltmp11:
.Ltmp7:
	.p2align	2, 0x0
	.long	241                             # Symbol subsection for main
	.long	.Ltmp13-.Ltmp12                 # Subsection size
.Ltmp12:
	.short	.Ltmp15-.Ltmp14                 # Record length
.Ltmp14:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end0-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4098                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2, 0x0
.Ltmp15:
	.short	.Ltmp17-.Ltmp16                 # Record length
.Ltmp16:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	16                              # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	81920                           # Flags (defines frame register)
	.p2align	2, 0x0
.Ltmp17:
	.short	.Ltmp19-.Ltmp18                 # Record length
.Ltmp18:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	0                               # Flags
	.asciz	"count"
	.p2align	2, 0x0
.Ltmp19:
	.cv_def_range	 .Ltmp0 .Ltmp5, frame_ptr_rel, 8
	.short	.Ltmp21-.Ltmp20                 # Record length
.Ltmp20:
	.short	4355                            # Record kind: S_BLOCK32
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	.Ltmp4-.Ltmp1                   # Code size
	.secrel32	.Ltmp1                  # Function section relative address
	.secidx	.Lfunc_begin0                   # Function section index
	.byte	0                               # Lexical block name
	.p2align	2, 0x0
.Ltmp21:
	.short	.Ltmp23-.Ltmp22                 # Record length
.Ltmp22:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	0                               # Flags
	.asciz	"i"
	.p2align	2, 0x0
.Ltmp23:
	.cv_def_range	 .Ltmp1 .Ltmp4, frame_ptr_rel, 4
	.short	2                               # Record length
	.short	6                               # Record kind: S_END
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp13:
	.p2align	2, 0x0
	.cv_linetable	0, main, .Lfunc_end0
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	.Ltmp25-.Ltmp24                 # Subsection size
.Ltmp24:
	.short	.Ltmp27-.Ltmp26                 # Record length
.Ltmp26:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4104                            # LF_BUILDINFO index
	.p2align	2, 0x0
.Ltmp27:
.Ltmp25:
	.p2align	2, 0x0
	.section	.debug$T,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	# ArgList (0x1000)
	.short	0x6                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x0                             # NumArgs
	# Procedure (0x1001)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x0                             # NumParameters
	.long	0x1000                          # ArgListType: ()
	# FuncId (0x1002)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: int ()
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
