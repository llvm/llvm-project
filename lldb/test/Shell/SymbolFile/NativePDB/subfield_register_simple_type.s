# clang-format off
# REQUIRES: lld, x86

# RUN: %clang_cl --target=i386-windows-msvc -c /Fo%t.obj -- %s
# RUN: lld-link /debug:full /nodefaultlib /entry:main %t.obj /out:%t.exe /base:0x400000
# RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
# RUN:     %p/Inputs/subfield_register_simple_type.lldbinit 2>&1 | FileCheck %s

# This file is compiled from following source file:
# clang-cl --target=i386-windows-msvc /Z7 /O1 -c /Fa a.cpp
# __int64 __attribute__((optnone)) bar(__int64 x) { return x; };
# __int64 foo(__int64 x) {
#   return bar(x);
# }
#
# int main(int argc, char** argv) {
#   foo(argc);
#   return 0;
# }

# FIXME: The following variable location have wrong register numbers due to
# https://github.com/llvm/llvm-project/issues/53575. Fix them after resolving
# the issue.

# CHECK:      (lldb) image lookup -a 0x40102f -v
# CHECK:      LineEntry: [0x00401026-0x00401039): C:\src\a.cpp:3
# CHECK-NEXT: Variable: id = {{.*}}, name = "x", type = "int64_t", valid ranges = [0x0040102f-0x00401036), location = DW_OP_reg0 EAX, DW_OP_piece 0x4, DW_OP_reg2 EDX, DW_OP_piece 0x4, decl =

	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 1
	.intel_syntax noprefix
	.file	"a.cpp"
	.def	"?bar@@YA_J_J@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,"?bar@@YA_J_J@Z"
	.globl	"?bar@@YA_J_J@Z"                # -- Begin function ?bar@@YA_J_J@Z
	.p2align	4, 0x90
"?bar@@YA_J_J@Z":                       # @"?bar@@YA_J_J@Z"
Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "C:\\src\\a.cpp" "CB99424BC3DD1AB059A2DBC6841147F2" 1
	.cv_loc	0 1 1 0                         # a.cpp:1:0
	.cv_fpo_proc	"?bar@@YA_J_J@Z" 8
# %bb.0:                                # %entry
	push	ebp
	.cv_fpo_pushreg	ebp
	mov	ebp, esp
	.cv_fpo_setframe	ebp
	and	esp, -8
	.cv_fpo_stackalign	8
	sub	esp, 8
	.cv_fpo_stackalloc	8
	.cv_fpo_endprologue
	mov	eax, dword ptr [ebp + 8]
	mov	ecx, dword ptr [ebp + 12]
	mov	dword ptr [esp], eax
	mov	dword ptr [esp + 4], ecx
Ltmp0:
	mov	eax, dword ptr [esp]
	mov	edx, dword ptr [esp + 4]
	mov	esp, ebp
	pop	ebp
	ret
Ltmp1:
	.cv_fpo_endproc
Lfunc_end0:
                                        # -- End function
	.def	"?foo@@YA_J_J@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,"?foo@@YA_J_J@Z"
	.globl	"?foo@@YA_J_J@Z"                # -- Begin function ?foo@@YA_J_J@Z
"?foo@@YA_J_J@Z":                       # @"?foo@@YA_J_J@Z"
Lfunc_begin1:
	.cv_func_id 1
	.cv_fpo_proc	"?foo@@YA_J_J@Z" 8
# %bb.0:                                # %entry
	#DEBUG_VALUE: foo:x <- [DW_OP_plus_uconst 4] [$esp+0]
	.cv_loc	1 1 3 0                         # a.cpp:3:0
	jmp	"?bar@@YA_J_J@Z"                # TAILCALL
Ltmp2:
	.cv_fpo_endproc
Lfunc_end1:
                                        # -- End function
	.def	_main;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,_main
	.globl	_main                           # -- Begin function main
_main:                                  # @main
Lfunc_begin2:
	.cv_func_id 2
	.cv_loc	2 1 6 0                         # a.cpp:6:0
	.cv_fpo_proc	_main 8
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argv <- [DW_OP_plus_uconst 8] [$esp+0]
	#DEBUG_VALUE: main:argc <- [DW_OP_plus_uconst 4] [$esp+0]
	.cv_inline_site_id 3 within 2 inlined_at 1 7 0
	.cv_loc	3 1 3 0                         # a.cpp:3:0
	mov	eax, dword ptr [esp + 4]
	mov	ecx, eax
	sar	ecx, 31
Ltmp3:
	#DEBUG_VALUE: foo:x <- [DW_OP_LLVM_fragment 0 32] $eax
	#DEBUG_VALUE: foo:x <- [DW_OP_LLVM_fragment 32 32] $ecx
	push	ecx
Ltmp4:
	push	eax
	call	"?bar@@YA_J_J@Z"
Ltmp5:
	add	esp, 8
Ltmp6:
	.cv_loc	2 1 8 0                         # a.cpp:8:0
	xor	eax, eax
	ret
Ltmp7:
	.cv_fpo_endproc
Lfunc_end2:
                                        # -- End function
	.section	.drectve,"yn"
	.ascii	" /DEFAULTLIB:libcmt.lib"
	.ascii	" /DEFAULTLIB:oldnames.lib"
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241
	.long	Ltmp9-Ltmp8                     # Subsection size
Ltmp8:
	.short	Ltmp11-Ltmp10                   # Record length
Ltmp10:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.asciz	"C:\\src\\a.obj"                # Object name
	.p2align	2
Ltmp11:
	.short	Ltmp13-Ltmp12                   # Record length
Ltmp12:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	7                               # CPUType
	.short	15                              # Frontend version
	.short	0
	.short	0
	.short	0
	.short	15000                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 15.0.0"          # Null-terminated compiler version string
	.p2align	2
Ltmp13:
Ltmp9:
	.p2align	2
	.long	246                             # Inlinee lines subsection
	.long	Ltmp15-Ltmp14                   # Subsection size
Ltmp14:
	.long	0                               # Inlinee lines signature

                                        # Inlined function foo starts at a.cpp:2
	.long	4098                            # Type index of inlined function
	.cv_filechecksumoffset	1               # Offset into filechecksum table
	.long	2                               # Starting line number
Ltmp15:
	.p2align	2
	.section	.debug$S,"dr",associative,"?bar@@YA_J_J@Z"
	.p2align	2
	.long	4                               # Debug section magic
	.cv_fpo_data	"?bar@@YA_J_J@Z"
	.long	241                             # Symbol subsection for bar
	.long	Ltmp17-Ltmp16                   # Subsection size
Ltmp16:
	.short	Ltmp19-Ltmp18                   # Record length
Ltmp18:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	Lfunc_end0-"?bar@@YA_J_J@Z"     # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4099                            # Function type index
	.secrel32	"?bar@@YA_J_J@Z"        # Function section relative address
	.secidx	"?bar@@YA_J_J@Z"                # Function section index
	.byte	0                               # Flags
	.asciz	"bar"                           # Function name
	.p2align	2
Ltmp19:
	.short	Ltmp21-Ltmp20                   # Record length
Ltmp20:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	12                              # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	147456                          # Flags (defines frame register)
	.p2align	2
Ltmp21:
	.short	Ltmp23-Ltmp22                   # Record length
Ltmp22:
	.short	4414                            # Record kind: S_LOCAL
	.long	19                              # TypeIndex
	.short	1                               # Flags
	.asciz	"x"
	.p2align	2
Ltmp23:
	.cv_def_range	 Ltmp0 Ltmp1, reg_rel, 30006, 0, -8
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
Ltmp17:
	.p2align	2
	.cv_linetable	0, "?bar@@YA_J_J@Z", Lfunc_end0
	.section	.debug$S,"dr",associative,"?foo@@YA_J_J@Z"
	.p2align	2
	.long	4                               # Debug section magic
	.cv_fpo_data	"?foo@@YA_J_J@Z"
	.long	241                             # Symbol subsection for foo
	.long	Ltmp25-Ltmp24                   # Subsection size
Ltmp24:
	.short	Ltmp27-Ltmp26                   # Record length
Ltmp26:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	Lfunc_end1-"?foo@@YA_J_J@Z"     # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4098                            # Function type index
	.secrel32	"?foo@@YA_J_J@Z"        # Function section relative address
	.secidx	"?foo@@YA_J_J@Z"                # Function section index
	.byte	0                               # Flags
	.asciz	"foo"                           # Function name
	.p2align	2
Ltmp27:
	.short	Ltmp29-Ltmp28                   # Record length
Ltmp28:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	0                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	0                               # Flags (defines frame register)
	.p2align	2
Ltmp29:
	.short	Ltmp31-Ltmp30                   # Record length
Ltmp30:
	.short	4414                            # Record kind: S_LOCAL
	.long	19                              # TypeIndex
	.short	1                               # Flags
	.asciz	"x"
	.p2align	2
Ltmp31:
	.cv_def_range	 Lfunc_begin1 Lfunc_end1, reg_rel, 30006, 0, 4
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
Ltmp25:
	.p2align	2
	.cv_linetable	1, "?foo@@YA_J_J@Z", Lfunc_end1
	.section	.debug$S,"dr",associative,_main
	.p2align	2
	.long	4                               # Debug section magic
	.cv_fpo_data	_main
	.long	241                             # Symbol subsection for main
	.long	Ltmp33-Ltmp32                   # Subsection size
Ltmp32:
	.short	Ltmp35-Ltmp34                   # Record length
Ltmp34:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	Lfunc_end2-_main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4103                            # Function type index
	.secrel32	_main                   # Function section relative address
	.secidx	_main                           # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2
Ltmp35:
	.short	Ltmp37-Ltmp36                   # Record length
Ltmp36:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	0                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	0                               # Flags (defines frame register)
	.p2align	2
Ltmp37:
	.short	Ltmp39-Ltmp38                   # Record length
Ltmp38:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"argc"
	.p2align	2
Ltmp39:
	.cv_def_range	 Lfunc_begin2 Ltmp4, reg_rel, 30006, 0, 4
	.short	Ltmp41-Ltmp40                   # Record length
Ltmp40:
	.short	4414                            # Record kind: S_LOCAL
	.long	4100                            # TypeIndex
	.short	1                               # Flags
	.asciz	"argv"
	.p2align	2
Ltmp41:
	.cv_def_range	 Lfunc_begin2 Ltmp4, reg_rel, 30006, 0, 8
	.short	Ltmp43-Ltmp42                   # Record length
Ltmp42:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4098                            # Inlinee type index
	.cv_inline_linetable	3 1 2 Lfunc_begin2 Lfunc_end2
	.p2align	2
Ltmp43:
	.short	Ltmp45-Ltmp44                   # Record length
Ltmp44:
	.short	4414                            # Record kind: S_LOCAL
	.long	19                              # TypeIndex
	.short	1                               # Flags
	.asciz	"x"
	.p2align	2
Ltmp45:
	.cv_def_range	 Ltmp3 Ltmp5, subfield_reg, 17, 0
	.cv_def_range	 Ltmp3 Ltmp5, subfield_reg, 18, 4
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
Ltmp33:
	.p2align	2
	.cv_linetable	2, _main, Lfunc_end2
	.section	.debug$S,"dr"
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	Ltmp47-Ltmp46                   # Subsection size
Ltmp46:
	.short	Ltmp49-Ltmp48                   # Record length
Ltmp48:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4109                            # LF_BUILDINFO index
	.p2align	2
Ltmp49:
Ltmp47:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	# ArgList (0x1000)
	.short	0xa                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x1                             # NumArgs
	.long	0x13                            # Argument: __int64
	# Procedure (0x1001)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x13                            # ReturnType: __int64
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1000                          # ArgListType: (__int64)
	# FuncId (0x1002)
	.short	0xe                             # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: __int64 (__int64)
	.asciz	"foo"                           # Name
	# FuncId (0x1003)
	.short	0xe                             # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: __int64 (__int64)
	.asciz	"bar"                           # Name
	# Pointer (0x1004)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x470                           # PointeeType: char*
	.long	0x800a                          # Attrs: [ Type: Near32, Mode: Pointer, SizeOf: 4 ]
	# ArgList (0x1005)
	.short	0xe                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x2                             # NumArgs
	.long	0x74                            # Argument: int
	.long	0x1004                          # Argument: char**
	# Procedure (0x1006)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x2                             # NumParameters
	.long	0x1005                          # ArgListType: (int, char**)
	# FuncId (0x1007)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1006                          # FunctionType: int (int, char**)
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x1008)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"C:\\src"                       # StringData
	.byte	241
	# StringId (0x1009)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"a.cpp"                         # StringData
	.byte	242
	.byte	241
