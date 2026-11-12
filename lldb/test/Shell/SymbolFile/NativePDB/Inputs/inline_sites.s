# Compiled from the following files, but replaced the call to abort with nop.
# clang-cl -fuse-ld=lld-link /Z7 /O1 /Faa.asm /winsysroot~/win_toolchain a.cpp
# a.cpp:
# #include "a.h"
# int main(int argc, char** argv) {
#   volatile int main_local = Namespace1::foo(2);
#   return 0;
# }
# a.h:
# #include <stdlib.h>
# #include "b.h"
# namespace Namespace1 {
# inline int foo(int x) {
#   volatile int foo_local = x + 1;
#   ++foo_local;
#   if (!foo_local)
#     abort();
#   return Class1::bar(foo_local);
# }
# } // namespace Namespace1
# b.h:
# #include "c.h"
# class Class1 {
# public:
#   inline static int bar(int x) {
#     volatile int bar_local = x + 1;
#     ++bar_local;
#     return Namespace2::Class2::func(bar_local);
#   }
# };
# c.h:
# namespace Namespace2 {
# class Class2 {
# public:
#   inline static int func(int x) {
#     volatile int func_local = x + 1;
#     func_local += x;
#     return func_local;
#   }
# };
# } // namespace Namespace2

	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.intel_syntax noprefix
	.file	"a.cpp"
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,main
	.globl	main                            # -- Begin function main
main:                                   # @main
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "/tmp/a.cpp" "4FFB96E5DF1A95CE7DB9732CFFE001D7" 1
	.cv_loc	0 1 2 0                         # a.cpp:2:0
.seh_proc main
# %bb.0:
	#DEBUG_VALUE: main:argv <- $rdx
	#DEBUG_VALUE: main:argc <- $ecx
	#DEBUG_VALUE: foo:x <- 2
	sub	rsp, 56
	.seh_stackalloc 56
	.seh_endprologue
.Ltmp0:
	.cv_file	2 "/tmp/./a.h" "BBFED90EF093E9C1D032CC9B05B5D167" 1
	.cv_inline_site_id 1 within 0 inlined_at 1 3 0
	.cv_loc	1 2 5 0                         # ./a.h:5:0
	mov	dword ptr [rsp + 44], 3
	.cv_loc	1 2 6 0                         # ./a.h:6:0
	inc	dword ptr [rsp + 44]
	.cv_loc	1 2 7 0                         # ./a.h:7:0
	mov	eax, dword ptr [rsp + 44]
	test	eax, eax
	je	.LBB0_2
.Ltmp1:
# %bb.1:
	#DEBUG_VALUE: main:argv <- $rdx
	#DEBUG_VALUE: main:argc <- $ecx
	#DEBUG_VALUE: foo:x <- 2
	.cv_loc	1 2 9 0                         # ./a.h:9:0
	mov	eax, dword ptr [rsp + 44]
.Ltmp2:
	#DEBUG_VALUE: bar:x <- $eax
	.cv_file	3 "/tmp/./b.h" "A26CC743A260115F33AF91AB11F95877" 1
	.cv_inline_site_id 2 within 1 inlined_at 2 9 0
	.cv_loc	2 3 5 0                         # ./b.h:5:0
	inc	eax
.Ltmp3:
	mov	dword ptr [rsp + 52], eax
	.cv_loc	2 3 6 0                         # ./b.h:6:0
	inc	dword ptr [rsp + 52]
	.cv_loc	2 3 7 0                         # ./b.h:7:0
	mov	eax, dword ptr [rsp + 52]
.Ltmp4:
	#DEBUG_VALUE: func:x <- $eax
	.cv_file	4 "/tmp/./c.h" "8AF4613F78624BBE96D1C408ABA39B2D" 1
	.cv_inline_site_id 3 within 2 inlined_at 3 7 0
	.cv_loc	3 4 5 0                         # ./c.h:5:0
	lea	ecx, [rax + 1]
.Ltmp5:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $ecx
	mov	dword ptr [rsp + 48], ecx
	.cv_loc	3 4 6 0                         # ./c.h:6:0
	add	dword ptr [rsp + 48], eax
	.cv_loc	3 4 7 0                         # ./c.h:7:0
	mov	eax, dword ptr [rsp + 48]
.Ltmp6:
	.cv_loc	0 1 3 0                         # a.cpp:3:0
	mov	dword ptr [rsp + 48], eax
	.cv_loc	0 1 4 0                         # a.cpp:4:0
	xor	eax, eax
	# Use fake debug info to tests inline info.
	.cv_loc	1 2 20 0
	add	rsp, 56
	ret
.Ltmp7:
.LBB0_2:
	#DEBUG_VALUE: main:argv <- $rdx
	#DEBUG_VALUE: main:argc <- $ecx
	#DEBUG_VALUE: foo:x <- 2
	.cv_loc	1 2 8 0                         # ./a.h:8:0
	nop
.Ltmp8:
	int3
.Ltmp9:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $ecx
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rdx
.Lfunc_end0:
	.seh_endproc
                                        # -- End function
	.section	.drectve,"yn"
	.ascii	" /DEFAULTLIB:libcmt.lib"
	.ascii	" /DEFAULTLIB:oldnames.lib"
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp11-.Ltmp10                 # Subsection size
.Ltmp10:
	.short	.Ltmp13-.Ltmp12                 # Record length
.Ltmp12:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.asciz	"/tmp/a-2b2ba0.obj"             # Object name
	.p2align	2
.Ltmp13:
	.short	.Ltmp15-.Ltmp14                 # Record length
.Ltmp14:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	208                             # CPUType
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
.Ltmp15:
.Ltmp11:
	.p2align	2
	.long	246                             # Inlinee lines subsection
	.long	.Ltmp17-.Ltmp16                 # Subsection size
.Ltmp16:
	.long	0                               # Inlinee lines signature

                                        # Inlined function foo starts at ./a.h:4
	.long	4099                            # Type index of inlined function
	.cv_filechecksumoffset	2               # Offset into filechecksum table
	.long	4                               # Starting line number

                                        # Inlined function bar starts at ./b.h:4
	.long	4106                            # Type index of inlined function
	.cv_filechecksumoffset	3               # Offset into filechecksum table
	.long	4                               # Starting line number

                                        # Inlined function func starts at ./c.h:4
	.long	4113                            # Type index of inlined function
	.cv_filechecksumoffset	4               # Offset into filechecksum table
	.long	4                               # Starting line number
.Ltmp17:
	.p2align	2
	.section	.debug$S,"dr",associative,main
	.p2align	2
	.long	4                               # Debug section magic
	.long	241                             # Symbol subsection for main
	.long	.Ltmp19-.Ltmp18                 # Subsection size
.Ltmp18:
	.short	.Ltmp21-.Ltmp20                 # Record length
.Ltmp20:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end0-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4117                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2
.Ltmp21:
	.short	.Ltmp23-.Ltmp22                 # Record length
.Ltmp22:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	56                              # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	81920                           # Flags (defines frame register)
	.p2align	2
.Ltmp23:
	.short	.Ltmp25-.Ltmp24                 # Record length
.Ltmp24:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"argc"
	.p2align	2
.Ltmp25:
	.cv_def_range	 .Lfunc_begin0 .Ltmp5 .Ltmp7 .Ltmp8, reg, 18
	.short	.Ltmp27-.Ltmp26                 # Record length
.Ltmp26:
	.short	4414                            # Record kind: S_LOCAL
	.long	4114                            # TypeIndex
	.short	1                               # Flags
	.asciz	"argv"
	.p2align	2
.Ltmp27:
	.cv_def_range	 .Lfunc_begin0 .Ltmp8, reg, 331
	.short	.Ltmp29-.Ltmp28                 # Record length
.Ltmp28:
	.short	4414                            # Record kind: S_LOCAL
	.long	4118                            # TypeIndex
	.short	0                               # Flags
	.asciz	"main_local"
	.p2align	2
.Ltmp29:
	.cv_def_range	 .Ltmp0 .Ltmp9, frame_ptr_rel, 48
	.short	.Ltmp31-.Ltmp30                 # Record length
.Ltmp30:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4099                            # Inlinee type index
	.cv_inline_linetable	1 2 4 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp31:
	.short	.Ltmp33-.Ltmp32                 # Record length
.Ltmp32:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	257                             # Flags
	.asciz	"x"
	.p2align	2
.Ltmp33:
	.short	.Ltmp35-.Ltmp34                 # Record length
.Ltmp34:
	.short	4414                            # Record kind: S_LOCAL
	.long	4118                            # TypeIndex
	.short	0                               # Flags
	.asciz	"foo_local"
	.p2align	2
.Ltmp35:
	.cv_def_range	 .Ltmp0 .Ltmp6 .Ltmp7 .Ltmp9, frame_ptr_rel, 44
	.short	.Ltmp37-.Ltmp36                 # Record length
.Ltmp36:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4106                            # Inlinee type index
	.cv_inline_linetable	2 3 4 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp37:
	.short	.Ltmp39-.Ltmp38                 # Record length
.Ltmp38:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"x"
	.p2align	2
.Ltmp39:
	.cv_def_range	 .Ltmp2 .Ltmp3, reg, 17
	.short	.Ltmp41-.Ltmp40                 # Record length
.Ltmp40:
	.short	4414                            # Record kind: S_LOCAL
	.long	4118                            # TypeIndex
	.short	0                               # Flags
	.asciz	"bar_local"
	.p2align	2
.Ltmp41:
	.cv_def_range	 .Ltmp2 .Ltmp6, frame_ptr_rel, 52
	.short	.Ltmp43-.Ltmp42                 # Record length
.Ltmp42:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4113                            # Inlinee type index
	.cv_inline_linetable	3 4 4 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp43:
	.short	.Ltmp45-.Ltmp44                 # Record length
.Ltmp44:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"x"
	.p2align	2
.Ltmp45:
	.cv_def_range	 .Ltmp4 .Ltmp6, reg, 17
	.short	.Ltmp47-.Ltmp46                 # Record length
.Ltmp46:
	.short	4414                            # Record kind: S_LOCAL
	.long	4118                            # TypeIndex
	.short	0                               # Flags
	.asciz	"func_local"
	.p2align	2
.Ltmp47:
	.cv_def_range	 .Ltmp4 .Ltmp6, frame_ptr_rel, 48
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp19:
	.p2align	2
	.cv_linetable	0, main, .Lfunc_end0
	.section	.debug$S,"dr"
	.long	241
	.long	.Ltmp49-.Ltmp48                 # Subsection size
.Ltmp48:
	.short	.Ltmp51-.Ltmp50                 # Record length
.Ltmp50:
	.short	4360                            # Record kind: S_UDT
	.long	4103                            # Type
	.asciz	"Class1"
	.p2align	2
.Ltmp51:
	.short	.Ltmp53-.Ltmp52                 # Record length
.Ltmp52:
	.short	4360                            # Record kind: S_UDT
	.long	4110                            # Type
	.asciz	"Namespace2::Class2"
	.p2align	2
.Ltmp53:
.Ltmp49:
	.p2align	2
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	.Ltmp55-.Ltmp54                 # Subsection size
.Ltmp54:
	.short	.Ltmp57-.Ltmp56                 # Record length
.Ltmp56:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4124                            # LF_BUILDINFO index
	.p2align	2
.Ltmp57:
.Ltmp55:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	# StringId (0x1000)
	.short	0x12                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"Namespace1"                    # StringData
	.byte	241
	# ArgList (0x1001)
	.short	0xa                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x1                             # NumArgs
	.long	0x74                            # Argument: int
	# Procedure (0x1002)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1001                          # ArgListType: (int)
	# FuncId (0x1003)
	.short	0xe                             # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x1000                          # ParentScope: Namespace1
	.long	0x1002                          # FunctionType: int (int)
	.asciz	"foo"                           # Name
	# Class (0x1004)
	.short	0x2a                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x0                             # MemberCount
	.short	0x280                           # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
	.long	0x0                             # FieldList
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x0                             # SizeOf
	.asciz	"Class1"                        # Name
	.asciz	".?AVClass1@@"                  # LinkageName
	.byte	242
	.byte	241
	# MemberFunction (0x1005)
	.short	0x1a                            # Record length
	.short	0x1009                          # Record kind: LF_MFUNCTION
	.long	0x74                            # ReturnType: int
	.long	0x1004                          # ClassType: Class1
	.long	0x0                             # ThisType
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1001                          # ArgListType: (int)
	.long	0x0                             # ThisAdjustment
	# FieldList (0x1006)
	.short	0xe                             # Record length
	.short	0x1203                          # Record kind: LF_FIELDLIST
	.short	0x1511                          # Member kind: OneMethod ( LF_ONEMETHOD )
	.short	0xb                             # Attrs: Public, Static
	.long	0x1005                          # Type: int Class1::(int)
	.asciz	"bar"                           # Name
	# Class (0x1007)
	.short	0x2a                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x1                             # MemberCount
	.short	0x200                           # Properties ( HasUniqueName (0x200) )
	.long	0x1006                          # FieldList: <field list>
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x1                             # SizeOf
	.asciz	"Class1"                        # Name
	.asciz	".?AVClass1@@"                  # LinkageName
	.byte	242
	.byte	241
	# StringId (0x1008)
	.short	0x12                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/tmp/./b.h"                    # StringData
	.byte	241
	# UdtSourceLine (0x1009)
	.short	0xe                             # Record length
	.short	0x1606                          # Record kind: LF_UDT_SRC_LINE
	.long	0x1007                          # UDT: Class1
	.long	0x1008                          # SourceFile: /tmp/./b.h
	.long	0x2                             # LineNumber
	# MemberFuncId (0x100A)
	.short	0xe                             # Record length
	.short	0x1602                          # Record kind: LF_MFUNC_ID
	.long	0x1004                          # ClassType: Class1
	.long	0x1005                          # FunctionType: int Class1::(int)
	.asciz	"bar"                           # Name
	# Class (0x100B)
	.short	0x42                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x0                             # MemberCount
	.short	0x280                           # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
	.long	0x0                             # FieldList
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x0                             # SizeOf
	.asciz	"Namespace2::Class2"            # Name
	.asciz	".?AVClass2@Namespace2@@"       # LinkageName
	.byte	243
	.byte	242
	.byte	241
	# MemberFunction (0x100C)
	.short	0x1a                            # Record length
	.short	0x1009                          # Record kind: LF_MFUNCTION
	.long	0x74                            # ReturnType: int
	.long	0x100b                          # ClassType: Namespace2::Class2
	.long	0x0                             # ThisType
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1001                          # ArgListType: (int)
	.long	0x0                             # ThisAdjustment
	# FieldList (0x100D)
	.short	0x12                            # Record length
	.short	0x1203                          # Record kind: LF_FIELDLIST
	.short	0x1511                          # Member kind: OneMethod ( LF_ONEMETHOD )
	.short	0xb                             # Attrs: Public, Static
	.long	0x100c                          # Type: int Namespace2::Class2::(int)
	.asciz	"func"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Class (0x100E)
	.short	0x42                            # Record length
	.short	0x1504                          # Record kind: LF_CLASS
	.short	0x1                             # MemberCount
	.short	0x200                           # Properties ( HasUniqueName (0x200) )
	.long	0x100d                          # FieldList: <field list>
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x1                             # SizeOf
	.asciz	"Namespace2::Class2"            # Name
	.asciz	".?AVClass2@Namespace2@@"       # LinkageName
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x100F)
	.short	0x12                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/tmp/./c.h"                    # StringData
	.byte	241
	# UdtSourceLine (0x1010)
	.short	0xe                             # Record length
	.short	0x1606                          # Record kind: LF_UDT_SRC_LINE
	.long	0x100e                          # UDT: Namespace2::Class2
	.long	0x100f                          # SourceFile: /tmp/./c.h
	.long	0x2                             # LineNumber
	# MemberFuncId (0x1011)
	.short	0x12                            # Record length
	.short	0x1602                          # Record kind: LF_MFUNC_ID
	.long	0x100b                          # ClassType: Namespace2::Class2
	.long	0x100c                          # FunctionType: int Namespace2::Class2::(int)
	.asciz	"func"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Pointer (0x1012)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x670                           # PointeeType: char*
	.long	0x1000c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
	# ArgList (0x1013)
	.short	0xe                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x2                             # NumArgs
	.long	0x74                            # Argument: int
	.long	0x1012                          # Argument: char**
	# Procedure (0x1014)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x2                             # NumParameters
	.long	0x1013                          # ArgListType: (int, char**)
	# FuncId (0x1015)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1014                          # FunctionType: int (int, char**)
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Modifier (0x1016)
	.short	0xa                             # Record length
	.short	0x1001                          # Record kind: LF_MODIFIER
	.long	0x74                            # ModifiedType: int
	.short	0x2                             # Modifiers ( Volatile (0x2) )
	.byte	242
	.byte	241
	# StringId (0x1017)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/tmp"                          # StringData
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x1018)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"a.cpp"                         # StringData
	.byte	242
	.byte	241
	# StringId (0x1019)
	.short	0xa                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.byte	0                               # StringData
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x101A)
	.short	0x4e                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/usr/local/google/home/zequanwu/llvm-project/build/release/bin/clang" # StringData
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x101B)
	.short	0x9f6                           # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"\"-cc1\" \"-triple\" \"x86_64-pc-windows-msvc19.20.0\" \"-S\" \"-disable-free\" \"-clear-ast-before-backend\" \"-disable-llvm-verifier\" \"-discard-value-names\" \"-mrelocation-model\" \"pic\" \"-pic-level\" \"2\" \"-mframe-pointer=none\" \"-relaxed-aliasing\" \"-fmath-errno\" \"-ffp-contract=on\" \"-fno-rounding-math\" \"-mconstructor-aliases\" \"-funwind-tables=2\" \"-target-cpu\" \"x86-64\" \"-mllvm\" \"-x86-asm-syntax=intel\" \"-tune-cpu\" \"generic\" \"-mllvm\" \"-treat-scalable-fixed-error-as-warning\" \"-D_MT\" \"-flto-visibility-public-std\" \"--dependent-lib=libcmt\" \"--dependent-lib=oldnames\" \"-stack-protector\" \"2\" \"-fms-volatile\" \"-fdiagnostics-format\" \"msvc\" \"-gno-column-info\" \"-gcodeview\" \"-debug-info-kind=constructor\" \"-ffunction-sections\" \"-fcoverage-compilation-dir=/tmp\" \"-resource-dir\" \"/usr/local/google/home/zequanwu/llvm-project/build/release/lib/clang/15.0.0\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/llvm-project/build/release/lib/clang/15.0.0/include\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/DIA SDK/include\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/VC/Tools/MSVC/14.26.28801/include\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/VC/Tools/MSVC/14.26.28801/atlmfc/include\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/ucrt\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/shared\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/um\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/winrt\" \"-internal-isystem\" \"/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/cppwinrt\" \"-Os\" \"-fdeprecated-macro\" \"-fdebug-compilation-dir=/tmp\" \"-ferror-limit\" \"19\" \"-fno-use-cxa-atexit\" \"-fms-extensions\" \"-fms-compatibility\" \"-fms-compatibility-version=19.20\" \"-std=c++14\" \"-fdelayed-template-parsing\" \"-fcolor-diagnostics\" \"-vectorize-loops\" \"-vectorize-slp\" \"-faddrsig\" \"-x\" \"c++\"" # StringData
	.byte	242
	.byte	241
	# BuildInfo (0x101C)
	.short	0x1a                            # Record length
	.short	0x1603                          # Record kind: LF_BUILDINFO
	.short	0x5                             # NumArgs
	.long	0x1017                          # Argument: /tmp
	.long	0x101a                          # Argument: /usr/local/google/home/zequanwu/llvm-project/build/release/bin/clang
	.long	0x1018                          # Argument: a.cpp
	.long	0x1019                          # Argument
	.long	0x101b                          # Argument: "-cc1" "-triple" "x86_64-pc-windows-msvc19.20.0" "-S" "-disable-free" "-clear-ast-before-backend" "-disable-llvm-verifier" "-discard-value-names" "-mrelocation-model" "pic" "-pic-level" "2" "-mframe-pointer=none" "-relaxed-aliasing" "-fmath-errno" "-ffp-contract=on" "-fno-rounding-math" "-mconstructor-aliases" "-funwind-tables=2" "-target-cpu" "x86-64" "-mllvm" "-x86-asm-syntax=intel" "-tune-cpu" "generic" "-mllvm" "-treat-scalable-fixed-error-as-warning" "-D_MT" "-flto-visibility-public-std" "--dependent-lib=libcmt" "--dependent-lib=oldnames" "-stack-protector" "2" "-fms-volatile" "-fdiagnostics-format" "msvc" "-gno-column-info" "-gcodeview" "-debug-info-kind=constructor" "-ffunction-sections" "-fcoverage-compilation-dir=/tmp" "-resource-dir" "/usr/local/google/home/zequanwu/llvm-project/build/release/lib/clang/15.0.0" "-internal-isystem" "/usr/local/google/home/zequanwu/llvm-project/build/release/lib/clang/15.0.0/include" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/DIA SDK/include" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/VC/Tools/MSVC/14.26.28801/include" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/VC/Tools/MSVC/14.26.28801/atlmfc/include" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/ucrt" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/shared" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/um" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/winrt" "-internal-isystem" "/usr/local/google/home/zequanwu/chromium/src/third_party/depot_tools/win_toolchain/vs_files/3bda71a11e/Windows Kits/10/Include/10.0.19041.0/cppwinrt" "-Os" "-fdeprecated-macro" "-fdebug-compilation-dir=/tmp" "-ferror-limit" "19" "-fno-use-cxa-atexit" "-fms-extensions" "-fms-compatibility" "-fms-compatibility-version=19.20" "-std=c++14" "-fdelayed-template-parsing" "-fcolor-diagnostics" "-vectorize-loops" "-vectorize-slp" "-faddrsig" "-x" "c++"
	.byte	242
	.byte	241
	.addrsig
