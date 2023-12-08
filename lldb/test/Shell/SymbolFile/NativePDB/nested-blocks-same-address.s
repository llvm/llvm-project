# clang-format off
# REQUIRES: lld, x86

# Test when nested S_BLOCK32 have same address range, ResolveSymbolContext should return the innnermost block.
# RUN: llvm-mc -triple=x86_64-windows-msvc --filetype=obj %s > %t.obj
# RUN: lld-link /debug:full /nodefaultlib /entry:main %t.obj /out:%t.exe /base:0x140000000
# RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -o "image lookup -a 0x14000103c -v" -o "exit" | FileCheck %s

# This file is compiled from following source file:
# $ clang-cl /Z7 /GS- /c /O2 test.cpp /Fatest.s
# __attribute__((optnone)) bool func(const char* cp, volatile char p[]) {
#   return false;
# }
#
# int main() {
#   const char* const kMfDLLs[] = {"a"};
#   asm("nop");
#   for (const char* kMfDLL : kMfDLLs) {
#     volatile char path[10] = {0};
#     if (func(kMfDLL, path))
#       break;
#   }
#   return 0;
# }

# CHECK:       Function: id = {{.*}}, name = "main", range = [0x0000000140001020-0x000000014000104d)
# CHECK-NEXT:  FuncType: id = {{.*}}, byte-size = 0, compiler_type = "int (void)"
# CHECK-NEXT:    Blocks: id = {{.*}}, range = [0x140001020-0x14000104d)
# CHECK-NEXT:            id = {{.*}}, range = [0x140001025-0x140001046)
# CHECK-NEXT:            id = {{.*}}, range = [0x140001025-0x140001046)
# CHECK-NEXT:            id = {{.*}}, range = [0x140001025-0x140001046)
# CHECK-NEXT: LineEntry: [0x0000000140001035-0x0000000140001046): /tmp/test.cpp:10
# CHECK-NEXT:  Variable: id = {{.*}}, name = "path", type = "volatile char[10]", valid ranges = <block>, location = [0x0000000140001025, 0x0000000140001046) -> DW_OP_breg7 RSP+40, decl =
# CHECK-NEXT:  Variable: id = {{.*}}, name = "kMfDLL", type = "const char *", valid ranges = <block>, location = [0x000000014000103c, 0x0000000140001046) -> DW_OP_reg2 RCX, decl =
# CHECK-NEXT:  Variable: id = {{.*}}, name = "__range1", type = "const char *const (&)[1]", valid ranges = <block>, location = <empty>, decl =
# CHECK-NEXT:  Variable: id = {{.*}}, name = "__begin1", type = "const char *const *", valid ranges = <block>, location = <empty>, decl =
# CHECK-NEXT:  Variable: id = {{.*}}, name = "__end1", type = "const char *const *", valid ranges = <block>, location = <empty>, decl =
# CHECK-NEXT:  Variable: id = {{.*}}, name = "kMfDLLs", type = "const char *const[1]", valid ranges = <block>, location = [0x000000014000103c, 0x0000000140001046) -> DW_OP_reg2 RCX, decl =


	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.intel_syntax noprefix
	.file	"test.cpp"
	.def	"?func@@YA_NPEBDQECD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,"?func@@YA_NPEBDQECD@Z"
	.globl	"?func@@YA_NPEBDQECD@Z"         # -- Begin function ?func@@YA_NPEBDQECD@Z
	.p2align	4, 0x90
"?func@@YA_NPEBDQECD@Z":                # @"?func@@YA_NPEBDQECD@Z"
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "/tmp/test.cpp" "8CDAA03EE93954606427F9B409CE7638" 1
	.cv_loc	0 1 1 0                         # test.cpp:1:0
.seh_proc "?func@@YA_NPEBDQECD@Z"
# %bb.0:                                # %entry
	sub	rsp, 16
	.seh_stackalloc 16
	.seh_endprologue
	mov	qword ptr [rsp + 8], rdx
	mov	qword ptr [rsp], rcx
.Ltmp0:
	.cv_loc	0 1 2 0                         # test.cpp:2:0
	xor	eax, eax
	and	al, 1
	movzx	eax, al
	add	rsp, 16
	ret
.Ltmp1:
.Lfunc_end0:
	.seh_endproc
                                        # -- End function
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,main
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.Lfunc_begin1:
	.cv_func_id 1
	.cv_loc	1 1 5 0                         # test.cpp:5:0
.seh_proc main
# %bb.0:                                # %entry
	sub	rsp, 56
	.seh_stackalloc 56
	.seh_endprologue
.Ltmp2:
	.cv_loc	1 1 7 0                         # test.cpp:7:0
	#APP
	nop
	#NO_APP
.Ltmp3:
	#DEBUG_VALUE: __range1 <- undef
	#DEBUG_VALUE: __begin1 <- undef
	#DEBUG_VALUE: __end1 <- [DW_OP_plus_uconst 8, DW_OP_stack_value] undef
	.cv_loc	1 1 9 0                         # test.cpp:9:0
	mov	word ptr [rsp + 48], 0
	mov	qword ptr [rsp + 40], 0
	.cv_loc	1 1 10 0                        # test.cpp:10:0
	lea	rcx, [rip + "??_C@_01MCMALHOG@a?$AA@"]
.Ltmp4:
	#DEBUG_VALUE: main:kMfDLLs <- $rcx
	#DEBUG_VALUE: kMfDLL <- $rcx
	lea	rdx, [rsp + 40]
	call	"?func@@YA_NPEBDQECD@Z"
.Ltmp5:
	#DEBUG_VALUE: __begin1 <- [DW_OP_LLVM_arg 0, DW_OP_LLVM_arg 1, DW_OP_constu 8, DW_OP_mul, DW_OP_plus, DW_OP_stack_value] undef, undef
	.cv_loc	1 1 14 0                        # test.cpp:14:0
	xor	eax, eax
	add	rsp, 56
	ret
.Ltmp6:
.Lfunc_end1:
	.seh_endproc
                                        # -- End function
	.section	.rdata,"dr",discard,"??_C@_01MCMALHOG@a?$AA@"
	.globl	"??_C@_01MCMALHOG@a?$AA@"       # @"??_C@_01MCMALHOG@a?$AA@"
"??_C@_01MCMALHOG@a?$AA@":
	.asciz	"a"

	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp8-.Ltmp7                   # Subsection size
.Ltmp7:
	.short	.Ltmp12-.Ltmp11                 # Record length
.Ltmp11:
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
.Ltmp12:
.Ltmp8:
	.p2align	2, 0x0
	.section	.debug$S,"dr",associative,"?func@@YA_NPEBDQECD@Z"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241                             # Symbol subsection for func
.Ltmp14:
	.p2align	2, 0x0
	.cv_linetable	0, "?func@@YA_NPEBDQECD@Z", .Lfunc_end0
	.section	.debug$S,"dr",associative,main
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241                             # Symbol subsection for main
	.long	.Ltmp24-.Ltmp23                 # Subsection size
.Ltmp23:
	.short	.Ltmp26-.Ltmp25                 # Record length
.Ltmp25:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end1-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4105                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2, 0x0
.Ltmp26:
	.short	.Ltmp28-.Ltmp27                 # Record length
.Ltmp27:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	56                              # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	1130504                         # Flags (defines frame register)
	.p2align	2, 0x0
.Ltmp28:
	.short	.Ltmp30-.Ltmp29                 # Record length
.Ltmp29:
	.short	4414                            # Record kind: S_LOCAL
	.long	4107                            # TypeIndex
	.short	0                               # Flags
	.asciz	"kMfDLLs"
	.p2align	2, 0x0
.Ltmp30:
	.cv_def_range	 .Ltmp4 .Ltmp5, reg, 330
	.short	.Ltmp32-.Ltmp31                 # Record length
.Ltmp31:
	.short	4355                            # Record kind: S_BLOCK32
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	.Ltmp5-.Ltmp3                   # Code size
	.secrel32	.Ltmp3                  # Function section relative address
	.secidx	.Lfunc_begin1                   # Function section index
	.byte	0                               # Lexical block name
	.p2align	2, 0x0
.Ltmp32:
	.short	.Ltmp34-.Ltmp33                 # Record length
.Ltmp33:
	.short	4414                            # Record kind: S_LOCAL
	.long	4108                            # TypeIndex
	.short	256                             # Flags
	.asciz	"__range1"
	.p2align	2, 0x0
.Ltmp34:
	.short	.Ltmp36-.Ltmp35                 # Record length
.Ltmp35:
	.short	4414                            # Record kind: S_LOCAL
	.long	4109                            # TypeIndex
	.short	256                             # Flags
	.asciz	"__begin1"
	.p2align	2, 0x0
.Ltmp36:
	.short	.Ltmp38-.Ltmp37                 # Record length
.Ltmp37:
	.short	4414                            # Record kind: S_LOCAL
	.long	4109                            # TypeIndex
	.short	256                             # Flags
	.asciz	"__end1"
	.p2align	2, 0x0
.Ltmp38:
	.short	.Ltmp40-.Ltmp39                 # Record length
.Ltmp39:
	.short	4355                            # Record kind: S_BLOCK32
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	.Ltmp5-.Ltmp3                   # Code size
	.secrel32	.Ltmp3                  # Function section relative address
	.secidx	.Lfunc_begin1                   # Function section index
	.byte	0                               # Lexical block name
	.p2align	2, 0x0
.Ltmp40:
	.short	.Ltmp42-.Ltmp41                 # Record length
.Ltmp41:
	.short	4414                            # Record kind: S_LOCAL
	.long	4097                            # TypeIndex
	.short	0                               # Flags
	.asciz	"kMfDLL"
	.p2align	2, 0x0
.Ltmp42:
	.cv_def_range	 .Ltmp4 .Ltmp5, reg, 330
	.short	.Ltmp44-.Ltmp43                 # Record length
.Ltmp43:
	.short	4355                            # Record kind: S_BLOCK32
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	.Ltmp5-.Ltmp3                   # Code size
	.secrel32	.Ltmp3                  # Function section relative address
	.secidx	.Lfunc_begin1                   # Function section index
	.byte	0                               # Lexical block name
	.p2align	2, 0x0
.Ltmp44:
	.short	.Ltmp46-.Ltmp45                 # Record length
.Ltmp45:
	.short	4414                            # Record kind: S_LOCAL
	.long	4110                            # TypeIndex
	.short	0                               # Flags
	.asciz	"path"
	.p2align	2, 0x0
.Ltmp46:
	.cv_def_range	 .Ltmp3 .Ltmp5, frame_ptr_rel, 40
	.short	2                               # Record length
	.short	6                               # Record kind: S_END
	.short	2                               # Record length
	.short	6                               # Record kind: S_END
	.short	2                               # Record length
	.short	6                               # Record kind: S_END
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp24:
	.p2align	2, 0x0
	.cv_linetable	1, main, .Lfunc_end1
	.section	.debug$S,"dr"
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
.Ltmp50:
.Ltmp48:
	.p2align	2, 0x0
	.section	.debug$T,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	# Modifier (0x1000)
	.short	0xa                             # Record length
	.short	0x1001                          # Record kind: LF_MODIFIER
	.long	0x70                            # ModifiedType: char
	.short	0x1                             # Modifiers ( Const (0x1) )
	.byte	242
	.byte	241
	# Pointer (0x1001)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x1000                          # PointeeType: const char
	.long	0x1000c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
	# Modifier (0x1002)
	.short	0xa                             # Record length
	.short	0x1001                          # Record kind: LF_MODIFIER
	.long	0x70                            # ModifiedType: char
	.short	0x2                             # Modifiers ( Volatile (0x2) )
	.byte	242
	.byte	241
	# Pointer (0x1003)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x1002                          # PointeeType: volatile char
	.long	0x1000c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
	# ArgList (0x1004)
	.short	0xe                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x2                             # NumArgs
	.long	0x1001                          # Argument: const char*
	.long	0x1003                          # Argument: volatile char*
	# Procedure (0x1005)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x30                            # ReturnType: bool
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x2                             # NumParameters
	.long	0x1004                          # ArgListType: (const char*, volatile char*)
	# FuncId (0x1006)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1005                          # FunctionType: bool (const char*, volatile char*)
	.asciz	"func"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# ArgList (0x1007)
	.short	0x6                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x0                             # NumArgs
	# Procedure (0x1008)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x0                             # NumParameters
	.long	0x1007                          # ArgListType: ()
	# FuncId (0x1009)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1008                          # FunctionType: int ()
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Pointer (0x100A)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x1000                          # PointeeType: const char
	.long	0x1040c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8, isConst ]
	# Array (0x100B)
	.short	0xe                             # Record length
	.short	0x1503                          # Record kind: LF_ARRAY
	.long	0x100a                          # ElementType: const char* const
	.long	0x23                            # IndexType: unsigned __int64
	.short	0x8                             # SizeOf
	.byte	0                               # Name
	.byte	241
	# Pointer (0x100C)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x100b                          # PointeeType
	.long	0x1002c                         # Attrs: [ Type: Near64, Mode: LValueReference, SizeOf: 8 ]
	# Pointer (0x100D)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x100a                          # PointeeType: const char* const
	.long	0x1000c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
	# Array (0x100E)
	.short	0xe                             # Record length
	.short	0x1503                          # Record kind: LF_ARRAY
	.long	0x1002                          # ElementType: volatile char
	.long	0x23                            # IndexType: unsigned __int64
	.short	0xa                             # SizeOf
	.byte	0                               # Name
	.byte	241
