# clang-format off
# REQUIRES: lld, system-windows

# RUN: %clang_cl --target=x86_64-windows-msvc /Fo%t.obj %s
# RUN: lld-link /debug %t.obj /out:%t.exe /base:0x140000000
# RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
# RUN:     %p/Inputs/local-variables-registers.lldbinit 2>&1 | FileCheck %s

# This file is compiled from following source file:
# clang-cl /Z7 /O1 /Falocal-variables-registers.s a.cpp
# struct S {
#   int x;
#   char y;
# };
# 
# __attribute__((noinline)) S CreateS(int p1, char p2) {
#   S s;
#   s.x = p1 + 1;
#   s.y = p2 + 2;
#   ++s.x;
#   ++s.y;
#   return s;
# }
# 
# int main(int argc, char** argv) {
#   int local = argc * 2;
#   S s = CreateS(local, 'a');
#   return s.x + s.y;
# }

# FIXME: The following variable location have wrong register numbers due to 
# https://github.com/llvm/llvm-project/issues/53575. Fix them after resolving 
# the issue.

# CHECK:      (lldb) image lookup -a 0x140001000 -v
# CHECK:          LineEntry: [0x0000000140001000-0x0000000140001003): C:\src\test\a.cpp:10
# CHECK-NEXT:      Variable: id = {{.*}}, name = "p1", type = "int", valid ranges = [0x0000000140001000-0x0000000140001003), location = DW_OP_reg26 XMM9
# CHECK-NEXT:      Variable: id = {{.*}}, name = "p2", type = "char", valid ranges = [0x0000000140001000-0x0000000140001006), location = DW_OP_regx 0x3f
# CHECK-EMPTY:
# CHECK:      (lldb) image lookup -a 0x140001003 -v
# CHECK:          LineEntry: [0x0000000140001003-0x0000000140001006): C:\src\test\a.cpp:11
# CHECK-NEXT:      Variable: id = {{.*}}, name = "p2", type = "char", valid ranges = [0x0000000140001000-0x0000000140001006), location = DW_OP_regx 0x3f
# CHECK-EMPTY:
# CHECK:      (lldb) image lookup -a 0x140001006 -v
# CHECK:          LineEntry: [0x0000000140001006-0x0000000140001011): C:\src\test\a.cpp:12
# CHECK-NEXT:      Variable: id = {{.*}}, name = "s", type = "S", valid ranges = [0x0000000140001006-0x0000000140001011), location = DW_OP_reg26 XMM9, DW_OP_piece 0x4, DW_OP_regx 0x3f, DW_OP_piece 0x1
# CHECK-EMPTY:
# CHECK:      (lldb) image lookup -a 0x140001011 -v
# CHECK:          LineEntry: [0x0000000140001011-0x0000000140001015): C:\src\test\a.cpp:15
# CHECK-NEXT:      Variable: id = {{.*}}, name = "argc", type = "int", valid ranges = [0x0000000140001011-0x0000000140001017), location = DW_OP_reg26 XMM9
# CHECK-NEXT:      Variable: id = {{.*}}, name = "argv", type = "char **", valid ranges = [0x0000000140001011-0x0000000140001019), location = DW_OP_reg3 RBX
# CHECK-EMPTY:
# CHECK:      (lldb) image lookup -a 0x140001017 -v
# CHECK:          LineEntry: [0x0000000140001017-0x000000014000101e): C:\src\test\a.cpp:17
# CHECK-NEXT:      Variable: id = {{.*}}, name = "argv", type = "char **", valid ranges = [0x0000000140001011-0x0000000140001019), location = DW_OP_reg3 RBX
# CHECK-NEXT:      Variable: id = {{.*}}, name = "local", type = "int", valid ranges = [0x0000000140001017-0x000000014000101e), location = DW_OP_reg26 XMM9
# CHECK-EMPTY:
# CHECK:      (lldb) image lookup -a 0x140001019 -v
# CHECK:          LineEntry: [0x0000000140001017-0x000000014000101e): C:\src\test\a.cpp:17
# CHECK-NEXT:      Variable: id = {{.*}}, name = "local", type = "int", valid ranges = [0x0000000140001017-0x000000014000101e), location = DW_OP_reg26 XMM9
# CHECK-EMPTY:
# CHECK:      (lldb) image lookup -a 0x14000101e -v
# CHECK:          LineEntry: [0x000000014000101e-0x0000000140001031): C:\src\test\a.cpp:18
# CHECK-NEXT:      Variable: id = {{.*}}, name = "s", type = "S", valid ranges = [0x000000014000101e-0x000000014000102c), location = DW_OP_reg24 XMM7, DW_OP_piece 0x4, DW_OP_piece 0x1
# CHECK-EMPTY:
# CHECK:      (lldb) image lookup -a 0x14000102c -v
# CHECK:          LineEntry: [0x000000014000101e-0x0000000140001031): C:\src\test\a.cpp:18
# CHECK-EMPTY:


	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.intel_syntax noprefix
	.file	"a.cpp"
	.def	 "?CreateS@@YA?AUS@@HD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,"?CreateS@@YA?AUS@@HD@Z"
	.globl	"?CreateS@@YA?AUS@@HD@Z"        # -- Begin function ?CreateS@@YA?AUS@@HD@Z
"?CreateS@@YA?AUS@@HD@Z":               # @"?CreateS@@YA?AUS@@HD@Z"
.Lfunc_begin0:
	.cv_func_id 0
# %bb.0:
	#DEBUG_VALUE: CreateS:p2 <- $dl
	#DEBUG_VALUE: CreateS:p1 <- $ecx
	#DEBUG_VALUE: CreateS:s <- [DW_OP_plus_uconst 1, DW_OP_stack_value, DW_OP_LLVM_fragment 0 32] $ecx
	#DEBUG_VALUE: CreateS:s <- [DW_OP_plus_uconst 2, DW_OP_stack_value, DW_OP_LLVM_fragment 32 8] $dl
	.cv_file	1 "C:\\src\\test\\a.cpp" "446925B46C8C870B01708834F4813A31" 1
	.cv_loc	0 1 10 0                        # a.cpp:10:0
                                        # kill: def $ecx killed $ecx def $rcx
	#DEBUG_VALUE: CreateS:s <- [DW_OP_plus_uconst 1, DW_OP_stack_value, DW_OP_LLVM_fragment 0 32] $ecx
	add	ecx, 2
.Ltmp0:
	#DEBUG_VALUE: CreateS:p1 <- [DW_OP_LLVM_entry_value 1] $ecx
	#DEBUG_VALUE: CreateS:s <- [DW_OP_LLVM_fragment 0 32] $ecx
	.cv_loc	0 1 11 0                        # a.cpp:11:0
	add	dl, 3
.Ltmp1:
	#DEBUG_VALUE: CreateS:p2 <- [DW_OP_LLVM_entry_value 1] $dl
	#DEBUG_VALUE: CreateS:s <- [DW_OP_LLVM_fragment 32 8] $dl
	.cv_loc	0 1 12 0                        # a.cpp:12:0
	movzx	eax, dl
	shl	rax, 32
	or	rax, rcx
	ret
.Ltmp2:
.Lfunc_end0:
                                        # -- End function
	.def	 main;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,main
	.globl	main                            # -- Begin function main
main:                                   # @main
.Lfunc_begin1:
	.cv_func_id 1
	.cv_loc	1 1 15 0                        # a.cpp:15:0
.seh_proc main
# %bb.0:
	#DEBUG_VALUE: main:argv <- $rdx
	#DEBUG_VALUE: main:argc <- $ecx
	sub	rsp, 40
	.seh_stackalloc 40
	.seh_endprologue
.Ltmp3:
	.cv_loc	1 1 16 0                        # a.cpp:16:0
	add	ecx, ecx
.Ltmp4:
	#DEBUG_VALUE: main:argc <- [DW_OP_LLVM_entry_value 1] $ecx
	#DEBUG_VALUE: main:local <- $ecx
	.cv_loc	1 1 17 0                        # a.cpp:17:0
	mov	dl, 97
.Ltmp5:
	#DEBUG_VALUE: main:argv <- [DW_OP_LLVM_entry_value 1] $rdx
	call	"?CreateS@@YA?AUS@@HD@Z"
.Ltmp6:
	#DEBUG_VALUE: main:s <- [DW_OP_LLVM_fragment 0 32] $eax
	#DEBUG_VALUE: main:s <- [DW_OP_constu 40, DW_OP_shr, DW_OP_LLVM_convert 64 7, DW_OP_LLVM_convert 24 7, DW_OP_stack_value, DW_OP_LLVM_fragment 40 24] $rax
	#DEBUG_VALUE: main:s <- [DW_OP_constu 32, DW_OP_shr, DW_OP_LLVM_convert 64 7, DW_OP_LLVM_convert 8 7, DW_OP_stack_value, DW_OP_LLVM_fragment 32 8] $rax
	.cv_loc	1 1 18 0                        # a.cpp:18:0
	mov	rcx, rax
	shr	rcx, 8
	sar	ecx, 24
	add	ecx, eax
	mov	eax, ecx
.Ltmp7:
	add	rsp, 40
	ret
.Ltmp8:
.Lfunc_end1:
	.seh_endproc
                                        # -- End function
	.section	.drectve,"yn"
	.ascii	" /DEFAULTLIB:libcmt.lib"
	.ascii	" /DEFAULTLIB:oldnames.lib"
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp10-.Ltmp9                  # Subsection size
.Ltmp9:
	.short	.Ltmp12-.Ltmp11                 # Record length
.Ltmp11:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	208                             # CPUType
	.short	13                              # Frontend version
	.short	0
	.short	0
	.short	0
	.short	13000                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 13.0.0"          # Null-terminated compiler version string
	.p2align	2
.Ltmp12:
.Ltmp10:
	.p2align	2
	.section	.debug$S,"dr",associative,"?CreateS@@YA?AUS@@HD@Z"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241                             # Symbol subsection for CreateS
	.long	.Ltmp14-.Ltmp13                 # Subsection size
.Ltmp13:
	.short	.Ltmp16-.Ltmp15                 # Record length
.Ltmp15:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end0-"?CreateS@@YA?AUS@@HD@Z" # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4103                            # Function type index
	.secrel32	"?CreateS@@YA?AUS@@HD@Z" # Function section relative address
	.secidx	"?CreateS@@YA?AUS@@HD@Z"        # Function section index
	.byte	0                               # Flags
	.asciz	"CreateS"                       # Function name
	.p2align	2
.Ltmp16:
	.short	.Ltmp18-.Ltmp17                 # Record length
.Ltmp17:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	0                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	0                               # Flags (defines frame register)
	.p2align	2
.Ltmp18:
	.short	.Ltmp20-.Ltmp19                 # Record length
.Ltmp19:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"p1"
	.p2align	2
.Ltmp20:
	.cv_def_range	 .Lfunc_begin0 .Ltmp0, reg, 18
	.short	.Ltmp22-.Ltmp21                 # Record length
.Ltmp21:
	.short	4414                            # Record kind: S_LOCAL
	.long	112                             # TypeIndex
	.short	1                               # Flags
	.asciz	"p2"
	.p2align	2
.Ltmp22:
	.cv_def_range	 .Lfunc_begin0 .Ltmp1, reg, 3
	.short	.Ltmp24-.Ltmp23                 # Record length
.Ltmp23:
	.short	4414                            # Record kind: S_LOCAL
	.long	4100                            # TypeIndex
	.short	0                               # Flags
	.asciz	"s"
	.p2align	2
.Ltmp24:
	.cv_def_range	 .Ltmp0 .Lfunc_end0, subfield_reg, 18, 0
	.cv_def_range	 .Ltmp1 .Lfunc_end0, subfield_reg, 3, 4
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp14:
	.p2align	2
	.cv_linetable	0, "?CreateS@@YA?AUS@@HD@Z", .Lfunc_end0
	.section	.debug$S,"dr",associative,main
	.p2align	2
	.long	4                               # Debug section magic
	.long	241                             # Symbol subsection for main
	.long	.Ltmp26-.Ltmp25                 # Subsection size
.Ltmp25:
	.short	.Ltmp28-.Ltmp27                 # Record length
.Ltmp27:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end1-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4107                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2
.Ltmp28:
	.short	.Ltmp30-.Ltmp29                 # Record length
.Ltmp29:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	40                              # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	81920                           # Flags (defines frame register)
	.p2align	2
.Ltmp30:
	.short	.Ltmp32-.Ltmp31                 # Record length
.Ltmp31:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"argc"
	.p2align	2
.Ltmp32:
	.cv_def_range	 .Lfunc_begin1 .Ltmp4, reg, 18
	.short	.Ltmp34-.Ltmp33                 # Record length
.Ltmp33:
	.short	4414                            # Record kind: S_LOCAL
	.long	4104                            # TypeIndex
	.short	1                               # Flags
	.asciz	"argv"
	.p2align	2
.Ltmp34:
	.cv_def_range	 .Lfunc_begin1 .Ltmp5, reg, 331
	.short	.Ltmp36-.Ltmp35                 # Record length
.Ltmp35:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	0                               # Flags
	.asciz	"local"
	.p2align	2
.Ltmp36:
	.cv_def_range	 .Ltmp4 .Ltmp6, reg, 18
	.short	.Ltmp38-.Ltmp37                 # Record length
.Ltmp37:
	.short	4414                            # Record kind: S_LOCAL
	.long	4100                            # TypeIndex
	.short	0                               # Flags
	.asciz	"s"
	.p2align	2
.Ltmp38:
	.cv_def_range	 .Ltmp6 .Ltmp7, subfield_reg, 17, 0
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp26:
	.p2align	2
	.cv_linetable	1, main, .Lfunc_end1
	.section	.debug$S,"dr"
	.long	241
	.long	.Ltmp40-.Ltmp39                 # Subsection size
.Ltmp39:
	.short	.Ltmp42-.Ltmp41                 # Record length
.Ltmp41:
	.short	4360                            # Record kind: S_UDT
	.long	4100                            # Type
	.asciz	"S"
	.p2align	2
.Ltmp42:
.Ltmp40:
	.p2align	2
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	.Ltmp44-.Ltmp43                 # Subsection size
.Ltmp43:
	.short	.Ltmp46-.Ltmp45                 # Record length
.Ltmp45:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4110                            # LF_BUILDINFO index
	.p2align	2
.Ltmp46:
.Ltmp44:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	# Struct (0x1000)
	.short	0x1e                            # Record length
	.short	0x1505                          # Record kind: LF_STRUCTURE
	.short	0x0                             # MemberCount
	.short	0x280                           # Properties ( ForwardReference (0x80) | HasUniqueName (0x200) )
	.long	0x0                             # FieldList
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x0                             # SizeOf
	.asciz	"S"                             # Name
	.asciz	".?AUS@@"                       # LinkageName
	# ArgList (0x1001)
	.short	0xe                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x2                             # NumArgs
	.long	0x74                            # Argument: int
	.long	0x70                            # Argument: char
	# Procedure (0x1002)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x1000                          # ReturnType: S
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x2                             # NumParameters
	.long	0x1001                          # ArgListType: (int, char)
	# FieldList (0x1003)
	.short	0x1a                            # Record length
	.short	0x1203                          # Record kind: LF_FIELDLIST
	.short	0x150d                          # Member kind: DataMember ( LF_MEMBER )
	.short	0x3                             # Attrs: Public
	.long	0x74                            # Type: int
	.short	0x0                             # FieldOffset
	.asciz	"x"                             # Name
	.short	0x150d                          # Member kind: DataMember ( LF_MEMBER )
	.short	0x3                             # Attrs: Public
	.long	0x70                            # Type: char
	.short	0x4                             # FieldOffset
	.asciz	"y"                             # Name
	# Struct (0x1004)
	.short	0x1e                            # Record length
	.short	0x1505                          # Record kind: LF_STRUCTURE
	.short	0x2                             # MemberCount
	.short	0x200                           # Properties ( HasUniqueName (0x200) )
	.long	0x1003                          # FieldList: <field list>
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x8                             # SizeOf
	.asciz	"S"                             # Name
	.asciz	".?AUS@@"                       # LinkageName
	# StringId (0x1005)
	.short	0x1a                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"C:\\src\\test\\a.cpp"          # StringData
	.byte	242
	.byte	241
	# UdtSourceLine (0x1006)
	.short	0xe                             # Record length
	.short	0x1606                          # Record kind: LF_UDT_SRC_LINE
	.long	0x1004                          # UDT: S
	.long	0x1005                          # SourceFile: C:\src\test\a.cpp
	.long	0x1                             # LineNumber
	# FuncId (0x1007)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1002                          # FunctionType: S (int, char)
	.asciz	"CreateS"                       # Name
	# Pointer (0x1008)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x670                           # PointeeType: char*
	.long	0x1000c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
	# ArgList (0x1009)
	.short	0xe                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x2                             # NumArgs
	.long	0x74                            # Argument: int
	.long	0x1008                          # Argument: char**
	# Procedure (0x100A)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x2                             # NumParameters
	.long	0x1009                          # ArgListType: (int, char**)
	# FuncId (0x100B)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x100a                          # FunctionType: int (int, char**)
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x100C)
	.short	0x12                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"C:\\src\\test"                 # StringData
	# StringId (0x100D)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"a.cpp"                         # StringData
	.byte	242
	.byte	241
	# BuildInfo (0x100E)
	.short	0x1a                            # Record length
	.short	0x1603                          # Record kind: LF_BUILDINFO
	.short	0x5                             # NumArgs
	.long	0x100c                          # Argument: C:\src\test
	.long	0x0                             # Argument
	.long	0x100d                          # Argument: a.cpp
	.long	0x0                             # Argument
	.long	0x0                             # Argument
	.byte	242
	.byte	241
	.addrsig
