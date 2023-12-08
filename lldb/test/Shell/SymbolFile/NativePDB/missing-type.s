# clang-format off
# REQUIRES: lld, x86

# Test when type index is missing in FieldList.
# RUN: llvm-mc -triple=x86_64-windows-msvc --filetype=obj %s > %t.obj
# RUN: lld-link /debug:full /nodefaultlib /entry:main %t.obj /out:%t.exe /base:0x140000000
# RUN: lldb-test symbols --find=type --name=S %t.exe | FileCheck %s

# CHECK:      name = "S", size = 4, compiler_type = {{.*}} struct S {
# CHECK-NEXT: }



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
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.seh_proc main
# %bb.0:                                # %entry
	sub	rsp, 24
	.seh_stackalloc 24
	.seh_endprologue
	mov	dword ptr [rsp + 20], 0
	mov	qword ptr [rsp + 8], rdx
	mov	dword ptr [rsp + 4], ecx
.Ltmp0:
	mov	eax, dword ptr [rsp]
	add	rsp, 24
	ret
.Ltmp1:
.Lfunc_end0:
	.seh_endproc
                                        # -- End function
	.section	.drectve,"yn"
.Ltmp25:
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	# Pointer (0x1000)
	.short	0xa                             # Record length
	.short	0x1002                          # Record kind: LF_POINTER
	.long	0x670                           # PointeeType: char*
	.long	0x1000c                         # Attrs: [ Type: Near64, Mode: Pointer, SizeOf: 8 ]
	# ArgList (0x1001)
	.short	0xe                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x2                             # NumArgs
	.long	0x74                            # Argument: int
	.long	0x1000                          # Argument: char**
	# Procedure (0x1002)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x2                             # NumParameters
	.long	0x1001                          # ArgListType: (int, char**)
	# FuncId (0x1003)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1002                          # FunctionType: int (int, char**)
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Struct (0x1004)
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
	# FieldList (0x1005)
	.short	0xe                             # Record length
	.short	0x1203                          # Record kind: LF_FIELDLIST
	.short	0x150d                          # Member kind: DataMember ( LF_MEMBER )
	.short	0x3                             # Attrs: Public
	.long	0                               # Type. It's intentionally written as 0 for testing.
	.short	0x0                             # FieldOffset
	.asciz	"x"                             # Name
	# Struct (0x1006)
	.short	0x1e                            # Record length
	.short	0x1505                          # Record kind: LF_STRUCTURE
	.short	0x1                             # MemberCount
	.short	0x200                           # Properties ( HasUniqueName (0x200) )
	.long	0x1005                          # FieldList: <field list>
	.long	0x0                             # DerivedFrom
	.long	0x0                             # VShape
	.short	0x4                             # SizeOf
	.asciz	"S"                             # Name
	.asciz	".?AUS@@"                       # LinkageName
