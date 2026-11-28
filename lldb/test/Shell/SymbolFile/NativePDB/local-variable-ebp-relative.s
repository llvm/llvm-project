# clang-format off
# REQUIRES: lld, x86

# RUN: llvm-mc -triple=i686-pc-windows-msvc --filetype=obj %s > %t.obj
# RUN: lld-link /debug:full /nodefaultlib /entry:mainCRTStartup /subsystem:console \
# RUN:     /base:0xc0000000 /machine:x86 %t.obj /out:%t.exe
# RUN: %lldb -f %t.exe -o "target symbols add %t.pdb" \
# RUN:     -o "disassemble -n Foo" \
# RUN:     -o "image lookup -a 0xc0001029 -v" | FileCheck %s

# This is a regression test for a bug where a 32-bit x86 frame-pointer-relative
# local variable (S_LOCAL + S_DEFRANGE_FRAMEPOINTER_REL, with S_FRAMEPROC
# saying `local fp reg = EBP`) resolved to the wrong live register under a
# RegisterContext whose numbering doesn't match lldb-x86-register-enums.h's
# fixed ordering -- for example, one built by
# GDBRemoteRegisterFallback.cpp's x86 register list, used for a gdb-remote
# connection to a stub with no target-description support (e.g. QEMU's `-s`).
# There, the plugin's old hardcoded "LLDB-kind" register index for EBP (6)
# actually names ESI, so `&xxx` would be computed from the wrong register
# entirely. See https://github.com/llvm/llvm-project/issues/53575.
#
# This file is equivalent to compiling the following with
# `clang-cl -target i686-pc-windows-msvc /Z7 /Od /Oy- /GS- /Gy-`:
#
#   extern "C" void SomeOtherCall() {}
#   extern "C" void AnotherCall() {}
#
#   extern "C" void Foo() {
#       SomeOtherCall();
#       int xxx = 33;
#       AnotherCall();
#   }
#
#   extern "C" int mainCRTStartup() {
#       Foo();
#       return 0;
#   }

# CHECK:      Foo:
# CHECK-NEXT: <+0>:  pushl  %ebp
# CHECK-NEXT: <+1>:  movl   %esp, %ebp
# CHECK-NEXT: <+3>:  pushl  %eax
# CHECK-NEXT: <+4>:  calll  {{.*}} ; SomeOtherCall at a.cpp:1
# CHECK-NEXT: <+9>:  movl   $0x21, -0x4(%ebp)
# CHECK-NEXT: <+16>: calll  {{.*}} ; AnotherCall at a.cpp:2

# The disassembly above shows `xxx` unambiguously lives at `ebp - 4`.
# CHECK:      Variable: id = {{.*}}, name = "xxx", type = "int", valid ranges = <block>, location = {{.*}} -> DW_OP_breg5 EBP-4

	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 1
	.att_syntax
	.file	"a.cpp"
	.def	_SomeOtherCall;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	_SomeOtherCall                  # -- Begin function SomeOtherCall
	.p2align	4
_SomeOtherCall:                         # @SomeOtherCall
Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "/tmp/test/a.cpp" "1A88B5884888984ACBAD88D3B85576A8" 1
	.cv_loc	0 1 1 0                         # a.cpp:1:0
	.cv_fpo_proc	_SomeOtherCall 0
# %bb.0:
	pushl	%ebp
	.cv_fpo_pushreg	%ebp
	movl	%esp, %ebp
	.cv_fpo_setframe	%ebp
	.cv_fpo_endprologue
Ltmp0:
	popl	%ebp
	retl
Ltmp1:
	.cv_fpo_endproc
Lfunc_end0:
                                        # -- End function
	.def	_AnotherCall;
	.scl	2;
	.type	32;
	.endef
	.globl	_AnotherCall                    # -- Begin function AnotherCall
	.p2align	4
_AnotherCall:                           # @AnotherCall
Lfunc_begin1:
	.cv_func_id 1
	.cv_loc	1 1 2 0                         # a.cpp:2:0
	.cv_fpo_proc	_AnotherCall 0
# %bb.0:
	pushl	%ebp
	.cv_fpo_pushreg	%ebp
	movl	%esp, %ebp
	.cv_fpo_setframe	%ebp
	.cv_fpo_endprologue
Ltmp2:
	popl	%ebp
	retl
Ltmp3:
	.cv_fpo_endproc
Lfunc_end1:
                                        # -- End function
	.def	_Foo;
	.scl	2;
	.type	32;
	.endef
	.globl	_Foo                            # -- Begin function Foo
	.p2align	4
_Foo:                                   # @Foo
Lfunc_begin2:
	.cv_func_id 2
	.cv_loc	2 1 4 0                         # a.cpp:4:0
	.cv_fpo_proc	_Foo 0
# %bb.0:
	pushl	%ebp
	.cv_fpo_pushreg	%ebp
	movl	%esp, %ebp
	.cv_fpo_setframe	%ebp
	pushl	%eax
	.cv_fpo_stackalloc	4
	.cv_fpo_endprologue
Ltmp4:
	.cv_loc	2 1 5 0                         # a.cpp:5:0
	calll	_SomeOtherCall
	.cv_loc	2 1 6 0                         # a.cpp:6:0
	movl	$33, -4(%ebp)
	.cv_loc	2 1 7 0                         # a.cpp:7:0
	calll	_AnotherCall
	.cv_loc	2 1 8 0                         # a.cpp:8:0
	addl	$4, %esp
	popl	%ebp
	retl
Ltmp5:
	.cv_fpo_endproc
Lfunc_end2:
                                        # -- End function
	.def	_mainCRTStartup;
	.scl	2;
	.type	32;
	.endef
	.globl	_mainCRTStartup                 # -- Begin function mainCRTStartup
	.p2align	4
_mainCRTStartup:                        # @mainCRTStartup
Lfunc_begin3:
	.cv_func_id 3
	.cv_loc	3 1 10 0                        # a.cpp:10:0
	.cv_fpo_proc	_mainCRTStartup 0
# %bb.0:
	pushl	%ebp
	.cv_fpo_pushreg	%ebp
	movl	%esp, %ebp
	.cv_fpo_setframe	%ebp
	.cv_fpo_endprologue
Ltmp6:
	.cv_loc	3 1 11 0                        # a.cpp:11:0
	calll	_Foo
	.cv_loc	3 1 12 0                        # a.cpp:12:0
	xorl	%eax, %eax
	popl	%ebp
	retl
Ltmp7:
	.cv_fpo_endproc
Lfunc_end3:
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241
	.long	Ltmp9-Ltmp8                     # Subsection size
Ltmp8:
	.short	Ltmp11-Ltmp10                   # Record length
Ltmp10:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.asciz	"a.s"                           # Object name
	.p2align	2, 0x0
Ltmp11:
	.short	Ltmp13-Ltmp12                   # Record length
Ltmp12:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	7                               # CPUType
	.short	23                              # Frontend version
	.short	1
	.short	6
	.short	0
	.short	23016                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang"                         # Null-terminated compiler version string
	.p2align	2, 0x0
Ltmp13:
Ltmp9:
	.p2align	2, 0x0
	.cv_fpo_data	_SomeOtherCall
	.long	241                             # Symbol subsection for SomeOtherCall
	.long	Ltmp15-Ltmp14                   # Subsection size
Ltmp14:
	.short	Ltmp17-Ltmp16                   # Record length
Ltmp16:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	Lfunc_end0-_SomeOtherCall       # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4098                            # Function type index
	.secrel32	_SomeOtherCall          # Function section relative address
	.secidx	_SomeOtherCall                  # Function section index
	.byte	193                             # Flags
	.asciz	"SomeOtherCall"                 # Function name
	.p2align	2, 0x0
Ltmp17:
	.short	Ltmp19-Ltmp18                   # Record length
Ltmp18:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	4                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	172032                          # Flags (defines frame register)
	.p2align	2, 0x0
Ltmp19:
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
Ltmp15:
	.p2align	2, 0x0
	.cv_linetable	0, _SomeOtherCall, Lfunc_end0
	.cv_fpo_data	_AnotherCall
	.long	241                             # Symbol subsection for AnotherCall
	.long	Ltmp21-Ltmp20                   # Subsection size
Ltmp20:
	.short	Ltmp23-Ltmp22                   # Record length
Ltmp22:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	Lfunc_end1-_AnotherCall         # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4099                            # Function type index
	.secrel32	_AnotherCall            # Function section relative address
	.secidx	_AnotherCall                    # Function section index
	.byte	193                             # Flags
	.asciz	"AnotherCall"                   # Function name
	.p2align	2, 0x0
Ltmp23:
	.short	Ltmp25-Ltmp24                   # Record length
Ltmp24:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	4                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	172032                          # Flags (defines frame register)
	.p2align	2, 0x0
Ltmp25:
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
Ltmp21:
	.p2align	2, 0x0
	.cv_linetable	1, _AnotherCall, Lfunc_end1
	.cv_fpo_data	_Foo
	.long	241                             # Symbol subsection for Foo
	.long	Ltmp27-Ltmp26                   # Subsection size
Ltmp26:
	.short	Ltmp29-Ltmp28                   # Record length
Ltmp28:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	Lfunc_end2-_Foo                 # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4100                            # Function type index
	.secrel32	_Foo                    # Function section relative address
	.secidx	_Foo                            # Function section index
	.byte	193                             # Flags
	.asciz	"Foo"                           # Function name
	.p2align	2, 0x0
Ltmp29:
	.short	Ltmp31-Ltmp30                   # Record length
Ltmp30:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	8                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	172032                          # Flags (defines frame register: local fp reg = EBP)
	.p2align	2, 0x0
Ltmp31:
	.short	Ltmp33-Ltmp32                   # Record length
Ltmp32:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex: int
	.short	0                               # Flags
	.asciz	"xxx"
	.p2align	2, 0x0
Ltmp33:
	.cv_def_range	 Ltmp4 Ltmp5, frame_ptr_rel, -4
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
Ltmp27:
	.p2align	2, 0x0
	.cv_linetable	2, _Foo, Lfunc_end2
	.cv_fpo_data	_mainCRTStartup
	.long	241                             # Symbol subsection for mainCRTStartup
	.long	Ltmp35-Ltmp34                   # Subsection size
Ltmp34:
	.short	Ltmp37-Ltmp36                   # Record length
Ltmp36:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	Lfunc_end3-_mainCRTStartup      # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4102                            # Function type index
	.secrel32	_mainCRTStartup         # Function section relative address
	.secidx	_mainCRTStartup                 # Function section index
	.byte	193                             # Flags
	.asciz	"mainCRTStartup"                # Function name
	.p2align	2, 0x0
Ltmp37:
	.short	Ltmp39-Ltmp38                   # Record length
Ltmp38:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	4                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	172032                          # Flags (defines frame register)
	.p2align	2, 0x0
Ltmp39:
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
Ltmp35:
	.p2align	2, 0x0
	.cv_linetable	3, _mainCRTStartup, Lfunc_end3
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
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
	.long	0x3                             # ReturnType: void
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x0                             # NumParameters
	.long	0x1000                          # ArgListType: ()
	# FuncId (0x1002)
	.short	0x1a                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: void ()
	.asciz	"SomeOtherCall"                 # Name
	.byte	242
	.byte	241
	# FuncId (0x1003)
	.short	0x16                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: void ()
	.asciz	"AnotherCall"                   # Name
	# FuncId (0x1004)
	.short	0xe                             # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: void ()
	.asciz	"Foo"                           # Name
	# Procedure (0x1005)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x0                             # NumParameters
	.long	0x1000                          # ArgListType: ()
	# FuncId (0x1006)
	.short	0x1a                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1005                          # FunctionType: int ()
	.asciz	"mainCRTStartup"                # Name
	.byte	241
