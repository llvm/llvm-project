# RUN: llvm-mc -filetype=asm < %s -triple i686-windows-msvc | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj < %s -triple i686-windows-msvc | llvm-readobj -codeview | FileCheck %s --check-prefix=OBJ

.globl _foo
_foo:
	.cv_fpo_proc _foo 4
	pushl	%ebp
	.cv_fpo_pushreg %ebp
	movl	%ebp, %esp
	.cv_fpo_setframe %ebp
	pushl	%ebx
	.cv_fpo_pushreg %ebx
	pushl	%edi
	.cv_fpo_pushreg %edi
	pushl	%esi
	.cv_fpo_pushreg esi
	subl $20, %esp
	.cv_fpo_stackalloc 20
	.cv_fpo_endprologue

	# ASM: .cv_fpo_proc _foo 4
	# ASM: pushl	%ebp
	# ASM: .cv_fpo_pushreg %ebp
	# ASM: movl	%ebp, %esp
	# ASM: .cv_fpo_setframe %ebp
	# ASM: pushl	%ebx
	# ASM: .cv_fpo_pushreg %ebx
	# ASM: pushl	%edi
	# ASM: .cv_fpo_pushreg %edi
	# ASM: pushl	%esi
	# ASM: .cv_fpo_pushreg %esi
	# ASM: subl $20, %esp
	# ASM: .cv_fpo_stackalloc 20
	# ASM: .cv_fpo_endprologue

	# Clobbers
	xorl %ebx, %ebx
	xorl %edi, %edi
	xorl %esi, %esi
	# Use that stack memory
	leal 4(%esp), %eax
	movl %eax, (%esp)
	calll _bar

	# ASM: calll _bar

	# Epilogue
	# FIXME: Get FPO data for this once we get it for DWARF.
	addl $20, %esp
	popl %esi
	popl %edi
	popl %ebx
	popl %ebp
	retl
	.cv_fpo_endproc

	# ASM: .cv_fpo_endproc

	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.cv_fpo_data _foo
	.cv_stringtable

	# ASM: .cv_fpo_data

# OBJ:       Subsection [
# OBJ-NEXT:    SubSectionType: FrameData (0xF5)
# OBJ-NEXT:    SubSectionSize:
# OBJ-NEXT:    LinkageName: _foo
# OBJ-NEXT:    FrameData {
# OBJ-NEXT:      RvaStart: 0x0
# OBJ-NEXT:      CodeSize: 0x23
# OBJ-NEXT:      LocalSize: 0x0
# OBJ-NEXT:      ParamsSize: 0x4
# OBJ-NEXT:      MaxStackSize: 0x0
# OBJ-NEXT:      FrameFunc: $T0 .raSearch = $eip $T0 ^ = $esp $T0 4 + =
# OBJ-NEXT:      PrologSize: 0x9
# OBJ-NEXT:      SavedRegsSize: 0x0
# OBJ-NEXT:      Flags [ (0x4)
# OBJ-NEXT:        IsFunctionStart (0x4)
# OBJ-NEXT:      ]
# OBJ-NEXT:    }
# OBJ-NEXT:    FrameData {
# OBJ-NEXT:      RvaStart: 0x1
# OBJ-NEXT:      CodeSize: 0x22
# OBJ-NEXT:      LocalSize: 0x0
# OBJ-NEXT:      ParamsSize: 0x4
# OBJ-NEXT:      MaxStackSize: 0x0
# OBJ-NEXT:      FrameFunc: $T0 .raSearch = $eip $T0 ^ = $esp $T0 4 + = $ebp $T0 4 - ^ =
# OBJ-NEXT:      PrologSize: 0x8
# OBJ-NEXT:      SavedRegsSize: 0x4
# OBJ-NEXT:      Flags [ (0x0)
# OBJ-NEXT:      ]
# OBJ-NEXT:    }
# OBJ-NEXT:    FrameData {
# OBJ-NEXT:      RvaStart: 0x3
# OBJ-NEXT:      CodeSize: 0x20
# OBJ-NEXT:      LocalSize: 0x0
# OBJ-NEXT:      ParamsSize: 0x4
# OBJ-NEXT:      MaxStackSize: 0x0
# OBJ-NEXT:      FrameFunc: $T0 $ebp 4 + = $eip $T0 ^ = $esp $T0 4 + = $ebp $T0 4 - ^ =
# OBJ-NEXT:      PrologSize: 0x6
# OBJ-NEXT:      SavedRegsSize: 0x4
# OBJ-NEXT:      Flags [ (0x0)
# OBJ-NEXT:      ]
# OBJ-NEXT:    }
# OBJ-NEXT:    FrameData {
# OBJ-NEXT:      RvaStart: 0x4
# OBJ-NEXT:      CodeSize: 0x1F
# OBJ-NEXT:      LocalSize: 0x0
# OBJ-NEXT:      ParamsSize: 0x4
# OBJ-NEXT:      MaxStackSize: 0x0
# OBJ-NEXT:      FrameFunc: $T0 $ebp 4 + = $eip $T0 ^ = $esp $T0 4 + = $ebp $T0 4 - ^ = $ebx $T0 8 - ^ =
# OBJ-NEXT:      PrologSize: 0x5
# OBJ-NEXT:      SavedRegsSize: 0x8
# OBJ-NEXT:      Flags [ (0x0)
# OBJ-NEXT:      ]
# OBJ-NEXT:    }
# OBJ-NEXT:    FrameData {
# OBJ-NEXT:      RvaStart: 0x5
# OBJ-NEXT:      CodeSize: 0x1E
# OBJ-NEXT:      LocalSize: 0x0
# OBJ-NEXT:      ParamsSize: 0x4
# OBJ-NEXT:      MaxStackSize: 0x0
# OBJ-NEXT:      FrameFunc: $T0 $ebp 4 + = $eip $T0 ^ = $esp $T0 4 + = $ebp $T0 4 - ^ = $ebx $T0 8 - ^ = $edi $T0 12 - ^ =
# OBJ-NEXT:      PrologSize: 0x4
# OBJ-NEXT:      SavedRegsSize: 0xC
# OBJ-NEXT:      Flags [ (0x0)
# OBJ-NEXT:      ]
# OBJ-NEXT:    }
# OBJ-NEXT:    FrameData {
# OBJ-NEXT:      RvaStart: 0x6
# OBJ-NEXT:      CodeSize: 0x1D
# OBJ-NEXT:      LocalSize: 0x0
# OBJ-NEXT:      ParamsSize: 0x4
# OBJ-NEXT:      MaxStackSize: 0x0
# OBJ-NEXT:      FrameFunc: $T0 $ebp 4 + = $eip $T0 ^ = $esp $T0 4 + = $ebp $T0 4 - ^ = $ebx $T0 8 - ^ = $edi $T0 12 - ^ = $esi $T0 16 - ^ =
# OBJ-NEXT:      PrologSize: 0x3
# OBJ-NEXT:      SavedRegsSize: 0x10
# OBJ-NEXT:      Flags [ (0x0)
# OBJ-NEXT:      ]
# OBJ-NEXT:    }
# OBJ-NOT: FrameData
